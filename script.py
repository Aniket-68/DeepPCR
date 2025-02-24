import os
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib.pyplot as plt
import psutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Start Horovod (data parallelism for parallel process only)
hvd.init()

# Set up GPU (RTX 4060)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[hvd.local_rank() % len(gpus)], 'GPU')
    logger.info(f"Worker {hvd.rank()}: GPU detected - {gpus[hvd.local_rank() % len(gpus)]}")
else:
    logger.warning(f"Worker {hvd.rank()}: No GPU, using CPU")

# Data setup
data_dir = "/data/poultry_fecal_images"  # Docker mount path
img_height, img_width = 224, 224
batch_size = 16 // max(1, hvd.size())  # Split across workers for parallel process
classes = ['healthy', 'cocci', 'salmo']

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    brightness_range=[0.8, 1.2], shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    validation_split=0.15
)
test_datagen = ImageDataGenerator(rescale=1./255)

try:
    train_generator = train_datagen.flow_from_directory(
        data_dir, target_size=(img_height, img_width), batch_size=batch_size,
        class_mode='categorical', subset='training', classes=classes, shuffle=True
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir, target_size=(img_height, img_width), batch_size=batch_size,
        class_mode='categorical', subset='validation', classes=classes, shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        data_dir, target_size=(img_height, img_width), batch_size=batch_size,
        class_mode='categorical', classes=classes, shuffle=False
    )
    logger.info(f"Worker {hvd.rank()}: Found {train_generator.samples} training images")
except Exception as e:
    logger.error(f"Worker {hvd.rank()}: Failed to load dataset - {str(e)}")
    raise

# Unified model for sequential process (no model parallelism)
def create_sequential_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(classes), activation='softmax')
    ])
    return model

# Model parts for DeepPCR-inspired model parallelism (parallel process only)
def create_lower_model():
    inputs = Input(shape=(img_height, img_width, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    return Model(inputs=inputs, outputs=x, name="lower_model")

def create_upper_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(128, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(classes), activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs, name="upper_model")

# Sequential training (no model parallelism)
def train_sequential_cnn():
    model = create_sequential_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    train_time = time.time() - start_time
    loss, accuracy = model.evaluate(test_generator)
    return train_time, accuracy

# Define training step with @tf.function for parallel process
@tf.function(reduce_retracing=True)
def train_step(lower_model, upper_model, batch_x, batch_y, optimizer_lower, optimizer_upper):
    with tf.GradientTape() as lower_tape:
        lower_output = lower_model(batch_x, training=True)
        loss = tf.keras.losses.categorical_crossentropy(batch_y, upper_model(lower_output))
    lower_grads = lower_tape.gradient(loss, lower_model.trainable_variables)
    optimizer_lower.apply_gradients(zip(lower_grads, lower_model.trainable_variables))
    
    with tf.GradientTape() as upper_tape:
        upper_output = upper_model(lower_output, training=True)
        loss = tf.keras.losses.categorical_crossentropy(batch_y, upper_output)
    upper_grads = upper_tape.gradient(loss, upper_model.trainable_variables)
    optimizer_upper.apply_gradients(zip(upper_grads, upper_model.trainable_variables))

# DeepPCR-inspired hybrid training (data + model parallelism)
def train_deeppcr_hybrid_cnn(epochs=10):
    lower_model = create_lower_model()
    upper_input_shape = lower_model.output_shape[1:]
    upper_model = create_upper_model(upper_input_shape)
    
    # Horovod for data parallelism
    optimizer_lower = tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size())
    optimizer_upper = tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size())
    optimizer_lower = hvd.DistributedOptimizer(optimizer_lower)
    optimizer_upper = hvd.DistributedOptimizer(optimizer_upper)
    
    lower_model.compile(optimizer=optimizer_lower, loss='categorical_crossentropy')
    upper_model.compile(optimizer=optimizer_upper, loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    sync_start = time.time()
    for epoch in range(epochs):
        train_generator.reset()
        for batch_x, batch_y in train_generator:
            batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
            batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)
            train_step(lower_model, upper_model, batch_x, batch_y, optimizer_lower, optimizer_upper)
            if train_generator.batch_index >= train_generator.samples // batch_size:
                break
        # Horovod syncs weights (data parallelism)
        hvd.broadcast(lower_model.variables, root_rank=0)
        hvd.broadcast(upper_model.variables, root_rank=0)
    
    sync_time = time.time() - sync_start - (time.time() - start_time)  # Communication cost
    train_time = time.time() - start_time
    
    # Evaluation
    test_generator.reset()
    total_loss, total_acc, count = 0, 0, 0
    for batch_x, batch_y in test_generator:
        batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)
        lower_out = lower_model(batch_x, training=False)
        preds = upper_model(lower_out, training=False)
        total_loss += tf.keras.losses.categorical_crossentropy(batch_y, preds).numpy().mean()
        total_acc += tf.keras.metrics.categorical_accuracy(batch_y, preds).numpy().mean()
        count += 1
        if test_generator.batch_index >= test_generator.samples // batch_size:
            break
    loss, accuracy = total_loss / count, total_acc / count
    return train_time, accuracy, sync_time

# Run and measure
if hvd.rank() == 0:
    logger.info("Running Sequential Method (One Worker)...")
    seq_time, seq_accuracy = train_sequential_cnn()
else:
    seq_time, seq_accuracy = None, None

hvd.broadcast(tf.constant(0), root_rank=0)  # Sync before parallel run

logger.info(f"Worker {hvd.rank()}: Starting DeepPCR-Inspired Hybrid Method...")
par_time, par_accuracy, sync_time = train_deeppcr_hybrid_cnn()

# Results (rank 0 reports)
if hvd.rank() == 0:
    speedup = seq_time / par_time if seq_time and par_time else 0
    logger.info(f"Sequential Time: {seq_time:.2f} seconds, Accuracy: {seq_accuracy:.4f}")
    logger.info(f"Parallel Time (DeepPCR-Inspired Hybrid): {par_time:.2f} seconds, Accuracy: {par_accuracy:.4f}")
    logger.info(f"Speedup with DeepPCR Inspiration: {speedup:.2f}x")
    logger.info(f"Estimated Communication Cost (Sync Time): {sync_time:.2f} seconds")
    
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    logger.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
    
    methods = ['Sequential', 'Parallel (DeepPCR Hybrid)']
    times = [seq_time, par_time]
    accuracies = [seq_accuracy, par_accuracy]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(methods, times, color=['blue', 'green'])
    plt.title('Time Comparison')
    plt.ylabel('Time (seconds)')
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, accuracies, color=['blue', 'green'])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('/data/output.png')
    plt.show()