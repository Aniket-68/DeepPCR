FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    matplotlib==3.8.3 \
    psutil==5.9.8 \
    scikit-build \
    scipy==1.11.4 \
    horovod==0.28.1
COPY script.py /app/
ENV PYTHONUNBUFFERED=1
# CMD ["mpirun", "--allow-run-as-root", "-np", "2", "python3", "/app/script.py"]
CMD [   "python3", "/app/script.py"]