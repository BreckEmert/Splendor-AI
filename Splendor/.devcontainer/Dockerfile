# Splendor/.devcontainer/Dockerfile

# Official TF-gpu runtime as a parent image
FROM tensorflow/tensorflow:2.16.1-gpu

# Set environment variables for CUDA and cuDNN
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

ENV TF_FORCE_GPU_ALLOW_GROWTH=true


# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    scipy \
    openpyxl \
    pillow \
    pygame \
    tf2onnx \
    keras \
    debugpy

# Install game viz dependencies
RUN apt-get update && apt-get install -y \
    fontconfig && \
    fc-cache -f -v && \
    apt-get clean

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libsdl2-2.0-0 \
        libgl1 libglx-mesa0 \
        libgl1-mesa-dri libglu1-mesa \
        libx11-6 libxext6 libxrender1 \
        libxrandr2 libxinerama1 libxcursor1 libxfixes3 libxi6 \
        mesa-utils && \
    rm -rf /var/lib/apt/lists/*


# Create a non-root user and set permissions
RUN useradd -m -s /bin/bash myuser && \
    mkdir -p /workspace && \
    chown -R myuser:myuser /workspace && \
    ldconfig && \
    chown root:1000 /etc /etc/ld.so.cache && \
    chmod g+w /etc /etc/ld.so.cache
USER myuser
WORKDIR /workspace
