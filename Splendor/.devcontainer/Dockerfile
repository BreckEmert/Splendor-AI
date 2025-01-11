# Official TF-gpu runtime as a parent image
FROM tensorflow/tensorflow:2.16.1-gpu

# Set environment variables for CUDA and cuDNN
ENV LD_LIBRARY_PATH /usr/local/cuda-12.3/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Install Python packages
RUN pip install --no-cache-dir numpy pandas scipy openpyxl pillow

# Font for visualization
RUN apt-get update && apt-get install -y \
    fontconfig \
    && fc-cache -f -v


# Create a non-root user and set permissions
RUN useradd -m -s /bin/bash myuser && \
    mkdir -p /workspace && \
    chown -R myuser:myuser /workspace

# Switch to the non-root user
USER myuser

# Set the working directory for the user
WORKDIR /workspace
