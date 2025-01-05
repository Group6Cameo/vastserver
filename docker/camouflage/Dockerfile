# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch
ENV PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update && \
  apt-get install -y \
  python3-pip \
  libgl1 \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt \
  && pip3 install torch torchvision \
  numpy scipy opencv-python tensorflow joblib matplotlib pandas \
  albumentations==0.5.2 pytorch-lightning tabulate easydict kornia webdataset \
  packaging gpustat tqdm pyyaml hydra-core scikit-learn tabulate scikit-image

# Copy your application code
COPY . .

# Create necessary directories
RUN mkdir -p /data/checkpoint /data/input /data/output

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]