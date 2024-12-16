# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  wget \
  curl \
  unzip \
  libgl1-mesa-glx \
  libglib2.0-0 \
  git \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip3 install --no-cache-dir \
  scikit-survival \
  kornia-rs \
  pytorch-lightning \
  hydra-core \
  webdataset \
  torch \
  torchvision \
  torchaudio \
  torchtext \
  wget \
  numpy \
  albumentations==0.5.2

# Install specific kornia version
RUN pip3 uninstall kornia -y && \
  pip3 install kornia --no-dependencies

# Install OpenCV dependencies correctly
RUN pip3 uninstall opencv-python-headless -y && \
  pip3 install --no-cache-dir \
  opencv-python \
  opencv-contrib-python \
  opencv-python-headless

# Install LaMa requirements
WORKDIR /app/lama
RUN pip3 install -r requirements.txt

# Download and extract the model
# RUN curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip && \
#   unzip big-lama.zip && \
#   rm big-lama.zip

# Download additional weights
RUN curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
  unzip big-lama.zip && \
  rm big-lama.zip

# Set working directory back to /app
WORKDIR /app

# Copy the rest of the application
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]