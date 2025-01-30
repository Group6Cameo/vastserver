# Dockerfile for Camouflage Generation Pipeline
#
# This Dockerfile creates a containerized environment optimized for running the
# camouflage generation pipeline on vast.ai GPU instances. It includes both
# YOLO object detection and LaMa inpainting models in a single container.
#
# Usage on vast.ai:
# 1. Create instance with:
#    - Min 24GB GPU RAM (or 2x12GB)
#    - CUDA 12.4 support
#    - Ubuntu 22.04 base
# 2. Run on the instance:
#    uvicorn app:app --host 0.0.0.0 --port 8000
#
# Environment Variables:
# - DEBIAN_FRONTEND=noninteractive: Prevents interactive prompts during build
# - PYTHONUNBUFFERED=1: Ensures real-time logging
# - TORCH_HOME=/app/.torch: Persistent model storage
# - PYTHONPATH=/app:/app/model/lama: Module resolution paths

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch
ENV PYTHONPATH=/app:/app/model/lama

# # Install system dependencies
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  libgl1-mesa-glx \
  libglib2.0-0 \
  wget \
  git \
  unzip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install numpy first
RUN pip3 install numpy==1.26.0

# Install Python packages for both YOLO and LaMa
RUN pip3 install \
  fastapi \
  uvicorn \
  python-multipart \
  ultralytics \
  opencv-python-headless \
  opencv-python \
  opencv-contrib-python \
  torch torchvision \
  tensorflow \
  scipy \
  joblib \
  matplotlib \
  pandas \
  albumentations==0.5.2 \
  pytorch-lightning==1.2.9 \
  tabulate \
  easydict==1.9.0 \
  kornia==0.5.0 \
  webdataset \
  packaging \
  gpustat \
  tqdm \
  pyyaml \
  hydra-core \
  scikit-image \
  scikit-learn \
  basicsr \
  realesrgan \
  gfpgan



# Create necessary directories
RUN mkdir -p /app/data /app/model/YOLO/weights /app/surroundings_data /app/output /app/model/big-lama /app/models

# Clone and setup LaMa
RUN git clone https://github.com/Group6Cameo/lama.git /app/model/lama && \
  cd /app/model/lama && \
  wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --continue --tries=0 https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
  unzip big-lama.zip && \
  rm big-lama.zip && \
  cd /app/model/ && \
  wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --continue --tries=0 https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
  unzip big-lama.zip && \
  rm big-lama.zip && \
  cd .. \
  wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --continue --tries=0 https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/models/RealESRGAN_x4plus.pth

# RUN pip install -r /app/model/lama/requirements.txt

# Fix for RealESRGAN
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /opt/conda/lib/python3.11/site-packages/basicsr/data/degradations.py

# Copy YOLO files
COPY ./model/YOLO/detect.py /app/model/YOLO/
COPY ./model/YOLO/weights/best.pt /app/model/YOLO/weights/

# Copy your application code
COPY ./app.py /app/
COPY ./model/interface.py /app/model/
COPY ./model/utils/camouflage_utils.py /app/model/utils/

COPY ./surroundings_data/ /app/surroundings_data/

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]