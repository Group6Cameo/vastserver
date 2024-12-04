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
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY model/init/setup.py .
RUN pip3 install --no-cache-dir -e .

# Download and extract the model (as done in setup.py)
RUN curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip \
  && unzip big-lama.zip \
  && rm big-lama.zip

# Copy the rest of the application
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]