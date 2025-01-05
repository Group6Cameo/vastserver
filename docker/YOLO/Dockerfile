FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install required packages
RUN pip3 install ultralytics opencv-python-headless numpy

# Create model directory and data directory
RUN mkdir -p /app/data /app/model/YOLO/weights

# Copy your detection script and model weights
COPY ./detect.py /app/
COPY ./weights/best.pt /app/model/YOLO/weights/

# Make sure the data directory exists
RUN mkdir -p /app/data

# Set the entrypoint to wait for input
ENTRYPOINT ["python3", "detect.py"]