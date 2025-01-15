FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch
ENV PYTHONPATH="/app:/app/model/lama:${PYTHONPATH}"

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
  albumentations \
  pytorch-lightning \
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
  scikit-learn

RUN pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data /app/model/YOLO/weights /app/surroundings_data /app/output /app/model/big-lama

# Clone and setup LaMa
RUN git clone https://github.com/Group6Cameo/lama.git /app/model/lama && \
  cd /app/model/lama && \
  wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
  unzip big-lama.zip && \
  rm big-lama.zip

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
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]