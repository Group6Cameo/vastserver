FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install ultralytics

COPY ./detect.py .

CMD ["python3", "detect.py"]