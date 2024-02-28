FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update \
 && apt install -y software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt update \
 && apt install -y python3.10 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 100 \
 && apt install -y build-essential python3-pip python3.10-dev libgl1 vim curl \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

CMD ["sleep", "infinity"]