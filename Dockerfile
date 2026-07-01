FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN sed -i 's@http://archive.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu@g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir jittor

WORKDIR /workspace
