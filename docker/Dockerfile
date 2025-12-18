FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    libsm6 \
    curl \
    libxext6 \
    wget \
    pkg-config \
    zip  \
    g++  \
    zlib1g-dev  \
    unzip  \
    python3  \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/root

USER root

SHELL ["/bin/bash", "-c"]

RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_25.1.1-2-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate base \
    && conda update -n base conda -y \
    && conda install python=3.9 \
    && python --version

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda init bash \
    && conda config --set auto_activate_base true

# ENV settings
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV MMCV_WITH_OPS=1
ENV MMCV_CUDA_ARGS=-std=c++17

RUN conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge --yes

RUN mkdir /deps

RUN cd /deps \
    && git clone https://github.com/open-mmlab/mmcv.git  \
    && cd mmcv \
    && git checkout v1.6.0 \
    && pip install -e .

RUN pip install mmdet==2.26.0 mmsegmentation==0.29.1

RUN cd /deps \
    && git clone https://github.com/open-mmlab/mmdetection3d.git  \
    && cd mmdetection3d \
    && git checkout v1.0.0rc6 \
    && pip install -e .

# manual version fixes
RUN pip install yapf==0.32

RUN pip install waymo-open-dataset-tf-2-12-0

# install waymo metric
WORKDIR /deps/
ARG BAZEL_VERSION=5.4.0
RUN git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
WORKDIR /deps/waymo-od/src
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
RUN bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
RUN bazel clean
RUN bazel build waymo_open_dataset/metrics/tools/compute_detection_let_metrics_main
RUN bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main 
RUN cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_let_metrics_main /deps/mmdetection3d/mmdet3d/core/evaluation/waymo_utils
RUN cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main /deps/mmdetection3d/mmdet3d/core/evaluation/waymo_utils

RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# add custom ops for IoU Calculation
COPY ops /deps/ops 
RUN cd /deps/ops && pip install .

WORKDIR /root

# final installs
RUN pip install wandb seaborn fvcore
RUN pip install setuptools==59.5.0 numpy==1.23.5
