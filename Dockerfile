FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ENV   TZ=Asia/Shanghai

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade -i https://pypi.doubanio.com/simple/ " && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    openssh-client \
    openssh-server \
    autossh \
    git \
    libopencv-dev \
    python3-tk \
    bash-completion \
    time \
    vim \
    unzip \
    tmux && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    software-properties-common 

RUN mkdir ~/.pip && echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url=https://pypi.doubanio.com/simple/" >> ~/.pip/pip.conf && \
    echo "format = columns" >> ~/.pip/pip.conf

RUN pip install --default-timeout=100 numpy \
    h5py \
    tqdm \
    cython \
    ipython \
    scikit-image \
    packaging \
    urllib3 \
    scipy \
    matplotlib \
    opencv-python \
    yacs\
    pycocotools \
    gpustat \
    pytorch-fid \
    natsort \
    imgaug \
    scikit-learn
        

RUN ldconfig && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN echo "root:ustc" | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config &&\
    echo PubkeyAuthentication yes &&\
    echo PermitRootLogin yes

RUN /etc/init.d/ssh restart
RUN groupadd --gid 1001 wangph \
  && useradd --home-dir /home/wangph --create-home --uid 1010 \
    --gid 1001 --shell /bin/bash --skel /dev/null wangph

ENV LC_ALL=C.UTF-8
USER wangph
RUN mkdir ~/.pip && echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url=https://pypi.doubanio.com/simple/" >> ~/.pip/pip.conf && \
    echo "format = columns" >> ~/.pip/pip.conf

WORKDIR /home/wangph
