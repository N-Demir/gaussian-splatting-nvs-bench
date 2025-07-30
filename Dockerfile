FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Set Torch CUDA Compatbility to be for RTX 4090, T4, and A100
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.9;9.0"

# Install git and various other helper dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    git \
    wget \
    unzip \
    cmake \
    build-essential \
    ninja-build \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    && rm -rf /var/lib/apt/lists/*

# TODO: May need `mkdir -p /run/sshd`

WORKDIR /root/workspace


###### Project Specific Installation ######


RUN git clone https://github.com/N-Demir/gaussian-splatting-nvs-leaderboard.git . --recursive

# Install (avoid conda installs because they don't work well in dockerfile situations)
# Separating these on separate lines helps if there are errors (previous lines will be cached) especially on the large package installs
RUN pip install plyfile
RUN pip install tqdm
RUN pip install opencv-python
RUN pip install joblib
RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn
RUN pip install submodules/fused-ssim