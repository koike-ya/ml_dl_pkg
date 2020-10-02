FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace/

# install basics
RUN apt-get update -y
RUN apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop sox libsox-dev libsox-fmt-all vim

# install python deps
RUN pip install cython visdom cffi tensorboardX wget jupyter

# install apex
RUN git clone --recursive https://github.com/NVIDIA/apex.git
RUN cd apex; pip install .

# install deepspeech.pytorch
ADD . /workspace/ml_pkg
RUN cd ml_pkg; pip install -r requirements.txt && pip install -e .

# launch jupyter
ENTRYPOINT /bin/bash
