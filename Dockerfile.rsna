FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

MAINTAINER Tati Gabru, Version 1.1

# Common arguments
ARG env_name=rsna
ARG HOST_USER
ARG PYTHON_VERSION=3.6

# System set up
RUN apt-get update --fix-missing && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
	     unzip \
	     zip \
	     locales \
	     emacs \
	     libgl1-mesa-glx \
	     openssh-server \
	     screen \	 	  
	     libturbojpeg \
	     rsync \
         tmux \
         wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*     


# Install Miniconda, fragments from: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/Dockerfile
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda update -n base conda

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install needed pytorch version and pycocotools with anaconda
RUN conda install -y -n env_name pytorch=0.4.1 cuda90 -c pytorch && \
    conda install -y -c conda-forge pycocotools