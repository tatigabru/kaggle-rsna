#!/bin/bash

conda create -y -n rsna2 python=3.6
conda activate rsna2

conda install -y -n rsna pytorch=0.4.1 cuda90 -c pytorch
pip install --upgrade pip
pip install -r requirements.txt
conda install -y -c conda-forge pycocotools