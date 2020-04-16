#!/usr/bin/env bash

pip install --upgrade pip
CUR_DIR=$pwd
DATA_DIR_LOC=dataset

cd ..
mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC

if [ "$(ls -A $(pwd))" ]
then
    echo "$(pwd) not empty!"
else
    echo "$(pwd) is empty!"
    pip install kaggle --upgrade
    kaggle competitions download -c rsna-pneumonia-detection-challenge
    unzip rsna-pneumonia-detection-challenge.zip
    mkdir train
    mkdir test
    unzip stage_2_train_images.zip -d train
    unzip stage_2_test_images.zip -d test
    rm rsna-pneumonia-detection-challenge.zip
    rm stage_2_train_images.zip
    rm stage_2_test_images.zip
fi

cd $CUR_DIR
echo $(pwd)
