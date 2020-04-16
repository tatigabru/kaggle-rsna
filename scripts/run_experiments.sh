#!/usr/bin/env bash
CUR_DIR=$pwd

# Download dataset
bash download_data.sh

# Run model training in debug mode
python src/train_runner.py --action train --debug True

# Run full model training
python src/train_runner.py --action train --model se_resnext101_dr0.75_512 --debug False --num-epochs 16

# Test model
python src/train_runner.py --action test_model --model se_resnext101_dr0.75_512 --debug True --epoch 12

# Generate and save oof predictions
python src/train_runner.py --action generate_predictions --model se_resnext101_dr0.75_512 --debug False --num-epochs 16

