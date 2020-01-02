#!/usr/bin/env bash
CUR_DIR=$pwd

# Download dataset
bash download_data.sh

# Run model training in debug mode
python src/train.py --action train --debug True

# Run full model training
python src/train.py --action train --debug False

# Plot predictions
python src/train.py --action test_model --debug True --epoch 12

# Check metric score
python src/train.py --action check_metric --debug False

# Generate and save oof predictions
python src/train.py --action generate_predictions --debug True

