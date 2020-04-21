"""

Generate model preditions for the test data

"""
import argparse
import collections
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform
from tqdm import tqdm

import metric
import pytorch_retinanet.dataloader
import pytorch_retinanet.model
import pytorch_retinanet.model_dpn
import pytorch_retinanet.model_incresv2
import pytorch_retinanet.model_nasnet_mobile
import pytorch_retinanet.model_pnasnet
import pytorch_retinanet.model_resnet
import pytorch_retinanet.model_se_resnext
import pytorch_retinanet.model_xception
import torch
from config import DATA_DIR, IMG_SIZE, RESULTS_DIR, TEST_DIR, WEIGHTS_DIR
from datasets.test_dataset import TestDataset
from models import MODELS
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from utils.logger import Logger
from utils.my_utils import set_seed

model_configs = MODELS.keys()


def load_model(checkpoint: str) -> nn.Module:
    """
    Helper to load model weihts
    """
    print(f"Loading model from: {checkpoint}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    model = torch.load(checkpoint)
    model = model.to(device)
    model.eval()
    # model = torch.nn.DataParallel(model).cuda()
    return model


def predict_test(model_name: str, fold: int, debug: bool, checkpoints_dir: str, save_oof: bool=True, img_size: int=IMG_SIZE, from_epoch: int=0, to_epoch: int=10):
    """
    Make test predicitons
    
    Args: 
        model_name : string name from the models configs listed in models.py file
        fold: evaluation fold number, 0-3
        debug: if True, runs the debugging on few images 
        epochs: number of epochs to train
        checkpoints_dir: directory with weights 
        from_epoch, to_epoch: the first ad last epochs for predicitions generation  
    """
    # creates directories for predicitons
    predictions_dir = f"{RESULTS_DIR}/test_oof/{model_name}_fold_{fold}"
    os.makedirs(predictions_dir, exist_ok=True)
    print("\n", model_name, "\n")

    # test dataset
    dataset_test = TestDataset(img_size=img_size, debug=debug)
    dataloader_test = DataLoader(dataset_test, num_workers=2, batch_size=4, shuffle=False, collate_fn=pytorch_retinanet.dataloader.collater2d)
    print("{} test images".format(len(dataset_test)))

    for epoch_num in range(from_epoch, to_epoch):
        # load model checkpoint
        checkpoint = f"{checkpoints_dir}/{model_name}_{epoch_num:03}.pt"
        print("load", checkpoint)
        try:
            retinanet = load_model(checkpoint)
        except FileNotFoundError:
            break

        data_iter = tqdm(enumerate(dataloader_test), total=len(dataloader_test))
        oof = collections.defaultdict(list)

        for iter_num, data in data_iter:
            res = retinanet([data["img"].cuda().float(), data["annot"].cuda().float(), data["category"].cuda()], return_loss=False, return_boxes=True)
            nms_scores, global_class, transformed_anchors = res
            if save_oof:
                # predictions
                oof["gt_boxes"].append(data["annot"].cpu().numpy().copy())
                oof["gt_category"].append(data["category"].cpu().numpy().copy())
                oof["boxes"].append(transformed_anchors.cpu().numpy().copy())
                oof["scores"].append(nms_scores.cpu().numpy().copy())
                oof["category"].append(global_class.cpu().numpy().copy())

        if save_oof:  # save predictions
            pickle.dump(oof, open(f"{predictions_dir}/{epoch_num:03}.pkl", "wb"))


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--model", type=str, default="se_resnext101_dr0.75_512", help="String model name from models dictionary")
    arg("--seed", type=int, default=1234, help="Random seed")
    arg("--fold", type=int, default=0, help="Validation fold")
    arg("--checkpoints_dir", type=str, default="../../checkpoints", help="Directory for loading model weights")
    arg("--from-epoch", type=int, default=1, help="Resume training from epoch")
    arg("--to-epochs", type=int, default=15, help="Number of epochs to run")
    arg("--debug", type=bool, default=False, help="If the debugging mode")
    args = parser.parse_args()

    set_seed(args.seed)
    
    weights = f"{args.checkpoints_dir}/{args.model}_fold_{args.fold}/"

    predict_test(
        model_name=args.model, fold=args.fold, debug=args.debug, checkpoints_dir=weights, save_oof=True, img_size=IMG_SIZE, from_epoch=0, to_epoch=10
    )


if __name__ == "__main__":
    main()
