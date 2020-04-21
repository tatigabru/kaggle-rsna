"""
Created on Thu Aug  8 16:14:23 2019

@author: Tanya
"""
import argparse
import collections
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metric
from config import IMG_SIZE, RESULTS_DIR, TEST_DIR, WEIGHTS_DIR
from utils.my_utils import set_seed


def load_oof(results_dir: str, model_name: str, fold: int, epoch_num: int) -> np.ndarray:
    """
    Helper,
    Loads oot-of-fold predictions from pickle
    """
    predictions_dir = f"{results_dir}/oof/{model_name}_fold_{fold}"
    fn = f"{predictions_dir}/{epoch_num:03}.pkl"
    print(f"Loading preds from {fn}")
    oof = pickle.load(open(fn, "rb"))
    return oof


def p1p2_to_xywh(p1p2: np.ndarray) -> np.ndarray:
    """
    Helper function
    converts box coordinates to 
    x0, y0, width, height format
    """
    xywh = np.zeros((p1p2.shape[0], 4))
    xywh[:, :2] = p1p2[:, :2]
    xywh[:, 2:4] = p1p2[:, 2:4] - p1p2[:, :2]
    return xywh


def get_epoch_metric(oof: np.ndarray, thresholds: list, predictions_dir: str, epoch_num: int) -> list:
    """
    Loads epoch predictions and
    calculates the epoch metric for a set of thresholds

    Args: 
        oof            : out-of-fold predictions
        thresholds     : list of thresholds
        predictions_dir: directory for saving scores
        epoch_num      : current epoch  for metric calculation        
    """
    epoch_scores = []
    nb_images = len(oof["scores"])
    # check range of thresholds
    for threshold in thresholds:
        threshold_scores = []
        for img_id in range(nb_images):
            gt_boxes = oof["gt_boxes"][img_id][0].copy()
            boxes = oof["boxes"][img_id].copy()
            scores = oof["scores"][img_id].copy()
            category = oof["category"][img_id]
            category = np.exp(category[0, 2])

            if len(scores):
                scores[scores < scores[0] * 0.5] = 0.0
            mask = scores * 5 > threshold

            if gt_boxes[0, 4] == -1.0:
                if np.any(mask):
                    threshold_scores.append(0.0)
            else:
                if len(scores[mask]) == 0:
                    score = 0.0
                else:
                    score = metric.map_iou(boxes_true=p1p2_to_xywh(gt_boxes), boxes_pred=p1p2_to_xywh(boxes[mask]), scores=scores[mask])
                threshold_scores.append(score)
        print("threshold {}, score {}".format(threshold, np.mean(threshold_scores)))
        epoch_scores.append(np.mean(threshold_scores))
    best_score = np.max(epoch_scores)
    print(f"epoch {epoch_num}, best score: {best_score}")

    return epoch_scores


def get_metrics(predictions_dir: str, model_name: str, fold: int, epochs: int, thresholds: list, save_metrics: bool=False) -> None:
    """
    Loads predicitons and
    calculates the metric for a set of thresholds for all epochs

    Args:         
        predictions_dir: directory for saving scores
        model_name     : string of model name
        fold           : validation fold
        epochs         : total number of epochs
        thresholds     : list of thresholds
        save_metrics   : boolean flag, if save all scores in pickle 
    """
    all_scores = []
    for epoch_num in range(1, epochs + 1):
        oof = load_oof(predictions_dir, model_name, fold, epoch_num)
        epoch_scores = get_epoch_metric(oof, thresholds, predictions_dir, epoch_num)
        all_scores.append(epoch_scores)

    best_score = np.max(all_scores)
    epochs = np.arange(1, epochs + 1)
    print("best score: ", best_score)

    plt.figure(1)
    plt.imshow(np.array(all_scores))
    plt.show()

    if save_metrics:
        scores_dir = f"{predictions_dir}/scores/{model_name}_fold_{fold}"
        os.makedirs(scores_dir, exist_ok=True)
        print("all scores.shape: {}, thresholds {}, epochs {}".format(np.array(all_scores).shape, thresholds, epochs))
        metric_scores = collections.defaultdict(list)
        metric_scores["scores"] = np.array(all_scores)
        metric_scores["tresholds"] = thresholds
        metric_scores["epochs"] = epochs
        pickle.dump(metric_scores, open(f"{scores_dir}/all_scores.pkl", "wb"))


def main() -> None:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--model-name", type=str, default="resnet101_512", help="String model name from models dictionary")
    arg("--results_dir", type=str, default="'../../output'", help="Directory for loading predictions")
    arg("--seed", type=int, default=1234, help="Random seed")
    arg("--fold", type=int, default=0, help="Validation fold")
    args = parser.parse_args()
    set_seed(args.seed)

    model_names = [
        "se_resnext101_dr0.75_512",
        "se_resnext101_dr_512",
        "se_resnext101_dr_512_without_pretrained",
        "se_resnext50_512_dr0.8",
        "se_resnext50_512",
        "resnet101_512",
        "resnet50_512",
    ]

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]

    get_metrics(
        predictions_dir = RESULTS_DIR, 
        model_name=args.model_name, 
        fold=args.fold, 
        epochs=19, 
        thresholds=thresholds, 
        save_metrics=True
        )


if __name__ == "__main__":
    main()
