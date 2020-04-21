"""

Training models and inference

"""
import argparse
import os
import pickle
import sys
#sys.path.append("/home/user/rsna/progs/rsna-repo/src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from .. config import IMG_SIZE, RESULTS_DIR, TEST_DIR, WEIGHTS_DIR
from .. utils.my_utils import set_seed


def load_scores(model_name: str, fold: int, output_dir: str=RESULTS_DIR) -> tuple:
    """
    Load scores from a pickle file
    Args: 
        model_name: string model name
        fold      : fold number to load
        output_dir: string, directory with outputs        
    """
    scores_dir = f"{output_dir}/scores/{model_name}_fold_{fold}"
    scores_file = f"{scores_dir}/all_scores.pkl"
    print("Loading scores from {}".format(scores_file))
    try:
        metrics = pickle.load(open(scores_file, "rb"))
    except:
        pass
    all_scores = metrics["scores"].copy()
    thresholds = metrics["tresholds"].copy()
    epochs = metrics["epochs"].copy()

    return all_scores, thresholds, epochs


def scores_heatmap(all_scores: np.ndarray, thresholds: np.ndarray, epochs: np.ndarray, model_name: str, output_dir: str, if_save: bool=False) -> None:
    """
    Plot all scores heatmap
    Args: 
        all_scores: array of mAP scores
        thresholds: array of thresholds 
        epochs    : array of epochs 
        model_name: string model name
        output_dir: string, directory with outputs for saving pics
        if_save   : saves figure if_save=True
    """
    # plt.imshow(all_scores, cmap='hot')
    cmap = plt.get_cmap("PiYG")
    levels = MaxNLocator(nbins=15).tick_values(all_scores.min(), all_scores.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    plt.imshow(all_scores, cmap="hot", norm=norm)
    if if_save:  # save figure
        plt.savefig(f"{output_dir}/pics/{model_name}_maps_2d.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


def map_2d(all_scores: np.ndarray, thresholds: np.ndarray, epochs: np.ndarray, model_name: str, output_dir: str, if_save: bool=False) -> None:
    """
    Plot all scores heatmap
    Args: 
        all_scores: array of mAP scores
        thresholds: array of thresholds 
        epochs    : array of epochs 
        model_name: string model name
        output_dir: string, directory with outputs for saving pics
        if_save   : saves figure if_save=True
    """
    # plt.imshow(all_scores, cmap='hot')
    X, Y = thresholds, epochs
    plt.pcolormesh(X, Y, all_scores)
    plt.ylabel("Epochs", fontsize=20)
    plt.xlabel("Thresholds", fontsize=20)
    if if_save:  # save figure
        plt.savefig(f"{output_dir}/pics/{model_name}_maps_2d.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_best_scores(all_scores: np.ndarray, thresholds: np.ndarray, epochs: np.ndarray, model_name: str, output_dir: str, if_save: bool=False) -> None:
    """
    Plot the best_scores per epoch
    Args: 
        all_scores: array of mAP scores
        thresholds: array of thresholds 
        epochs    : array of epochs 
        model_name: string model names for legend 
        output_dir: string, directory with outputs for saving pics
        if_save   : saves figure if_save=True
    """
    # find the best score per epoch
    best_scores = np.max(all_scores, axis=1)
    print(best_scores)
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, best_scores, "o--k", label=model_name)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("mAP", fontsize=20)
    plt.legend(loc="upper left", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.grid(True)
    plt.show()
    if if_save:  # save figure
        plt.savefig(f"{output_dir}/pics/{model_name}_bestmap.eps", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.savefig(f"{output_dir}/pics/{model_name}_bestmap.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.savefig(f"{output_dir}/pics/{model_name}_bestmap.png", dpi=300, bbox_inches="tight", pad_inches=0)


def plot_score(all_scores: np.ndarray, thresholds: np.ndarray, epochs: np.ndarray, model_name: str, output_dir: str, if_save: bool=False) -> None:
    """
    Plot scores per epoch for fixed threshold
    Args: 
        all_scores: array of mAP scores
        thresholds: array of thresholds 
        epochs    : array of epochs 
        model_name: string model names for legend 
        output_dir: string, directory with outputs for saving pics
        if_save   : saves figure if_save=True
    """
    # find the best per epoch
    thres = thresholds[10]
    print(thres)
    scores = all_scores[:, 10]  # 0.55
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, scores, "o--k", label=model_name)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("mAP", fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.title(f"Threshold {thres}")
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.grid(True)
    if if_save:  # save figure
        # plt.savefig(f"{output_dir}/pics/{model_name}_map_thres_{thres}.eps", dpi=600, bbox_inches = 'tight',
        # pad_inches = 0)
        # plt.savefig(f"{output_dir}/pics/{model_name}_map_thres_{thres}.pdf", dpi=300, bbox_inches = 'tight',
        # pad_inches = 0)
        plt.savefig(f"{output_dir}/pics/{model_name}_map_thres_{thres}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_scores(all_scores: np.ndarray, thresholds: np.ndarray, epochs: np.ndarray, model_names: list, labels: list, output_dir: str, if_save: bool=False) -> None:
    """
    Plot validation losses per epoch
    Args: 
        all_scores: array of mAP scores
        thresholds: array of thresholds 
        epochs    : array of epochs 
        run_loss  : array with run results
        model_name: string model names for legend
        output_dir: string, directory with outputs for saving pics 
        if_save   : saves figure if_save=True
    """
    plt.figure(figsize=(10, 7))
    colors = ["k", "r", "b", "g", "m", "c", "y"]
    thres = thresholds[10]
    print(thres)
    for num, model in enumerate(model_names):
        all_scores, thresholds, epochs = load_scores(model_name=model, fold=0, output_dir=output_dir)
        scores = all_scores[:, 10]  # 0.35
        plt.plot(epochs, scores, "-{}".format(colors[num]), label=labels[num])

    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("mAP", fontsize=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.title(f"Threshold {thres}")
    plt.tick_params(axis="both", which="major", labelsize=20)
    # plt.xticks(np.arange(0, 20, step=2))
    plt.grid(True)
    if if_save:
        # plt.savefig(f"{output_dir}/pics/maps_thres_{thres}.eps", dpi=600, bbox_inches = 'tight',
        # pad_inches = 0)
        # plt.savefig(f"{output_dir}/pics/maps_thres_{thres}.pdf", dpi=300, bbox_inches = 'tight',
        # pad_inches = 0)
        plt.savefig(f"{output_dir}/pics/maps_thres_{thres}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--seed", type=int, default=1234, help="Random seed")
    arg("--fold", type=int, default=0, help="Validation fold")
    arg("--output_dir", type=str, default="../../output", help="Directory for loading results")
    arg("--save-pics", type=bool, default=True)
    arg("--debug", type=bool, default=False, help="If the debugging mode")
    args = parser.parse_args()
    set_seed(args.seed)

    RESULTS_DIR = "../../../../output/"

    model_names = [
        "se_resnext101_dr0.75_512",
        "se_resnext101_dr_512",
        "se_resnext101_512",
        "se_resnext50_512_dr0.8",
        "se_resnext50_512",
        "resnet101_512",
        "inc_resnet_v2_512",
    ]
    labels = model_names

    model = model_names[0]
    all_scores, thresholds, epochs = load_scores(model_name=model, fold=args.fold, output_dir=RESULTS_DIR)

    scores_heatmap(all_scores, thresholds, epochs, model, RESULTS_DIR)
    # plot_best_scores(all_scores, thresholds, epochs,
    #                 model_name=model, output_dir = RESULTS_DIR, if_save=False)

    map_2d(all_scores, thresholds, epochs, model, RESULTS_DIR, if_save=False)

    plot_score(all_scores, thresholds, epochs, output_dir=RESULTS_DIR, model_name=model, if_save=True)

    plot_scores(all_scores, thresholds, epochs, model_names, labels, RESULTS_DIR, if_save=True)


if __name__ == "__main__":
    main()
