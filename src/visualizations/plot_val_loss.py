# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:14:23 2019

@author: Tanya
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_val_loss(model_name: str, results_dir: str) -> pd.DataFrame:
    """Load validation loss from runs
    """
    loss_file = f"{results_dir}/run-{model_name}_fold_0-tag-loss_valid_regression.csv"
    run_loss = pd.read_csv(loss_file)

    return run_loss


def plot_val_loss(run_loss: pd.DataFrame, model_name: str, if_save: bool=False) -> None:
    """
    Plot validation loss per epoch
    Input: 
        run_loss  : dataframe run results
        model_name: string model names for legend 
        if_save   : saves figure if_save=True
    """
    val_loss = run_loss.Value.values
    epochs = run_loss.Step.values
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, val_loss, "o--k", label=model_name)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Validation Loss", fontsize=20)
    plt.legend(loc="upper left", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.grid(True)
    plt.show()
    if if_save:  # save figure
        plt.savefig("fig.eps", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.savefig("fig.pdf", dpi=300, bbox_inches="tight", pad_inches=0)


def plot_val_losses(model_names: list, labels: list, results_dir: str, if_save: bool=False) -> None:
    """
    Plot validation losses per epoch
    Input: 
        run_loss  : array with run results
        model_name: string model names for legend 
        if_save   : saves figure if_save=True
    """
    plt.figure(figsize=(10, 7))
    colors = ["k", "r", "b", "g", "m", "c", "y"]
    for num, model in enumerate(model_names):
        run_loss = load_val_loss(model_name=model, results_dir=results_dir)
        val_loss = run_loss.Value.values
        epochs = run_loss.Step.values
        plt.plot(epochs, val_loss, "-{}".format(colors[num]), label=labels[num])

    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Validation Loss", fontsize=20)
    plt.legend(loc="upper right", fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.xlim(0, 20)
    plt.ylim(0.133, 0.165)
    plt.xticks(np.arange(0, 20, step=2))
    plt.grid(True)
    plt.show()
    if if_save:  # save figure
        plt.savefig("runs.eps", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.savefig("runs.png", dpi=300, bbox_inches="tight", pad_inches=0)


def main() -> None:
    RESULTS_DIR = "../../output/runs"

    model_names = [
        "xception_512_dr",
        "se_resnext101_dr0.75_512",
        "inc_resnet_v2_512_dr",
        "nasnet_mobile_512",
        "pnas_512_dr_075",
        "resnet101_512",
        #'dpn92_512_dr'
    ]

    labels = [
        "Xception",
        "SE_ResNext-101",
        "Inception-ResNet-v2",
        "NASNnet-A-Mobile",
        "PNASNet",
        "ResNet-101",
        #'DPN92'
    ]

    seresnext = [
        "se_resnext101_512_",
        "se_resnext101_dr_512",
        "se_resnext101_dr0.75_512_3",
        "se_resnext101_dr_512_without_pretrained",
        # 'se_resnext50_512_dr0.8',
        # 'se_resnext50_512',
    ]

    labels_se = [
        "no dropout",
        "droupout 0.5",
        "droupout 0.75",
        "no pretrain",
    ]

    model = model_names[0]
    run_loss = load_val_loss(model_name=model, results_dir=RESULTS_DIR)

    plot_val_loss(run_loss, model_name=model, if_save=False)

    plot_val_losses(model_names=seresnext, labels=labels_se, results_dir=RESULTS_DIR, if_save=True)

    # plot_val_losses(model_names = model_names,
    #           labels = labels,
    #           results_dir = RESULTS_DIR,
    #           if_save = True)


if __name__ == "__main__":
    main()
