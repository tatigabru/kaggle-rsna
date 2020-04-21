"""

Training and validating models

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
import pydicom
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
from config import IMG_SIZE, RESULTS_DIR, TEST_DIR, WEIGHTS_DIR
from datasets.detection_dataset import DetectionDataset
from datasets.dataset_valid import DatasetValid
from models import MODELS
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from utils.logger import Logger
from utils.my_utils import set_seed

import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

model_configs = MODELS.keys()


def train(
    model_name: str,
    fold: int,
    debug: bool,
    epochs: int,
    num_workers: int=4,
    run: str=None,
    resume_weights: str="",
    resume_epoch: int=0,
):
    """
    Model training
    
    Args: 
        model_name : string name from the models configs listed in models.py file
        fold: evaluation fold number, 0-3
        debug: if True, runs the debugging on few images 
        epochs: number of epochs to train
        num_workers: number of workers available
        run : experiment run string to add for checkpoints name
        resume_weights: directory with weights (if avaialable)
        resume_epoch: number of epoch to continue training    
    """
    model_info = MODELS[model_name]
    run_str = "" if run is None or run == "" else f"_{run}"

    # creates directories for checkpoints, tensorboard and predicitons
    checkpoints_dir = f"{WEIGHTS_DIR}/{model_name}{run_str}_fold_{fold}"
    tensorboard_dir = f"{RESULTS_DIR}/tensorboard/{model_name}{run_str}_fold_{fold}"
    predictions_dir = f"{RESULTS_DIR}/oof/{model_name}{run_str}_fold_{fold}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    print("\n", model_name, "\n")

    logger = Logger(tensorboard_dir)
    retinanet = model_info.factory(**model_info.args)

    # load weights to continue training
    if resume_weights != "":
        print("load model from: ", resume_weights)
        retinanet = torch.load(resume_weights).cuda()
    else:
        retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()

    # datasets for train and validation
    dataset_train = DetectionDataset(
        fold=fold,
        img_size=model_info.img_size,
        is_training=True,
        debug=debug,
        **model_info.dataset_args,
    )

    dataset_valid = DetectionDataset(
        fold=fold, img_size=model_info.img_size, is_training=False, debug=debug
    )

    # dataloaders for train and validation
    dataloader_train = DataLoader(
        dataset_train,
        num_workers=num_workers,
        batch_size=model_info.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=pytorch_retinanet.dataloader.collater2d,
    )

    dataloader_valid = DataLoader(
        dataset_valid,
        num_workers=num_workers,
        batch_size=4,
        shuffle=False,
        drop_last=True,
        collate_fn=pytorch_retinanet.dataloader.collater2d,
    )
    print("{} training images".format(len(dataset_train)))
    print("{} validation images".format(len(dataset_valid)))

    # set optimiser and scheduler
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, verbose=True, factor=0.2
    )
    scheduler_by_epoch = False

    # train cycle
    for epoch_num in range(resume_epoch + 1, epochs):
        retinanet.train()
        if epoch_num < 1:
            retinanet.module.freeze_encoder()  # train FC layers with freezed encoder for the first epoch
        else:
            retinanet.module.unfreeze_encoder()
        retinanet.module.freeze_bn()
        # set losses
        epoch_loss, loss_cls_hist, loss_cls_global_hist, loss_reg_hist = [], [], [], []

        with torch.set_grad_enabled(True):
            data_iter = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for iter_num, data in data_iter:
                optimizer.zero_grad()
                # model inputs
                inputs = [
                    data["img"].cuda().float(),
                    data["annot"].cuda().float(),
                    data["category"].cuda(),
                ]
                # get losses
                (classification_loss, regression_loss, global_classification_loss,) = retinanet(
                    inputs, return_loss=True, return_boxes=False
                )
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                global_classification_loss = global_classification_loss.mean()
                loss = classification_loss + regression_loss + global_classification_loss * 0.1
#                 # uncomment for regress_only
#                 loss = classification_loss + regression_loss
#                 # uncomment for classification0.1
#                 loss = classification_loss * 0.1 + regression_loss
                # back prop
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.05)
                optimizer.step()
                # loss history
                loss_cls_hist.append(float(classification_loss))
                loss_cls_global_hist.append(float(global_classification_loss))
                loss_reg_hist.append(float(regression_loss))
                epoch_loss.append(float(loss))
                # print losses with tqdm interator
                data_iter.set_description(
                    f"{epoch_num} cls: {np.mean(loss_cls_hist):1.4f} cls g: {np.mean(loss_cls_global_hist):1.4f} Reg: {np.mean(loss_reg_hist):1.4f} Loss: {np.mean(epoch_loss):1.4f}"
                )
                del classification_loss
                del regression_loss

        # save model and log loss history
        torch.save(retinanet.module, f"{checkpoints_dir}/{model_name}_{epoch_num:03}.pt")
        logger.scalar_summary("loss_train", np.mean(epoch_loss), epoch_num)
        logger.scalar_summary("loss_train_classification", np.mean(loss_cls_hist), epoch_num)
        logger.scalar_summary(
            "loss_train_global_classification", np.mean(loss_cls_global_hist), epoch_num
        )
        logger.scalar_summary("loss_train_regression", np.mean(loss_reg_hist), epoch_num)

        # validation
        (
            loss_hist_valid,
            loss_cls_hist_valid,
            loss_cls_global_hist_valid,
            loss_reg_hist_valid,
        ) = validation(
            retinanet,
            dataloader_valid,
            epoch_num,
            predictions_dir,
            save_oof=True,
        )

        # log validation loss history
        logger.scalar_summary("loss_valid", np.mean(loss_hist_valid), epoch_num)
        logger.scalar_summary("loss_valid_classification", np.mean(loss_cls_hist_valid), epoch_num)
        logger.scalar_summary(
            "loss_valid_global_classification", np.mean(loss_cls_global_hist_valid), epoch_num,
        )
        logger.scalar_summary("loss_valid_regression", np.mean(loss_reg_hist_valid), epoch_num)

        if scheduler_by_epoch:
            scheduler.step(epoch=epoch_num)
        else:
            scheduler.step(np.mean(loss_reg_hist_valid))
    retinanet.eval()
    torch.save(retinanet, f"{checkpoints_dir}/{model_name}_final.pt")


def validation(
    retinanet: nn.Module, dataloader_valid: nn.Module, epoch_num: int, predictions_dir: str, save_oof=True,
) -> tuple:
    """
    Validate model at the epoch end 
       
    Args: 
        retinanet: current model 
        dataloader_valid: dataloader for the validation fold
        epoch_num: current epoch
        save_oof: boolean flag, if calculate oof predictions and save them in pickle
        predictions_dir: directory fro saving predictions

    Outputs:
        loss_hist_valid: total validation loss, history 
        loss_cls_hist_valid, loss_cls_global_hist_valid: classification validation losses
        loss_reg_hist_valid: regression validation loss
    """
    with torch.no_grad():
        retinanet.eval()
        loss_hist_valid, loss_cls_hist_valid, loss_cls_global_hist_valid, loss_reg_hist_valid = [],[],[],[]
        data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))
        if save_oof:
            oof = collections.defaultdict(list)
        for iter_num, data in data_iter:
            res = retinanet(
                [
                    data["img"].cuda().float(),
                    data["annot"].cuda().float(),
                    data["category"].cuda(),
                ],
                return_loss=True,
                return_boxes=True,
            )

            (
                classification_loss,
                regression_loss,
                global_classification_loss,
                nms_scores,
                global_class,
                transformed_anchors,
            ) = res
            if save_oof:
                # predictions
                oof["gt_boxes"].append(data["annot"].cpu().numpy().copy())
                oof["gt_category"].append(data["category"].cpu().numpy().copy())
                oof["boxes"].append(transformed_anchors.cpu().numpy().copy())
                oof["scores"].append(nms_scores.cpu().numpy().copy())
                oof["category"].append(global_class.cpu().numpy().copy())

            # get losses
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            global_classification_loss = global_classification_loss.mean()
            loss = classification_loss + regression_loss + global_classification_loss * 0.1
            # loss history
            loss_hist_valid.append(float(loss))
            loss_cls_hist_valid.append(float(classification_loss))
            loss_cls_global_hist_valid.append(float(global_classification_loss))
            loss_reg_hist_valid.append(float(regression_loss))
            data_iter.set_description(
                f"{epoch_num} cls: {np.mean(loss_cls_hist_valid):1.4f} cls g: {np.mean(loss_cls_global_hist_valid):1.4f} Reg: {np.mean(loss_reg_hist_valid):1.4f} Loss {np.mean(loss_hist_valid):1.4f}"
            )
            del classification_loss
            del regression_loss

        if save_oof:  # save predictions
            pickle.dump(oof, open(f"{predictions_dir}/{epoch_num:03}.pkl", "wb"))

    return loss_hist_valid, loss_cls_hist_valid, loss_cls_global_hist_valid, loss_reg_hist_valid


def test_model(model_name: str, fold: int, debug: bool, checkpoint: str, pics_dir: str):
    """
    Loads model weights from the checkpoint, plots ground truth and predictions
    
    Args: 
        model_name : string name from the models configs listed in models.py file
        fold       : evaluation fold number, 0-3
        debug      : if True, runs debugging on few images 
        checkpoint : directory with weights (if avaialable) 
        pics_dir   : directory for saving prediction images 
       
    """
    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    model = torch.load(checkpoint)
    model = model.to(device)
    model.eval()
    # load data
    dataset_valid = DetectionDataset(
        fold=fold, img_size=model_info.img_size, is_training=False, debug=debug
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        num_workers=1,
        batch_size=1,
        shuffle=False,
        collate_fn=pytorch_retinanet.dataloader.collater2d,
    )

    data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))
    for iter_num, data in data_iter:
        (
            classification_loss,
            regression_loss,
            global_classification_loss,
            nms_scores,
            nms_class,
            transformed_anchors,
        ) = model(
            [
                data["img"].to(device).float(),
                data["annot"].to(device).float(),
                data["category"].cuda(),
            ],
            return_loss=True,
            return_boxes=True,
        )

        nms_scores = nms_scores.cpu().detach().numpy()
        nms_class = nms_class.cpu().detach().numpy()
        transformed_anchors = transformed_anchors.cpu().detach().numpy()
        print(
            "nms_scores {}, transformed_anchors.shape {}".format(
                nms_scores, transformed_anchors.shape
            )
        )
        print(
            "cls loss:",
            float(classification_loss),
            "global cls loss:",
            global_classification_loss,
            " reg loss:",
            float(regression_loss),
        )
        print(
            "category:",
            data["category"].numpy()[0],
            np.exp(nms_class[0]),
            dataset_valid.categories[data["category"][0]],
        )

        # plot data and ground truth
        plt.figure(iter_num, figsize=(6, 6))
        plt.cla()
        plt.imshow(data["img"][0, 0].cpu().detach().numpy(), cmap=plt.cm.gist_gray)
        plt.axis("off")
        gt = data["annot"].cpu().detach().numpy()[0]
        for i in range(gt.shape[0]):
            if np.all(np.isfinite(gt[i])):
                p0 = gt[i, 0:2]
                p1 = gt[i, 2:4]
                plt.gca().add_patch(
                    plt.Rectangle(
                        p0,
                        width=(p1 - p0)[0],
                        height=(p1 - p0)[1],
                        fill=False,
                        edgecolor="b",
                        linewidth=2,
                    )
                )
        # add predicted boxes to the plot
        for i in range(len(nms_scores)):
            nms_score = nms_scores[i]
            if nms_score < 0.1:
                break
            p0 = transformed_anchors[i, 0:2]
            p1 = transformed_anchors[i, 2:4]
            color = "r"
            if nms_score < 0.3:
                color = "y"
            if nms_score < 0.25:
                color = "g"
            plt.gca().add_patch(
                plt.Rectangle(
                    p0,
                    width=(p1 - p0)[0],
                    height=(p1 - p0)[1],
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
            )
            plt.gca().text(p0[0], p0[1], f"{nms_score:.3f}", color=color)
        plt.show()

        os.makedirs(pics_dir, exist_ok=True)
        plt.savefig(
            f"{pics_dir}/predict_{iter_num}.eps", dpi=300, bbox_inches="tight", pad_inches=0,
        )
        plt.savefig(
            f"{pics_dir}/predict_{iter_num}.png", dpi=300, bbox_inches="tight", pad_inches=0,
        )
        plt.close()
        print(nms_scores)


def generate_predictions(
    model_name: str, fold: int, debug: bool, weights_dir: str, from_epoch: int=0, to_epoch: int=10, save_oof: bool = True, run: str=None,
):
    """
    Loads model weights the epoch checkpoints, 
    calculates oof predictions for and saves them to pickle
    
    Args: 
        model_name : string name from the models configs listed in models.py file
        fold       : evaluation fold number, 0-3
        debug      : if True, runs debugging on few images 
        weights_dir: directory qith model weigths
        from_epoch : the first epoch for predicitions generation 
        to_epoch   : the last epoch for predicitions generation 
        save_oof   : boolean flag weathe rto save precitions
        run        : string name to be added in the experinet name
    """
    predictions_dir = f"{RESULTS_DIR}/test1/{model_name}_fold_{fold}"
    os.makedirs(predictions_dir, exist_ok=True)

    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    for epoch_num in range(from_epoch, to_epoch):
        prediction_fn = f"{predictions_dir}/{epoch_num:03}.pkl"
        if os.path.exists(prediction_fn):
            continue
        print("epoch", epoch_num)
        # load model checkpoint
        checkpoint = (
            f"{weights_dir}/{model_name}_fold_{fold}/{model_name}_{epoch_num:03}.pt"
        )
        print("load", checkpoint)
        try:
            model = torch.load(checkpoint)
        except FileNotFoundError:
            break
        model = model.to(device)
        model.eval()
        # load data
        dataset_valid = DatasetValid(
            is_training=False,
            meta_file= "stage_1_test_meta.csv", 
            debug=debug, 
            img_size=512,
            )
        dataloader_valid = DataLoader(
            dataset_valid,
            num_workers=2,
            batch_size=4,
            shuffle=False,
            collate_fn=pytorch_retinanet.dataloader.collater2d,
        )

        oof = collections.defaultdict(list)
        for iter_num, data in tqdm(enumerate(dataset_valid), total=len(dataloader_valid)):
            data = pytorch_retinanet.dataloader.collater2d([data])
            img = data["img"].to(device).float()
            nms_scores, global_classification, transformed_anchors = model(
                img, return_loss=False, return_boxes=True
            )
            # model outputs to numpy
            nms_scores = nms_scores.cpu().detach().numpy()
            global_classification = global_classification.cpu().detach().numpy()
            transformed_anchors = transformed_anchors.cpu().detach().numpy()
            # out-of-fold predictions
            oof["gt_boxes"].append(data["annot"].cpu().detach().numpy())
            oof["gt_category"].append(data["category"].cpu().detach().numpy())
            oof["boxes"].append(transformed_anchors)
            oof["scores"].append(nms_scores)
            oof["category"].append(global_classification)
        # save epoch predictions
        if save_oof:  
            pickle.dump(oof, open(f"{predictions_dir}/{epoch_num:03}.pkl", "wb"))
    
        
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


def check_metric(
    model_name: str,
    run: str,
    fold: int,
    oof_dir: str,
    start_epoch: int,
    end_epoch: int,
    save_metrics: bool=False,
):
    """
    Loads epoch predicitons and
    calculates the metric for a set of thresholds

    Args: 
        model_name  : string name from the models configs listed in models.py file
        run         : experiment run string to add for checkpoints name
        fold        : evaluation fold number, 0-3
        oof_dir     : directory with out-of-fold predictions
        start_epoch, end_epoch: the first ad last epochs for metric calculation
        save_metrics: boolean flag weather to save metrics
        
    Output:
        thresholds: list of thresholds for mean average precision calculation
        epochs    : range of epochs
        all_scores: all metrics values for all thresholds and epochs
    """
    run_str = "" if run is None or run == "" else f"_{run}"
    predictions_dir = f"{oof_dir}/{model_name}{run_str}_fold_{fold}"
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
    all_scores = []

    for epoch_num in range(start_epoch, end_epoch):
        fn = f"{predictions_dir}/{epoch_num:03}.pkl"
        try:
            oof = pickle.load(open(fn, "rb"))
        except FileNotFoundError:
            continue

        print("epoch ", epoch_num)
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
                        score = metric.map_iou(
                            boxes_true=p1p2_to_xywh(gt_boxes),
                            boxes_pred=p1p2_to_xywh(boxes[mask]),
                            scores=scores[mask],
                        )
                    # print(score)
                    threshold_scores.append(score)

            print("threshold {}, score {}".format(threshold, np.mean(threshold_scores)))
            epoch_scores.append(np.mean(threshold_scores))
        all_scores.append(epoch_scores)

    best_score = np.max(all_scores)
    epochs = np.arange(start_epoch, end_epoch)
    print("best score: ", best_score)
    plt.imshow(np.array(all_scores))
    plt.show()
    if save_metrics:
        scores_dir = f"{RESULTS_DIR}/scores/{model_name}{run_str}_fold_{fold}"
        os.makedirs(scores_dir, exist_ok=True)
        print(
            "all scores.shape: {}, thresholds {}, epochs {}".format(
                np.array(all_scores).shape, thresholds, epochs
            )
        )
        metric_scores = collections.defaultdict(list)
        metric_scores["scores"] = np.array(all_scores)
        metric_scores["tresholds"] = thresholds
        metric_scores["epochs"] = epochs
        pickle.dump(metric_scores, open(f"{scores_dir}/scores.pkl", "wb"))
    return np.array(all_scores), thresholds, epochs


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--action", type=str, default="train", help="Choose action: train, test_model, check_metric, generate_predictions")
    arg("--model", type=str, default="se_resnext101_dr0.75_512", help="String model name from models dictionary")
    arg("--run", type=str, default="", help="Experiment id string to be added for saving model")
    arg("--seed", type=int, default=1234, help="Random seed")
    arg("--fold", type=int, default=0, help="Validation fold")
    arg("--weights_dir", type=str, default="../../checkpoints", help="Directory for loading model weights")
    arg("--epoch", type=int, default=12, help="Current epoch")
    arg("--from-epoch", type=int, default=2, help="Resume training from epoch")
    arg("--num-epochs", type=int, default=20, help="Number of epochs to run")
    arg("--batch-size", type=int, default=4, help="Batch size for training")
    arg("--learning-rate", type=float, default=1e-5, help="Initial learning rate")
    arg("--debug", type=bool, default=False, help="If the debugging mode")
    args = parser.parse_args()

    set_seed(args.seed)
  
    if args.action == "train":
        train(
            model_name=args.model,
            run=args.run,
            fold=args.fold,
            debug=args.debug,
            epochs=args.num_epochs,
        )

    if args.action == "test_model":
        run_str = "" if args.run is None or args.run == "" else f"_{args.run}"
        weights = (
            f"{WEIGHTS_DIR}/{args.model}{run_str}_fold_{args.fold}/{args.model}_{args.epoch:03}.pt"
        )
        test_model(
            model_name=args.model,
            fold=args.fold,
            debug=args.debug,
            checkpoint=weights,
            pics_dir=f"{RESULTS_DIR}/pics",
        )

    if args.action == "check_metric":
        all_scores, thresholds, epochs = check_metric(
            model_name=args.model,
            run=args.run,
            fold=args.fold,
            oof_dir=f"{RESULTS_DIR}/oof",
            start_epoch=1,
            end_epoch=15,
            save_metrics=True,
        )

    if args.action == "generate_predictions":
        generate_predictions(
            model_name=args.model,
            run=args.run,
            fold=args.fold,
            weights_dir=WEIGHTS_DIR,
            debug=args.debug,
            from_epoch=args.from_epoch,
            to_epoch=args.num_epochs + 1,
            save_oof=True,
        )


if __name__ == "__main__":
    main()
