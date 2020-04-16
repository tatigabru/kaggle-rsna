import os
import pickle
import sys
from collections import defaultdict
sys.path.append("/home/user/rsna/progs/rsna/src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
import skimage.io
from matplotlib import gridspec
from tqdm import tqdm

import pydicom
import torch
from config import CACHE_DIR, DATA_DIR, TRAIN_DIR, IMG_SIZE
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from utils.utils import TransformCfg, timeit_context


class DatasetValid(Dataset):
    """
    RSNA Challenge Pneumonia detection dataset   
    """

    def __init__(self, 
                is_training: bool = False,
                meta_file: str = "stage_1_test_meta.csv",
                debug: bool = False, 
                img_size: int = IMG_SIZE, 
                augmentation_level=10, 
                crop_source=1024):
        """
        Args:
            is_training       : if True, runs the training mode, else runs evaluation mode
            meta_file         : String name of the file with meta, samples id data
            debug             : if True, runs the debugging on few images
            img_size          : the desired image size to resize to        
            augmentation_level: level of augmentations from the set        
        """
        super(DatasetValid, self).__init__()  # inherit it from torch Dataset
        self.is_training = is_training
        self.img_size = img_size
        self.debug = debug
        self.crop_source = crop_source
        self.augmentation_level = augmentation_level
        self.categories = ["No Lung Opacity / Not Normal", "Normal", "Lung Opacity"]
        self.samples = pd.read_csv(os.path.join(DATA_DIR, meta_file))
        
        if self.debug:
            self.samples = self.samples.head(32)
            print("Debug mode, samples: ", self.samples)

        self.patient_ids = list(sorted(self.samples.patientId.unique()))
        self.patient_categories = {}
        self.annotations = defaultdict(list)
        # add annotation points for rotation
        for _, row in self.samples.iterrows():
            patient_id = row["patientId"]
            self.patient_categories[patient_id] = self.categories.index(row["class"])
            if row["Target"] > 0:
                x, y, w, h = row.x, row.y, row.width, row.height
                points = np.array(
                    [
                        [x, y + h / 3],
                        [x, y + h * 2 / 3],
                        [x + w, y + h / 3],
                        [x + w, y + h * 2 / 3],
                        [x + w / 3, y],
                        [x + w * 2 / 3, y],
                        [x + w / 3, y + h],
                        [x + w * 2 / 3, y + h],
                    ]
                )
#                 # uncomment for using only basic rotations
#                 points = np.array([[x, y], [x, y + h], [x + w, y], [x + w, y + h],])
                self.annotations[patient_id].append(points)

    def get_image(self, patient_id):
        """Load a dicom image to an array"""
        try:
            dcm_data = pydicom.read_file(f"{TRAIN_DIR}/{patient_id}.dcm")
            img = dcm_data.pixel_array
            return img
        except:
            pass

    def num_classes(self):
        return 3

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        img = self.get_image(patient_id)

        if self.crop_source != 1024:
            img_source_w = self.crop_source
            img_source_h = self.crop_source
        else:
            img_source_h, img_source_w = img.shape[:2]
        img_h, img_w = img.shape[:2]

        # set augmentation levels
        augmentation_sigma = {
            10: dict(scale=0.1, angle=5.0, shear=2.5, gamma=0.2, hflip=False),
            11: dict(scale=0.1, angle=0.0, shear=2.5, gamma=0.2, hflip=False),
            15: dict(scale=0.15, angle=6.0, shear=4.0, gamma=0.2, hflip=np.random.choice([True, False])),
            20: dict(scale=0.15, angle=6.0, shear=4.0, gamma=0.25, hflip=np.random.choice([True, False])),
            21: dict(scale=0.15, angle=0.0, shear=4.0, gamma=0.25, hflip=np.random.choice([True, False])),
        }[self.augmentation_level]
        # training mode augments
        if self.is_training:
            cfg = TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w / 2 + np.random.uniform(-32, 32),
                src_center_y=img_h / 2 + np.random.uniform(-32, 32),
                scale_x=self.img_size / img_source_w * (2 ** np.random.normal(0, augmentation_sigma["scale"])),
                scale_y=self.img_size / img_source_h * (2 ** np.random.normal(0, augmentation_sigma["scale"])),
                angle=np.random.normal(0, augmentation_sigma["angle"]),
                shear=np.random.normal(0, augmentation_sigma["shear"]),
                hflip=augmentation_sigma["hflip"],
                vflip=False,
            )
        # validation mode augments
        else:
            cfg = TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w / 2,
                src_center_y=img_h / 2,
                scale_x=self.img_size / img_source_w,
                scale_y=self.img_size / img_source_h,
                angle=0,
                shear=0,
                hflip=False,
                vflip=False,
            )
        # add more augs in training modes
        crop = cfg.transform_image(img)
        if self.is_training:
            crop = np.power(crop, 2.0 ** np.random.normal(0, augmentation_sigma["gamma"]))
            if self.augmentation_level == 20 or self.augmentation_level == 21:
                aug = iaa.Sequential(
                    [
                        iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2))),
                        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0))),
                        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
                    ]
                )
                crop = (
                    aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[:, :, 0].astype(np.float32)
                    / 255.0
                )
            if self.augmentation_level == 15:
                aug = iaa.Sequential(
                    [iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0.0, 1.0))), iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)))]
                )
                crop = (
                    aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[:, :, 0].astype(np.float32)
                    / 255.0
                )
        # add annotation points
        annotations = []
        for annotation in self.annotations[patient_id]:
            points = cfg.transform().inverse(annotation)
            res = np.zeros((1, 5))
            p0 = np.min(points, axis=0)
            p1 = np.max(points, axis=0)
            res[0, 0:2] = p0
            res[0, 2:4] = p1
            res[0, 4] = 0
            annotations.append(res)
        if len(annotations):
            annotations = np.row_stack(annotations)
        else:
            annotations = np.zeros((0, 5))
        # print('patient_id', patient_id)
        sample = {"img": crop, "annot": annotations, "scale": 1.0, "category": self.patient_categories[patient_id]}
        return sample


def test_dataset_sample(sample_num=0):
    """Test dataset on a single sample
    Args:
        sample_num: sample number from the dataset
    """
    dataset = DatasetValid(
        is_training=False, 
        meta_file= "stage_1_test_meta.csv", 
        debug=False, 
        img_size=224)
    # print and plot sample
    print("dataset sample: \n", dataset[sample_num])
    plt.figure()
    plt.imshow(dataset[sample_num]["img"], cmap=plt.cm.gist_gray)
    annot = dataset[sample_num]["annot"]
    print("annotations: \n", annot)
    for annot in dataset[sample_num]["annot"]:
        p0 = annot[0:2]
        p1 = annot[2:4]
        plt.gca().add_patch(plt.Rectangle(p0, width=(p1 - p0)[0], height=(p1 - p0)[1], fill=False, edgecolor="r", linewidth=2))
    plt.show()


def test_augmentations(sample_num=12, aug_level=20):
    """Test augmentations on a single sample
    Args:
        sample_num: sample number from the dataset
        aug_level: augmentations level        
    """
    with timeit_context("load ds"):
        ds = DatasetValid(
            is_training=False, 
            meta_file= "stage_1_test_meta.csv", 
            debug=True, 
            img_size=224,
            augmentation_level=aug_level)
        # print and plot sample
        print(ds[sample_num])
        plt.figure()
        plt.imshow(ds[sample_num]["img"], cmap=plt.cm.gist_gray)
        for annot in ds[sample_num]["annot"]:
            p0 = annot[0:2]
            p1 = annot[2:4]
            plt.gca().add_patch(plt.Rectangle(p0, width=(p1 - p0)[0], height=(p1 - p0)[1], fill=False, edgecolor="r", linewidth=2))
        # plot augmented sample, 10 examples
        plt.figure()
        for i in range(10):
            sample = ds[sample_num]
            plt.imshow(sample["img"], cmap=plt.cm.gist_gray)
            for annot in sample["annot"]:
                p0 = annot[0:2]
                p1 = annot[2:4]
                plt.gca().add_patch(plt.Rectangle(p0, width=(p1 - p0)[0], height=(p1 - p0)[1], fill=False, edgecolor="r", linewidth=2))
            plt.show()


def plot_augmented_image(sample_num: int, aug_level=20, save=False):
    """Plot augmentations
    Args:
        sample_num: sample number from the dataset
        aug_level: augmentations level   
    """
    ds = DatasetValid(
            is_training=True, 
            meta_file= "stage_1_test_meta.csv", 
            debug=False, 
            img_size=224,
            augmentation_level=aug_level)
    print(ds[sample_num])
    # plot sample with agments
    plt.figure(figsize=(12, 7.25))
    gs = gridspec.GridSpec(3, 5)
    gs.update(top=1, bottom=0, right=1, left=0, wspace=0.0, hspace=0.0)  # set the spacing between axes
    for i in range(15):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.imshow(ds[sample_num]["img"], cmap=plt.cm.gist_gray)
        for annot in ds[sample_num]["annot"]:
            print("ds sample annot", ds[sample_num]["annot"])
            p0 = annot[0:2]
            p1 = annot[2:4]
            plt.gca().add_patch(plt.Rectangle(p0, width=(p1 - p0)[0], height=(p1 - p0)[1], fill=False, edgecolor="r", linewidth=2))
    if save:
        # save figure
        plt.savefig("augs20.eps", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.savefig("augs20.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    test_dataset_sample(sample_num = 3)
    #test_augmentations(sample_num = 12)
    plot_augmented_image(sample_num=3, aug_level=10)
