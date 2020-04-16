import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
import skimage.io
from tqdm import tqdm

import pydicom
from config import CACHE_DIR, DATA_DIR, IMG_SIZE, TRAIN_DIR
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from utils.utils import TransformCfg

sys.path.append("/home/user/rsna/progs/rsna/src")


class TestDataset(Dataset):
    """
    RSNA Challenge Pneumonia detection dataset, test patients   
    """

    def __init__(self, debug: bool, img_size: int):
        """
        Args:
            debug   : if True, runs the debugging on few images
            img_size: the desired image size to resize to        
        """
        super(TestDataset, self).__init__()  # inherit it from torch Dataset
        self.img_size = img_size
        self.debug = debug
        self.categories = ["No Lung Opacity / Not Normal", "Normal", "Lung Opacity"]
        self.samples = pd.read_csv(DATA_DIR + "stage_1_test_meta.csv")

        if self.debug:
            self.samples = self.samples.head(16)
            print("Debug mode, samples: /n", self.samples)

        self.patient_ids = list(sorted(self.samples.patientId.unique()))
        self.patient_categories = {}
        self.annotations: DefaultDict[str, list] = defaultdict(list)

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
        img_h, img_w = img.shape[:2]

        # test mode augments
        cfg = TransformCfg(
            crop_size=self.img_size,
            src_center_x=img_w / 2,
            src_center_y=img_h / 2,
            scale_x=self.img_size / img_w,
            scale_y=self.img_size / img_h,
            angle=0,
            shear=0,
            hflip=False,
            vflip=False,
        )
        crop = cfg.transform_image(img)

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
    dataset = TestDataset(debug=True, img_size=224)
    # print and plot sample
    print("dataset sample: \n", dataset[sample_num])
    plt.figure()
    plt.imshow(dataset[sample_num]["img"])
    annot = dataset[sample_num]["annot"]
    print("annotations: \n", annot)
    for annot in dataset[sample_num]["annot"]:
        p0 = annot[0:2]
        p1 = annot[2:4]
        plt.gca().add_patch(plt.Rectangle(p0, width=(p1 - p0)[0], height=(p1 - p0)[1], fill=False, edgecolor="r", linewidth=2))
    plt.show()


if __name__ == "__main__":
    test_dataset_sample(sample_num=1)
