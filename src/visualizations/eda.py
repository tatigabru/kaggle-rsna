# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 02:01:15 2019

EDA

"""
import sys
from typing import Dict, List, Optional, Set, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pydicom

from ..config import DATA_DIR, STAGE, TRAIN_DIR

#sys.path.append("/home/user/rsna/progs/rsna-repo/src")


def parse_data(df: pd.DataFrame) -> dict:
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:
    Source: https://www.kaggle.com/zahaviguy/what-are-lung-opacities
    
      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }
    """
    # Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row["y"], row["x"], row["height"], row["width"]]
    parsed: Dict[str, Any] = {}
    for n, row in df.iterrows():
        # Initialize patient entry into parsed
        pid = row["patientId"]
        if pid not in parsed:
            parsed[pid] = {"dicom": '../' + TRAIN_DIR + "/%s.dcm" % pid, "label": row["Target"], "boxes": []}
        # Add box if opacity is present
        if parsed[pid]["label"] == 1:
            parsed[pid]["boxes"].append(extract_box(row))
    return parsed


def plot_dicom(data: dict) -> None:
    """
    Helper to plot single patient X-Ray with bounding box(es) if present 
    """
    # open DICOM file
    dicom = pydicom.read_file(data["dicom"])
    img = dicom.pixel_array
    # convert from single-channel grayscale to 3-channel RGB
    img = np.stack([img] * 3, axis=2)
    # add boxes with specified color, if present
    for box in data["boxes"]:
        color = [255, 0, 0]  # red color code
        img = overlay_box(img=img, box=box, color=color, stroke=6)
    # plot image
    plt.imshow(img, cmap=plt.cm.gist_gray)
    plt.axis("off")


def dicom_to_img(data: dict) -> np.ndarray:
    """
    Helper to convert X-Ray dicom data to image with with bounding box(es) if present 
    """
    # open DICOM file
    dicom = pydicom.read_file(data["dicom"])
    img = dicom.pixel_array
    # convert from single-channel grayscale to 3-channel RGB
    img = np.stack([img] * 3, axis=2)
    # add boxes with specified color, if present
    for box in data["boxes"]:
        color = [255, 0, 0]
        img = overlay_box(img=img, box=box, color=color, stroke=6)
    return img


def overlay_box(img: np.ndarray, box: list, color: List[int], stroke: int=1) -> np.ndarray:
    """
    Helper to overlay single box on image
    """
    # convert coordinates to integers
    box = [int(b) for b in box]
    # extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width
    img[y1 : y1 + stroke, x1:x2] = color
    img[y2 : y2 + stroke, x1:x2] = color
    img[y1:y2, x1 : x1 + stroke] = color
    img[y1:y2, x2 : x2 + stroke] = color
    return img


def plot_patient(df: pd.DataFrame, num: int, patient_class: pd.DataFrame, fig_num=1):
    """ Plot a single patient example 
    Args:
        df: dataframe with meta data
    	num : patient number
    	class_info: dataframe with class labels
    	fig_num : figure number
    """
    patientId = df["patientId"][num]
    parsed = parse_data(df)
    print(parsed[patientId])
    print(patient_class.loc[patientId]["class"])
    class_str = patient_class.loc[patientId]["class"]
    plt.figure(fig_num, figsize=(10, 8))
    plt.title(f"Sample - {class_str}")
    plot_dicom(parsed[patientId])
    plt.show()
    

def main() -> None:
    class_info = pd.read_csv('../' + DATA_DIR + f"stage_{STAGE}_detailed_class_info.csv")
    df = pd.read_csv('../' + DATA_DIR + f"stage_{STAGE}_train_labels.csv")
    
    # sanity checkes
    print(class_info.head())
    print(class_info.shape[0], "class infos loaded")
    print(class_info.sample(3))

    # classes distribution
    print(class_info["patientId"].value_counts().shape[0], "patients")
    plt.figure(1, figsize=[10, 6])
    class_info.groupby("class").size().plot.bar()
    plt.show()

    # parse data
    patient_class = pd.read_csv('../' + DATA_DIR + "stage_1_detailed_class_info.csv", index_col=0)
    parsed = parse_data(df)

    # Normal example
    plot_patient(df, 10, patient_class, fig_num=2)
    # Lung opacity example
    # plot_patient(df, 8, patient_class, fig_num = 3)

    # all three examples
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, wspace=0, hspace=0)
    axs[0].set_title("(a) Normal", fontsize=20)
    img = dicom_to_img(parsed[df["patientId"][3]])
    axs[0].imshow(img, cmap=plt.cm.gist_gray)
    axs[0].axis("off")
    axs[1].set_title("(b) No Lung Opacity / Not Normal", fontsize=20)
    img = dicom_to_img(parsed[df["patientId"][10]])
    axs[1].imshow(img, cmap=plt.cm.gist_gray)
    axs[1].axis("off")
    axs[2].set_title("(c) Lung Opacity", fontsize=20)
    img = dicom_to_img(parsed[df["patientId"][16]])
    axs[2].imshow(img, cmap=plt.cm.gist_gray)
    axs[2].axis("off")
    plt.show()
    
    # save figure
    # plt.savefig("eda.eps", dpi=300, bbox_inches = 'tight',
    # pad_inches = 0)
    # plt.savefig("eda.pdf", dpi=300, bbox_inches = 'tight',
    # pad_inches = 0)


if __name__ == "__main__":
    main()
