import imagehash
import os
import multiprocessing
from PIL import Image
import pandas as pd
import numpy as np
from config import TRAIN_DIR
import pydicom


def img_hash(fn):
    img = Image.open(fn)
    return str(imagehash.dhash(img)) + str(imagehash.phash(img))


def hash_nih_files():
    data_dir = "../data/nih/images/"
    files = [fn for fn in sorted(os.listdir(data_dir)) if fn.endswith("png")]

    pool = multiprocessing.Pool(40)
    hashes = pool.map(img_hash, [data_dir + fn for fn in files])

    df = pd.DataFrame({"fn": files, "hash": hashes})
    df.to_csv("../input/nih_hash.csv", index=False)

    print(df.shape)
    print(len(set(df.hash)))


def dcm_hash(patient_id):
    dcm_data = pydicom.read_file(f"{TRAIN_DIR}/{patient_id}.dcm")
    img = Image.fromarray(dcm_data.pixel_array)
    return str(imagehash.dhash(img)) + str(imagehash.phash(img))


def hash_train_files():
    samples = pd.read_csv("../input/folds.csv")

    pool = multiprocessing.Pool(40)
    hashes = pool.map(dcm_hash, samples.patientId)
    samples["hash"] = hashes
    samples.to_csv("../input/folds_with_hash.csv", index=False)

    print(samples.shape)
    print(len(set(samples.hash)))


def merge_hashes():
    train_folds = pd.read_csv("../input/folds_with_hash.csv")
    nih_df = pd.read_csv("../input/nih_hash.csv")
    train_folds.drop_duplicates(["hash"])

    train_folds = train_folds[["fold", "hash"]].set_index("hash", drop=True)
    nih_df = nih_df.join(train_folds, on="hash")
    nih_df.loc[pd.isna(nih_df.fold), "fold"] = np.random.choice([0, 1, 2, 3], size=np.sum(pd.isna(nih_df.fold)))

    categories = pd.read_csv("../data/nih/Data_Entry_2017.csv")
    categories = categories[["Image Index", "Finding Labels"]].set_index("Image Index", drop=True)
    nih_df = nih_df.join(categories, on="fn")

    nih_df.to_csv("../input/nih_folds.csv", index=False)

    categories = set()
    for combined_categories in nih_df["Finding Labels"].unique():
        categories = categories.union(set(combined_categories.split("|")))

    print(list(sorted(categories)))


if __name__ == "__main__":
    # hash_nih_files()
    # hash_train_files()
    merge_hashes()
