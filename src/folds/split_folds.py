"""

Creates stratfied folds
Saves the train meta with folds to a csv file
 
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .. config import DATA_DIR, STAGE


def create_folds(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, nb_folds: int, if_save: bool) -> pd.DataFrame:
    """
    Create folds
    Args: 
        df       : train meta dataframe
        X        : X Series to split
        y        : y Series to use for stratification
        nb_folds : number of folds
        if_save  : boolean flag weather to save the folds
    Output: 
        df: train meta with splitted folds
    """
    df["fold"] = -1  # set all folds to -1 initially
    skf = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=42)
    # split folds
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        df.loc[test_index, "fold"] = fold
    # save dataframe with folds (optionally)
    if if_save:
        df.to_csv(os.path.join(DATA_DIR, f"folds_stage{str(STAGE)}.csv"), index=False)
    return df


def test_folds(df: pd.DataFrame, nb_folds: int) -> None:
    """
    Test class distribution in folds
    Args: 
        df       : train meta dataframe with folds
        nb_folds : number of folds
    """
    for cls in df["class"].unique():
        print(cls)
        cls_samples = df[df["class"] == cls].reset_index(drop=True)
        for fold in range(nb_folds):
            print(fold, len(cls_samples[cls_samples.fold == fold]))


def remove_patients_without_boxes() -> pd.DataFrame:
    """
    Helper, removes patients without annotated boxes, i.e. no pathology
    
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "folds.csv"))
    df = df[df["class"] == "Lung Opacity"]
    print(df.head())
    return df


def main() -> None:
    nb_folds = 4
    # load meta data
    training_samples = pd.read_csv(os.path.join(DATA_DIR, f"stage_{str(STAGE)}_detailed_class_info.csv"))
    training_samples = training_samples.drop_duplicates().reset_index(drop=True)
    X = training_samples["patientId"]
    y = training_samples["class"]
    print(y.unique())

    # create folds
    folds = create_folds(training_samples, X, y, nb_folds, if_save=False)
    test_folds(folds, nb_folds)


if __name__ == "__main__":
    main()
