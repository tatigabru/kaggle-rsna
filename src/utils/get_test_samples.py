import config
from config import DATA_DIR
import pandas as pd


def get_test_labels() -> None:
    """Helper, gets test patients and labels"""
    # load stage 1 meta data
    training_samples = pd.read_csv(DATA_DIR + "stage_1_train_labels.csv")
    # training_samples = training_samples.drop_duplicates().reset_index(drop=True)
    train_patients = training_samples["patientId"]
    print(train_patients.size)
    # load stage 2 meta data
    all_samples = pd.read_csv(DATA_DIR + "stage_2_train_labels.csv")
    all_patients = all_samples["patientId"]
    print(all_patients.size)

    # save meta data for test
    test_meta = all_samples[~all_samples["patientId"].isin(train_patients)]
    print(test_meta.head())
    print(test_meta.size)
    test_meta.to_csv(DATA_DIR + "test_labels_stage_1.csv", index=False)

    # choose test only
    all_samples = all_samples.drop_duplicates().reset_index(drop=True)
    all_patients = all_samples["patientId"]
    print(all_patients.size)
    test_patients = all_patients[~all_patients.isin(train_patients)]
    print(test_patients.size)

    # save test patients ids
    # test_patients.to_csv(DATA_DIR + 'test_patients_id_stage_1.csv', index=False)


def merge_data() -> None:
    samples = pd.read_csv(DATA_DIR + "test_labels_stage_1.csv")
    samples = samples.merge(pd.read_csv(DATA_DIR + "test_meta_stage_1.csv"), on="patientId", how="left")
    samples.to_csv(DATA_DIR + "stage_1_test_meta.csv", index=False)
    print(samples["patientId"].size)
    print(len(samples["patientId"].unique()))
    print(samples.head())


if __name__ == "__main__":
    merge_data()
