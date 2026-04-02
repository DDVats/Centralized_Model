import sys, os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except:
    BASE_DIR = Path.cwd()

sys.path.insert(0, str(BASE_DIR))

from configs.config import (
    DATA_PROC, SPLITS, CENTERS,
    TEST_RATIO_SEG, TEST_RATIO_CLS,
    VAL_RATIO, RANDOM_SEED, CV_FOLDS
)

os.makedirs(SPLITS, exist_ok=True)


def safe_concat(dfs):
    dfs = [d for d in dfs if len(d) > 0]
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def patient_split(df, test_ratio, val_ratio):
    train_rows, val_rows, test_rows = [], [], []

    for center in CENTERS:
        cdf = df[df["center"] == center].copy()
        if len(cdf) == 0:
            continue

        patient_df = (
            cdf.groupby("patient")["label"]
            .agg(lambda x: int(x.mode()[0]))
            .reset_index()
        )

        patients = patient_df["patient"].values
        labels = patient_df["label"].values

        if len(np.unique(labels)) < 2 or len(patients) < 3:
            train_rows.append(cdf)
            continue

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=RANDOM_SEED
        )

        train_idx, test_idx = next(sss.split(patients, labels))

        train_patients = patients[train_idx]
        test_patients = patients[test_idx]

        train_labels = labels[train_idx]

        if len(np.unique(train_labels)) > 1 and len(train_patients) > 2:
            sss_val = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_ratio,
                random_state=RANDOM_SEED
            )
            tr_idx, val_idx = next(sss_val.split(train_patients, train_labels))
            final_train = train_patients[tr_idx]
            val_patients = train_patients[val_idx]
        else:
            final_train = train_patients
            val_patients = []

        train_rows.append(cdf[cdf["patient"].isin(final_train)])
        val_rows.append(cdf[cdf["patient"].isin(val_patients)])
        test_rows.append(cdf[cdf["patient"].isin(test_patients)])

    return (
        safe_concat(train_rows),
        safe_concat(val_rows),
        safe_concat(test_rows)
    )


def add_cv_folds(train_df):
    if len(train_df) == 0:
        return train_df

    patient_df = (
        train_df.groupby("patient")["label"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index()
    )

    if len(patient_df) < CV_FOLDS:
        train_df["cv_fold"] = 0
        return train_df

    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    folds = np.zeros(len(patient_df), dtype=int)

    for fold, (_, val_idx) in enumerate(
        skf.split(patient_df["patient"], patient_df["label"])
    ):
        folds[val_idx] = fold

    patient_df["cv_fold"] = folds
    fold_map = dict(zip(patient_df["patient"], patient_df["cv_fold"]))

    train_df["cv_fold"] = train_df["patient"].map(fold_map)

    return train_df


def create_splits():
    labels_csv = Path(DATA_PROC) / "labels.csv"

    if not labels_csv.exists():
        print("Run preprocessing first")
        return

    df = pd.read_csv(labels_csv)

    print(f"Total slices: {len(df)}")
    print(f"MIBC: {df['label'].sum()} | NMIBC: {(df['label']==0).sum()}")

    print("\n=== SEGMENTATION SPLITS ===")
    tr, va, te = patient_split(df, TEST_RATIO_SEG, VAL_RATIO)
    tr = add_cv_folds(tr)

    tr.to_csv(f"{SPLITS}/seg_train.csv", index=False)
    va.to_csv(f"{SPLITS}/seg_val.csv", index=False)
    te.to_csv(f"{SPLITS}/seg_test.csv", index=False)

    print(f"Seg → train:{len(tr)} val:{len(va)} test:{len(te)}")

    print("\n=== CLASSIFICATION SPLITS ===")
    tr, va, te = patient_split(df, TEST_RATIO_CLS, VAL_RATIO)
    tr = add_cv_folds(tr)

    tr.to_csv(f"{SPLITS}/cls_train.csv", index=False)
    va.to_csv(f"{SPLITS}/cls_val.csv", index=False)
    te.to_csv(f"{SPLITS}/cls_test.csv", index=False)

    print(f"Cls → train:{len(tr)} val:{len(va)} test:{len(te)}")

    print("\nSplits created successfully")


if __name__ == "__main__":
    create_splits()