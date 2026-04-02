import os, sys, random
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except:
    BASE_DIR = Path.cwd()

sys.path.insert(0, str(BASE_DIR))

from configs.config import (
    DATA_RAW, DATA_PROC, CENTERS,
    PATCH_SEG, PATCH_CLS,
    EXPAND_VOXEL, RANDOM_OFFSET, RANDOM_SEED
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_nii(path):
    try:
        return np.asarray(nib.load(path).dataobj, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to load: {path}")
        return None


def roi_crop(img, mask, expand):
    idx = np.argwhere(mask > 0)
    if len(idx) == 0:
        return img, mask

    d1_min, d1_max = idx[:, 0].min(), idx[:, 0].max()
    d2_min, d2_max = idx[:, 1].min(), idx[:, 1].max()
    d3_min, d3_max = idx[:, 2].min(), idx[:, 2].max()

    d1_min = max(0, d1_min - expand)
    d1_max = min(img.shape[0], d1_max + expand)
    d2_min = max(0, d2_min - expand)
    d2_max = min(img.shape[1], d2_max + expand)
    d3_min = max(0, d3_min - 1)
    d3_max = min(img.shape[2], d3_max + 1)

    return img[d1_min:d1_max, d2_min:d2_max, d3_min:d3_max], \
           mask[d1_min:d1_max, d2_min:d2_max, d3_min:d3_max]


def crop_and_resize(arr, size, cx=None, cy=None, interp=cv2.INTER_LINEAR):
    h, w = arr.shape

    if h < size:
        arr = np.pad(arr, ((0, size - h), (0, 0)), mode='reflect')
    if w < size:
        arr = np.pad(arr, ((0, 0), (0, size - w)), mode='reflect')

    h, w = arr.shape

    if cx is None: cx = h // 2
    if cy is None: cy = w // 2

    cx = np.clip(cx, size // 2, h - size // 2)
    cy = np.clip(cy, size // 2, w - size // 2)

    half = size // 2

    x1 = int(cx - half)
    y1 = int(cy - half)

    crop = arr[x1:x1 + size, y1:y1 + size]

    if crop.shape != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=interp)

    return crop


def normalise(arr):
    p1, p99 = np.percentile(arr, (1, 99))
    arr = np.clip(arr, p1, p99)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def find_annotation(ann_dir, case_id, filename):
    for candidate in [
        Path(ann_dir) / filename,
        Path(ann_dir) / f"{case_id}_1.nii.gz",
        Path(ann_dir) / f"{case_id}_2.nii.gz",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def run_preprocessing():
    seg_img_dir = Path(DATA_PROC) / "images"
    seg_msk_dir = Path(DATA_PROC) / "masks"
    cls_img_dir = Path(DATA_PROC) / "cls_images"

    seg_img_dir.mkdir(parents=True, exist_ok=True)
    seg_msk_dir.mkdir(parents=True, exist_ok=True)
    cls_img_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print("=" * 55)
    print("Preprocessing all centers")
    print("=" * 55)

    for center in CENTERS:
        center_dir = Path(DATA_RAW) / center
        img_dir = center_dir / "Image"
        ann_dir = center_dir / "Annotation"

        label_files = list(center_dir.glob("*.xlsx"))
        if len(label_files) == 0:
            print(f"[SKIP] {center} no label file")
            continue

        df_lbl = pd.read_excel(label_files[0])

        mibc_col = next(
            (c for c in df_lbl.columns if any(k in c.upper() for k in ["MIBC","LABEL","INVASION","MUSCLE"])),
            df_lbl.columns[0]
        )

        id_col = [c for c in df_lbl.columns if c != mibc_col][0]

        label_dict = dict(zip(
            df_lbl[id_col].astype(str),
            df_lbl[mibc_col].astype(int)
        ))

        img_files = sorted(img_dir.glob("*.nii.gz"))

        for img_file in tqdm(img_files, desc=center):
            case_id = img_file.stem.replace(".nii", "")

            img_vol = load_nii(str(img_file))
            if img_vol is None:
                continue

            ann_path = find_annotation(ann_dir, case_id, img_file.name)
            if ann_path is None:
                continue

            mask_vol = load_nii(ann_path)
            if mask_vol is None:
                continue

            mask_vol = (mask_vol >= 0.5).astype(np.uint8)

            label = label_dict.get(case_id)
            if label is None:
                continue

            img_vol, mask_vol = roi_crop(img_vol, mask_vol, EXPAND_VOXEL)

            for sl in range(img_vol.shape[2]):
                img_sl = img_vol[:, :, sl]
                mask_sl = mask_vol[:, :, sl]

                if mask_sl.sum() == 0:
                    continue

                ox = random.randint(-RANDOM_OFFSET, RANDOM_OFFSET)
                oy = random.randint(-RANDOM_OFFSET, RANDOM_OFFSET)

                cx = img_sl.shape[0] // 2 + ox
                cy = img_sl.shape[1] // 2 + oy

                seg_img = normalise(crop_and_resize(img_sl, PATCH_SEG, cx, cy))
                seg_msk = crop_and_resize(mask_sl, PATCH_SEG, cx, cy, cv2.INTER_NEAREST)

                fname = f"{center}_{case_id}_sl{sl:03d}"

                cv2.imwrite(str(seg_img_dir / f"{fname}.png"), (seg_img * 255).astype(np.uint8))
                cv2.imwrite(str(seg_msk_dir / f"{fname}.png"), (seg_msk * 255).astype(np.uint8))

                cls_img = normalise(crop_and_resize(img_sl, PATCH_CLS))
                cv2.imwrite(str(cls_img_dir / f"{fname}.png"), (cls_img * 255).astype(np.uint8))

                records.append({
                    "filename": fname,
                    "center": center,
                    "patient": case_id,
                    "slice": sl,
                    "label": int(label)
                })

    df = pd.DataFrame(records)
    df.to_csv(Path(DATA_PROC) / "labels.csv", index=False)

    print("\nDone")
    print(f"Total slices: {len(df)}")
    print(f"MIBC: {df['label'].sum()}")
    print(f"NMIBC: {(df['label']==0).sum()}")

    return df


if __name__ == "__main__":
    run_preprocessing()