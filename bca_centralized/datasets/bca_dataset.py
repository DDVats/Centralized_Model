import sys
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except:
    BASE_DIR = Path.cwd()

sys.path.insert(0, str(BASE_DIR))

from utils.transform import JointTransform

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class BcaSegDataset(Dataset):
    def __init__(self, csv_path, data_proc_dir, is_train=True, cv_fold=None):
        df = pd.read_csv(csv_path)

        if cv_fold is not None and "cv_fold" in df.columns:
            df = df[df["cv_fold"] != cv_fold] if is_train else df[df["cv_fold"] == cv_fold]

        self.df = df.reset_index(drop=True)
        self.img_dir = Path(data_proc_dir) / "images"
        self.msk_dir = Path(data_proc_dir) / "masks"
        self.joint_tf = JointTransform(augment=is_train)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row["filename"])

        img = Image.open(self.img_dir / f"{fname}.png").convert("L").copy()
        mask = Image.open(self.msk_dir / f"{fname}.png").convert("L").copy()

        img, mask = self.joint_tf(img, mask)

        img_t = self.to_tensor(img).contiguous()

        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).contiguous()

        cstr = str(row.get("center", "Center_1"))
        center_id = int(cstr.split("_")[-1]) if "_" in cstr else 1

        return img_t, mask_t, center_id


class BcaClsDataset(Dataset):
    def __init__(self, csv_path, data_proc_dir, is_train=True, cv_fold=None):
        df = pd.read_csv(csv_path)

        if cv_fold is not None and "cv_fold" in df.columns:
            df = df[df["cv_fold"] != cv_fold] if is_train else df[df["cv_fold"] == cv_fold]

        self.df = df.reset_index(drop=True)
        self.img_dir = Path(data_proc_dir) / "cls_images"

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row["filename"])

        img = Image.open(self.img_dir / f"{fname}.png").convert("RGB").copy()
        img_t = self.transform(img).contiguous()

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        cstr = str(row.get("center", "Center_1"))
        center_id = int(cstr.split("_")[-1]) if "_" in cstr else 1

        return img_t, label, center_id


class BcaMultiTaskDataset(Dataset):
    def __init__(self, csv_path, data_proc_dir, is_train=True, cv_fold=None):
        df = pd.read_csv(csv_path)

        if cv_fold is not None and "cv_fold" in df.columns:
            df = df[df["cv_fold"] != cv_fold] if is_train else df[df["cv_fold"] == cv_fold]

        self.df = df.reset_index(drop=True)

        self.img_dir = Path(data_proc_dir) / "images"
        self.msk_dir = Path(data_proc_dir) / "masks"

        self.joint_tf = JointTransform(augment=is_train)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row["filename"])

        img = Image.open(self.img_dir / f"{fname}.png").convert("L").copy()
        mask = Image.open(self.msk_dir / f"{fname}.png").convert("L").copy()

        img, mask = self.joint_tf(img, mask)

        img_t = self.to_tensor(img).contiguous()

        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).contiguous()

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return img_t, mask_t, label