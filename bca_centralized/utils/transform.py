import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import sys
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except:
    BASE_DIR = Path.cwd()

sys.path.insert(0, str(BASE_DIR))


class JointTransform:
    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, img, mask):
        if not self.augment:
            return img, mask

        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if random.random() < 0.5:
            angle = random.uniform(-20, 20)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.4:
            scale = random.uniform(0.85, 1.15)
            shear = random.uniform(-10, 10)
            img = TF.affine(
                img,
                angle=0,
                translate=[0, 0],
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.affine(
                mask,
                angle=0,
                translate=[0, 0],
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST
            )

        if random.random() < 0.3:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)

        if random.random() < 0.2:
            img_t = TF.to_tensor(img)
            noise = torch.randn_like(img_t) * 0.03
            img_t = torch.clamp(img_t + noise, 0.0, 1.0)
            img = TF.to_pil_image(img_t)

        return img, mask