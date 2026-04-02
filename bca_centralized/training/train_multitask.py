import sys, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    BASE_DIR = os.getcwd()

sys.path.insert(0, BASE_DIR)

from models.unet import MultiTaskUNet
from datasets.bca_dataset import BcaMultiTaskDataset
from losses.dice_loss import DiceFocalLoss
from utils.metrics import compute_seg_metrics, compute_cls_metrics, MetricStore
from configs.config import (
    SPLITS, DATA_PROC, CKPT_SEG, LOG_SEG,
    SEG_EPOCHS, SEG_BATCH, SEG_LR,
    SEG_IN_CH, SEG_OUT_CH, SEG_THRESHOLD,
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    SAVE_EVERY, LOG_EVERY, PATIENCE, CV_FOLDS
)


def train_one_fold(fold, run_cv=True):
    tag = f"fold{fold}" if run_cv else "full"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(CKPT_SEG) / f"multitask_{tag}"
    log_dir  = Path(LOG_SEG)  / f"multitask_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    cv_fold = fold if run_cv else None

    train_ds = BcaMultiTaskDataset(f"{SPLITS}/seg_train.csv", DATA_PROC, True, cv_fold)
    val_ds   = BcaMultiTaskDataset(f"{SPLITS}/seg_val.csv",   DATA_PROC, False, None)

    train_dl = DataLoader(train_ds, SEG_BATCH, True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dl   = DataLoader(val_ds, SEG_BATCH, False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"\n[{tag}] Train:{len(train_ds)} Val:{len(val_ds)}")

    model = MultiTaskUNet(SEG_IN_CH, SEG_OUT_CH).to(device)

    seg_loss_fn = DiceFocalLoss()
    cls_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=SEG_LR)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score = 0.0
    no_improve = 0

    for epoch in range(1, SEG_EPOCHS + 1):

        model.train()
        train_loss = 0.0

        for imgs, masks, labels in tqdm(train_dl, desc=f"E{epoch:03d} train", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                seg_out, cls_out = model(imgs)

                loss_seg = seg_loss_fn(seg_out, masks)
                loss_cls = cls_loss_fn(cls_out, labels)

                loss = loss_seg + 0.5 * loss_cls

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0

        seg_store = MetricStore()
        probs_all = []
        labels_all = []

        with torch.no_grad():
            for imgs, masks, labels in val_dl:
                imgs = imgs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                seg_out, cls_out = model(imgs)

                loss_seg = seg_loss_fn(seg_out, masks)
                loss_cls = cls_loss_fn(cls_out, labels)
                val_loss += (loss_seg + 0.5 * loss_cls).item() * imgs.size(0)

                seg_probs = torch.sigmoid(seg_out)

                for p, m in zip(seg_probs.cpu().numpy(), masks.cpu().numpy()):
                    seg_store.update(compute_seg_metrics(p[0], m[0], SEG_THRESHOLD))

                probs = torch.softmax(cls_out, dim=1)[:, 1]
                probs_all.extend(probs.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        val_loss /= len(val_ds)

        seg_metrics = seg_store.mean()
        val_dsc = seg_metrics.get("DSC", 0.0)

        cls_metrics = compute_cls_metrics(np.array(probs_all), np.array(labels_all))
        val_auc = cls_metrics["AUC"]

        score = val_dsc + val_auc

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Seg/DSC", val_dsc, epoch)
        writer.add_scalar("Cls/AUC", val_auc, epoch)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"E{epoch:03d} | Loss:{train_loss:.4f} | DSC:{val_dsc:.4f} | AUC:{val_auc:.4f}")

        if score > best_score:
            best_score = score
            no_improve = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_score": best_score,
            }, str(ckpt_dir / "best_model.pth"))
        else:
            no_improve += 1

        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), str(ckpt_dir / f"epoch_{epoch:03d}.pth"))

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()
    print(f"[{tag}] Best Score: {best_score:.4f}")
    return best_score


def train_multitask(use_cv=True):
    print("=" * 55)
    print("Training MultiTask UNet")
    print("=" * 55)

    if use_cv:
        scores = []
        for fold in range(CV_FOLDS):
            print(f"\nFOLD {fold+1}/{CV_FOLDS}")
            score = train_one_fold(fold, True)
            scores.append(score)

        print("\nResults:")
        print(f"Mean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    else:
        train_one_fold(0, False)


if __name__ == "__main__":
    train_multitask(True)