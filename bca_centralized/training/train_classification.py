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

from models.resnet import build_resnet50, freeze_backbone, unfreeze_for_finetuning, prepare_input
from datasets.bca_dataset import BcaClsDataset
from utils.metrics import compute_cls_metrics
from configs.config import (
    SPLITS, DATA_PROC, CKPT_CLS, LOG_CLS,
    CLS_EPOCHS, CLS_BATCH, CLS_LR_HEAD, CLS_LR_FINETUNE,
    CLS_FINETUNE_EP, CLS_CLASSES, CLS_PRETRAINED,
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    SAVE_EVERY, LOG_EVERY, PATIENCE, CV_FOLDS
)


def train_one_fold(fold, run_cv=True):
    tag = f"fold{fold}" if run_cv else "full"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")

    ckpt_dir = Path(CKPT_CLS) / tag
    log_dir  = Path(LOG_CLS)  / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    cv_fold = fold if run_cv else None

    train_ds = BcaClsDataset(f"{SPLITS}/cls_train.csv", DATA_PROC, True, cv_fold)
    val_ds   = BcaClsDataset(f"{SPLITS}/cls_val.csv",   DATA_PROC, False, None)

    train_dl = DataLoader(train_ds, CLS_BATCH, True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dl   = DataLoader(val_ds, CLS_BATCH, False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"\n[{tag}] Train:{len(train_ds)} Val:{len(val_ds)}")

    labels = train_ds.df["label"].values
    n0, n1 = (labels == 0).sum(), (labels == 1).sum()
    weight = torch.tensor([1.0, n0 / (n1 + 1e-8)], dtype=torch.float).to(device)

    model = build_resnet50(CLS_CLASSES, CLS_PRETRAINED, device="cuda")

    criterion = nn.CrossEntropyLoss(weight=weight)

    freeze_backbone(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CLS_LR_HEAD
    )

    scaler = torch.cuda.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=10
    )

    finetuned = False
    best_auc = 0.0
    no_improve = 0

    for epoch in range(1, CLS_EPOCHS + 1):

        if epoch == CLS_FINETUNE_EP + 1 and not finetuned:
            unfreeze_for_finetuning(model, optimizer, CLS_LR_FINETUNE)
            finetuned = True

        model.train()
        train_loss = 0.0

        for imgs, labels_b, _ in tqdm(train_dl, desc=f"E{epoch:03d} train", leave=False):
            imgs = imgs.to(device)
            labels_b = labels_b.to(device)

            imgs = prepare_input(imgs)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels_b, _ in val_dl:
                imgs = imgs.to(device)
                labels_b = labels_b.to(device)

                imgs = prepare_input(imgs)

                logits = model(imgs)
                val_loss += criterion(logits, labels_b).item() * imgs.size(0)

                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels_b.cpu().numpy())

        val_loss /= len(val_ds)

        metrics = compute_cls_metrics(np.array(all_probs), np.array(all_labels))
        val_auc = metrics["AUC"]

        scheduler.step(val_auc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"E{epoch:03d} | Loss:{train_loss:.4f} | AUC:{val_auc:.4f} | Best:{best_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_auc": best_auc,
            }, str(ckpt_dir / "best_model.pth"))
        else:
            no_improve += 1

        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), str(ckpt_dir / f"epoch_{epoch:03d}.pth"))

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()
    print(f"[{tag}] Best AUC: {best_auc:.4f}")
    return best_auc


def train_classification(use_cv=True):
    print("=" * 55)
    print("Training ResNet-50 Classification")
    print("=" * 55)

    if use_cv:
        scores = []
        for fold in range(CV_FOLDS):
            print(f"\nFOLD {fold+1}/{CV_FOLDS}")
            auc = train_one_fold(fold, True)
            scores.append(auc)

        print("\nResults:")
        print(f"Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    else:
        train_one_fold(0, False)


if __name__ == "__main__":
    train_classification(True)