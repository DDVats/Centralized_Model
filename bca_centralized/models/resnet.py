import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes=2, pretrained=True, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)

    print(f"Model loaded on: {device}")
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen backbone | Trainable: {trainable:,} / {total:,}")


def unfreeze_for_finetuning(model, optimizer, finetune_lr=1e-5):
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    new_params = [
        p for n, p in model.named_parameters()
        if ("layer4" in n) and p.requires_grad
    ]

    if len(new_params) > 0:
        optimizer.add_param_group({"params": new_params, "lr": finetune_lr})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Finetuning | Trainable: {trainable:,} / {total:,}")


def prepare_input(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.contiguous()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_resnet50(device=device)

    x = torch.randn(2, 1, 128, 128).to(device)
    x = prepare_input(x)

    out = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")