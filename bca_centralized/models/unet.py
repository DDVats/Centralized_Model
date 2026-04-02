import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_classes=2, features=(64, 128, 256, 512)):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.enc = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1] * 2),
            nn.Dropout2d(0.2)
        )

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decs.append(DoubleConv(f * 2, f))

        self.seg_head = nn.Conv2d(features[0], out_channels, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[-1] * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        skips = []

        for enc_block in self.enc:
            x = enc_block(x)
            skips.append(x)
            x = self.pool(x)

        bottleneck = self.bottleneck(x)

        cls_out = self.cls_head(bottleneck.contiguous())

        x = bottleneck
        skips = skips[::-1]

        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skips[i]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = self.decs[i](x)

        seg_out = self.seg_head(x)

        return seg_out, cls_out


def get_model(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    model = MultiTaskUNet(in_channels=1, out_channels=1, num_classes=2)
    model = model.to(device)

    print(f"Model loaded on: {device}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(device=device)

    x = torch.randn(2, 1, 160, 160).to(device)

    seg, cls = model(x)

    print(f"Input: {x.shape}")
    print(f"Seg Output: {seg.shape}")
    print(f"Cls Output: {cls.shape}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")