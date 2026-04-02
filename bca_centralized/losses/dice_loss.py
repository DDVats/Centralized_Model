import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        p = pred.view(pred.size(0), -1)
        t = target.view(target.size(0), -1)

        inter = (p * t).sum(dim=1)
        dice = (2 * inter + self.smooth) / (p.sum(dim=1) + t.sum(dim=1) + self.smooth)

        return 1 - dice.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.bce(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        bce = F.binary_cross_entropy(pred_sig, target, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, smooth=1.0, gamma=2.0, alpha=0.25):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.focal = FocalLoss(gamma, alpha)

    def forward(self, pred, target):
        return 0.5 * self.dice(pred, target) + 0.5 * self.focal(pred, target)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        p = pred.view(pred.size(0), -1)
        t = target.view(target.size(0), -1)

        tp = (p * t).sum(dim=1)
        fp = ((1 - t) * p).sum(dim=1)
        fn = (t * (1 - p)).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tversky.mean()


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss()

    def forward(self, pred, target):
        return (
            0.4 * self.dice(pred, target) +
            0.3 * self.bce(pred, target) +
            0.3 * self.tversky(pred, target)
        )