from __future__ import annotations

import torch
import torch.nn as nn


EEGNET_CONFIG = {
    "n_classes": 3,
    "chans": 63,
    "samples": 1000,
    "dropout_rate": 0.5,
    "F1": 8,
    "D": 2,
    "F2": 16,
    "kernel_length": 64,
}

EEGNET_BINARY_CONFIG = {
    "n_classes": 2,
    "chans": 63,
    "samples": 1000,
    "dropout_rate": 0.5,
    "F1": 8,
    "D": 2,
    "F2": 16,
    "kernel_length": 64,
}


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes=3,
        chans=63,
        samples=1000,
        dropout_rate=0.5,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat = self.block2(self.block1(dummy))
            self.flattened_dim = feat.numel()

        self.classifier = nn.Linear(self.flattened_dim, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
    
def build_eegnet_model(n_classes: int = 3):
    cfg = EEGNET_CONFIG.copy()
    cfg["n_classes"] = n_classes
    return EEGNet(**cfg)


def build_binary_eegnet_model():
    cfg = EEGNET_BINARY_CONFIG.copy()
    return EEGNet(**cfg)