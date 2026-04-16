# model.py

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Input:  (B, T, C, H, W)
    Output: (B, 2)
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # merge batch and time
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        x = x.view(x.size(0), -1)

        logits = self.classifier(x)   # (B*T, 2)
        logits = logits.view(B, T, -1)
        logits = logits.mean(dim=1)   # (B, 2)

        return logits