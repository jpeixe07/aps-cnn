from __future__ import annotations

import torch.nn as nn
import torch
from torchvision.models import ResNet18_Weights, resnet18


class SmallCNN(nn.Module):
    """CNN compacta para MNIST e CIFAR-10; dropout é a variável de ablação."""

    def __init__(self, in_channels: int, num_classes: int = 10, dropout: bool = True):
        super().__init__()

        def regularizer2d() -> nn.Module:
            return nn.Dropout2d(0.15) if dropout else nn.Identity()

        def regularizer() -> nn.Module:
            return nn.Dropout(0.35) if dropout else nn.Identity()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), regularizer2d(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), regularizer2d(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            regularizer(), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_pretrained_resnet18(num_classes: int = 10, weights_path=None) -> nn.Module:
    """ResNet-18 ImageNet usada como extratora fixa, com cabeça ajustável."""
    if weights_path is not None and weights_path.exists():
        model = resnet18(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model._frozen_feature_extractor = True
    return model
