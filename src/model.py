import torch
from torch import nn

class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class EcoSortNet(nn.Module):
    """Small depthwise-separable CNN: 14,350 parameters, 96x96 RGB input."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            DSBlock(16, 24, 2),
            DSBlock(24, 32, 2),
            DSBlock(32, 48, 2),
            DSBlock(48, 64, 2),
            DSBlock(64, 80, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.15), nn.Linear(80, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
