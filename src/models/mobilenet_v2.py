"""
src/models/mobilenet_v2.py — MobileNetV2 model for 38-class plant disease classification.
Uses pre-trained ImageNet weights with a custom classifier head (transfer learning).
Reference: Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018.
"""
import torch.nn as nn
from torchvision import models


def get_model(num_classes, pretrained=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)

    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )

    return model
