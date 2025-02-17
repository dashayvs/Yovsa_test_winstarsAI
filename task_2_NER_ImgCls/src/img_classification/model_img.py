import torch
from torch import nn
from torchvision import models
from torchvision.models.efficientnet import EfficientNet


def create_model(output_shape: int = 10) -> EfficientNet:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=output_shape, bias=True)
    )

    return model
