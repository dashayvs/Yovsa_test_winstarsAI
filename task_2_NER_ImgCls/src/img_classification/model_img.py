from torch import nn
from torchvision import models
from torchvision.models.efficientnet import EfficientNet


def create_model(output_shape: int = 10) -> EfficientNet:
    # Load pre-trained EfficientNet B0 model with default weights
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze the feature extraction layers (preventing their weights from being updated during training)
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier head to match the desired output shape
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=output_shape, bias=True)
    )

    return model
