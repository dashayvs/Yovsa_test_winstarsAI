from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models


def get_data_loader(dataset_path: str | Path, batch_size: int) -> DataLoader:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()
    dataset = ImageFolder(root=dataset_path, transform=auto_transforms)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return data_loader
