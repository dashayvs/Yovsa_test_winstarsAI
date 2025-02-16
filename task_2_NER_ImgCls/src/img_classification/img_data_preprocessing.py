from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder


def get_data_loader(dataset_path, batch_size):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()
    dataset = ImageFolder(root=dataset_path, transform=auto_transforms)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return data_loader
