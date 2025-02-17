import numpy as np
import numpy.typing as npt
import sklearn
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split


def load_data(
        library: str = "pytorch", batch_size: int = 64, test_split: float = 0.2, random_seed: int = 42
) -> (
        tuple[
            DataLoader[tuple[torch.Tensor, torch.Tensor]],
            DataLoader[tuple[torch.Tensor, torch.Tensor]],
            torch.Tensor,
            torch.Tensor,
        ]
        | tuple[
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
        ]
):
    """
    Loads the MNIST dataset using either PyTorch or scikit-learn.

    Parameters:
        library (str): Specifies which library to use ('pytorch' or 'sklearn').
        batch_size (int): The batch size for PyTorch DataLoader.
        test_split (float): The proportion of the dataset to use for testing (only for sklearn).
        random_seed (int): Random seed for reproducibility.

    Returns:
        - If 'pytorch': Tuple containing DataLoaders for training and validation,
          and Tensors for test images and labels.
        - If 'sklearn': Tuple containing NumPy arrays for training and test data.
    """

    if library == "pytorch":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        full_train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        val_size = int(0.2 * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size

        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
        )
        test_images, test_labels = zip(*test_dataset, strict=False)
        test_images = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)

        return train_loader, val_loader, test_images, test_labels

    if library == "sklearn":
        mnist = sklearn.datasets.fetch_openml("mnist_784", version=1)
        X = mnist["data"].to_numpy().astype("float32")
        y = mnist["target"].astype("int64")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_seed
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    raise ValueError("Unsupported library. Use 'pytorch' or 'sklearn'.")
