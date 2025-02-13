import numpy as np
import torch
import torchvision
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy.typing as npt


def load_data(library: str = 'pytorch',
              batch_size: int = 64,
              test_split: float = 0.2,
              random_seed: int = 42) -> (tuple[DataLoader, DataLoader] |
                                         tuple[npt.NDArray[np.float32], npt.NDArray[np.float32],
                                         npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    if library == 'pytorch':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    elif library == 'sklearn':
        mnist = sklearn.datasets.fetch_openml('mnist_784', version=1)
        X = mnist['data'].values.astype('float32')
        y = mnist['target'].astype('int64')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    else:
        raise ValueError("Unsupported library. Use 'pytorch' or 'sklearn'.")
