import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn import datasets
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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        test_size = int(len(full_dataset) * test_split)
        train_size = len(full_dataset) - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(
                                                                        random_seed))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    elif library == 'sklearn':
        mnist = datasets.fetch_openml('mnist_784', version=1)
        X = mnist['data'].values.astype('float32')
        y = mnist['target'].astype('int64')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    else:
        raise ValueError("Unsupported library. Use 'pytorch' or 'sklearn'.")
