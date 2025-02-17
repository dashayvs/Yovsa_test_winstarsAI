from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from task_1_ImgСls.src.classifiers.nn_classifiers import (
    CNNMnistClassifier,
    FeedForwardNNMnistClassifier,
)
from task_1_ImgСls.src.classifiers.random_forest_classifier import RandomForestMnistClassifier
from torch import Tensor


class MnistClassifier:
    """
        A unified interface for different MNIST classification models.

        This class allows users to choose among different classification algorithms
        ('rf' for RandomForest, 'ffnn' for a Feed-Forward Neural Network, and 'cnn' for a Convolutional Neural Network).
    """

    def __init__(self, algorithm: str = "ffnn"):
        if algorithm == "rf":
            self.model = RandomForestMnistClassifier()
        elif algorithm == "ffnn":
            self.model = FeedForwardNNMnistClassifier()
        elif algorithm == "cnn":
            self.model = CNNMnistClassifier()
        else:
            raise ValueError("Invalid algorithm. Choose from 'rf', 'ffnn', or 'cnn'.")

    def train(self, *args: Any, **kwargs: Any) -> None:
        """
        Trains the selected model.

        Parameters (depending on the selected algorithm):
            - Random Forest ('rf'):
                * args[0]: x_train (npt.NDArray[np.float32]) - Training feature set.
                * args[1]: y_train (npt.NDArray[np.int64]) - Training labels.

            - Feed-Forward Neural Network ('ffnn') and Convolutional Neural Network ('cnn'):
                * args[0]: train_loader (DataLoader) - Training data loader.
                * args[1]: val_loader (DataLoader) - Validation data loader.
                * kwargs['epochs']: (int, optional) - Number of training epochs (default: 50).
                * kwargs['patience']: (int, optional) - Early stopping patience (default: 5).
        """
        self.model.train(*args, **kwargs)

    def predict(self, x: npt.NDArray[np.float32] | Tensor) -> npt.NDArray[np.int64]:
        return self.model.predict(x)

    def save(self, path: Path | str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, algorithm: str, path: Path | str) -> Self:
        instance = cls(algorithm)
        instance.model = instance.model.__class__.load(path)
        return instance
