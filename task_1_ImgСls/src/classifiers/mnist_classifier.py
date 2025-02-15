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
