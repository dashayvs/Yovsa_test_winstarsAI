from pathlib import Path
from typing import Self, cast

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from task_1_ImgСls.src.classifiers.interface import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self) -> None:
        self.model: RandomForestClassifier = RandomForestClassifier(
            max_depth=45,
            max_features="sqrt",
            min_samples_leaf=2,
            min_samples_split=2,
            n_estimators=300,
            random_state=42,
        )

    def train(self, x_train: npt.NDArray[np.float32], y_train: npt.NDArray[np.int64]) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
        return cast(npt.NDArray[np.int64], self.model.predict(x))

    def save(self, path: Path | str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        rf_classifier = cls()
        rf_classifier.model = joblib.load(path)
        return rf_classifier
