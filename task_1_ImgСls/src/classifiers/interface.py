from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def predict(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]: ...

    @abstractmethod
    def save(self, path: Path | str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> Self: ...
