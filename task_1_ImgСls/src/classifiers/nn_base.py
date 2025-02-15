from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import torch
from task_1_ImgСls.src.classifiers.interface import MnistClassifierInterface
from torch import nn, optim
from torch.utils.data import DataLoader


class BaseNNMnistClassifier(MnistClassifierInterface):
    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, patience: int = 5
    ) -> None:
        best_loss = float("inf")
        best_model_weights = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate(val_loader)
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = self.model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        if best_model_weights:
            self.model.load_state_dict(best_model_weights)

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for images, labels in train_loader:
            images_device, labels_device = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images_device)
            loss = self.criterion(outputs, labels_device)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images_device, labels_device = images.to(self.device), labels.to(self.device)
                outputs = self.model(images_device)
                loss = self.criterion(outputs, labels_device)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def predict(self, images: torch.Tensor) -> npt.NDArray[np.int64]:
        self.model.eval()
        with torch.no_grad():
            if images.ndimension() == 3:
                images = images.unsqueeze(0)
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def save(self, path: Path | str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_class": self.model.__class__.__name__,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> Self:
        checkpoint = torch.load(path)
        instance = cls()
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        return instance
