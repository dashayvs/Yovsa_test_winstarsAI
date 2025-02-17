from typing import Any

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.best_loss: float = float('inf')
        self.counter: int = 0

    def __call__(self, val_loss: float, model: Any) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                return True
        return False

