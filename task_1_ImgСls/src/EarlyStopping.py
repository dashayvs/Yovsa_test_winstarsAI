from torch import nn


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_weights = None

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_weights = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False

    def load_best_model(self, model: nn.Module):
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
