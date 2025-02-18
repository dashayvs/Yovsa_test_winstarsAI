import argparse

import torch
from img_classification.early_stopping import EarlyStopping
from img_classification.img_data_preprocessing import get_data_loader
from img_classification.model_img import create_model
from paths import IMG_MODEL_PATH, TRAIN_DIR, VAL_DIR
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train Image Classification Model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    return parser.parse_args()


def train_model(
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 5
) -> EfficientNet:
    """
    Trains the EfficientNet model on the provided dataset with early stopping and evaluation.

    This function performs training and validation of the model over a specified number of epochs,
    using early stopping to prevent overfitting. It saves the best model based on validation loss.
    """
    model = create_model().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=patience)

    best_model_wts = model.state_dict()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Validation loop
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_fn(val_outputs, val_labels).item()

                _, val_predicted = torch.max(val_outputs, 1)  # Get predicted class for validation
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        # Calculate and print validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total * 100

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Check for early stopping based on validation loss
        if early_stopping(val_loss, model):
            best_model_wts = model.state_dict()
            break

        model.train()  # Set the model back to training mode

    # Load the best model weights and save the model to disk
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), IMG_MODEL_PATH)
    print(f"Model saved to {IMG_MODEL_PATH}")
    return model


if __name__ == "__main__":
    args = parse_args()

    train_loader = get_data_loader(TRAIN_DIR, args.batch_size)
    val_loader = get_data_loader(VAL_DIR, args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_model(train_loader, val_loader, device, args.epochs, args.learning_rate, args.patience)
