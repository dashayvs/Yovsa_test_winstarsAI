import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import EfficientNet

from img_classification.model_img import create_model
from paths import CLASSES_PATH, IMG_MODEL_PATH
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    return parser.parse_args()


def initialize_model() -> tuple[EfficientNet, str, list[str]]:
    with Path.open(CLASSES_PATH) as f:
        output_classes: list[str] = json.load(f)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: EfficientNet = create_model(output_shape=len(output_classes))

    model.load_state_dict(torch.load(IMG_MODEL_PATH))  # type: ignore[arg-type]
    model.to(device)
    model.eval()

    return model, device, output_classes


def predict(image_path: str | Path) -> str:
    model, device, output_classes = initialize_model()

    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return output_classes[predicted.item()]


if __name__ == "__main__":
    args = parse_args()

    predicted_class = predict(args.image_path)
    print(f"Predicted class: {predicted_class}")
