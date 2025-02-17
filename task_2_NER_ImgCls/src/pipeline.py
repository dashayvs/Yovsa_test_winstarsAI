
import argparse

import inflect

from ner.infer_ner import infer_ner
from img_classification.infer_img import predict


def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification and NER Inference Pipeline")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--text", type=str, required=True, help="Text to check if it matches the image description")
    return parser.parse_args()


def check(img_pth, text):
    img_class = predict(img_pth)

    entity = infer_ner(text)
    if entity:
        p = inflect.engine()
        if img_class == entity[0] or img_class == p.singular_noun(entity[0]):
            return True

    return False


if __name__ == "__main__":
    args = parse_args()

    print(check(args.image_path, args.text))
