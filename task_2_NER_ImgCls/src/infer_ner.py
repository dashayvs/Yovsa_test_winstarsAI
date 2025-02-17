import argparse

import spacy
from paths import NER_MODEL_PATH


def infer_ner(text: str) -> list:
    nlp = spacy.load(NER_MODEL_PATH)

    doc = nlp(text)

    return [ent.text for ent in doc.ents]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with a trained NER model.")
    parser.add_argument("--text", type=str, required=True, help="Text to perform NER on")
    args = parser.parse_args()

    entities = infer_ner(args.text)
    print(entities)
