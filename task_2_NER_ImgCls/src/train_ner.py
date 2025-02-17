import argparse
import json
import random

import spacy
from paths import NER_MODEL_PATH, TRAIN_NER
from spacy.training import Example
from spacy.util import minibatch


def train_ner(train_data, iterations=50):
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    best_loss = float("inf")
    no_improvement_count = 0

    optimizer = nlp.begin_training()

    for i in range(iterations):
        random.shuffle(examples)
        losses = {}

        for batch in minibatch(examples, size=4):
            nlp.update(batch, drop=0.5, losses=losses)

        loss_value = losses.get("ner", 0)
        print(f"Iteration {i + 1}, Loss: {loss_value}")

        if best_loss - loss_value > 0.001:
            best_loss = loss_value
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= 7:
            print(f"Early stopping triggered after {i + 1} iterations.")
            break

    nlp.to_disk(NER_MODEL_PATH)
    print(f"Model saved to {NER_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an NER model.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    args = parser.parse_args()

    with open(TRAIN_NER) as file:
        train_data = json.load(file)

    train_ner(train_data, args.iterations)
