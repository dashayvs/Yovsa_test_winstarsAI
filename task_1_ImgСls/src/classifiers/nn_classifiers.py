from task_1_ImgСls.src.classifiers.nn_architectures import CNN, FeedForwardNN
from task_1_ImgСls.src.classifiers.nn_base import BaseNNMnistClassifier


class CNNMnistClassifier(BaseNNMnistClassifier):
    def __init__(self):
        super().__init__(CNN())


class FeedForwardNNMnistClassifier(BaseNNMnistClassifier):
    def __init__(self):
        super().__init__(FeedForwardNN())
