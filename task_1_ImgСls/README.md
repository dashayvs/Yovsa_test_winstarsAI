# MNIST Classification Project

This repository contains an example of how to implement three different classification models for the MNIST dataset (images of handwritten digits). The three models are:

1. **Random Forest**  
2. **Feed-Forward Neural Network (FNN)**  
3. **Convolutional Neural Network (CNN)**  

All three models implement a common interface (`MnistClassifierInterface`) and are wrapped by a single class (`MnistClassifier`) that allows you to switch between models by specifying the algorithm name (`"rf"`, `"nn"`, or `"cnn"`).

## Project Structure

```bash
task_1_ImgCls
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── models
│   ├── cnn_cls.pth                # Saved CNN model
│   ├── fnn_cls.joblib             # Saved Feed-Forward NN model
│   └── rf_cls.joblib              # Saved Random Forest model
├── notebooks
│   ├── gridsearch_rf.ipynb        # Notebook for Random Forest hyperparameter tuning
│   └── training.ipynb             # Main notebook for training the models
├── src
│   ├── classifiers
│   │   ├── __init__.py
│   │   ├── interface.py           # MnistClassifierInterface (train, predict)
│   │   ├── random_forest_classifier.py
│   │   ├── nn_architectures.py    # Helper modules for neural network architectures
│   │   ├── nn_classifiers.py      # Feed-Forward NN and CNN implementations
│   │   ├── nn_base                # Base class for FFNN and CNN
│   │   └── mnist_classifier.py    # Main wrapper class for rf, nn, cnn
│   ├── data_loader.py         # MNIST data loading function
│   ├── EarlyStopping.py       # Early stopping utility for NN training
└── └── paths.py               # file with all necessary paths
```

## Usage

Depending on the chosen algorithm, the `train()` method in `MnistClassifier` expects different input formats:

1. **Random Forest (`"rf"`)**  
   - The `train()` method expects your feature matrix `X` and labels `y` as NumPy arrays.  

2. **Neural Networks ("nn" or "cnn")**
   - For the Feed-Forward Neural Network and Convolutional Neural Network, the train() method expects two PyTorch DataLoader objects: one for training and one for validation.

