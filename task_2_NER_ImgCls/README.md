# **NER + Image Classification Pipeline**
This project is focused on building a Machine Learning (ML) pipeline that combines two distinct models for different tasks:

- **Named Entity Recognition (NER) model**: This model extracts **ANIMAL** entity from a given text using a transformer-based approach. The model is built using **spaCy**.
- **Animal Classification model**: This model classifies animals in images into 10 categories. The model is built using the **EfficientNet** architecture.

The goal of the pipeline is to combine these two tasks to answer whether a text description matches the content of an image. The input consists of:

- A text message (e.g., "There is a cow in the picture.")
- An image (which contains an animal)
The pipeline will output a boolean value (True or False), indicating if the animal mentioned in the text is correctly classified in the image.

## Project Structure

```
task_2_NER_ImgCls
├── data
├── models
│   └── ner_model               # Saved NER model
│   └── img_model.pth           # Saved Image Classification model
├── notebooks
│   ├── EDA.ipynb                      # Exploratory Data Analysis notebook for dataset
│   └── demo.ipynb
├── src
│   ├── img_classification
│   │   ├── __init__.py
│   │   ├── early_stopping.py          # Early stopping utility for model training
│   │   ├── img_data_preprocessing.py  # Data preprocessing for image data
│   │   ├── model_img.py               # Image classification model training
│   │   └── infer_img.py               # Inference script for image classification
│   ├── __init__.py
│   ├── infer_ner.py                   # Inference script for NER model
│   ├── paths.py                       # Helper paths
│   ├── pipeline.py                    # Main pipeline to combine text and image input
│   ├── train_img.py                   # Training script for Image Classification model
│   ├── train_ner.py                   # Training script for NER model
├── utils
│   ├── data_loading.py                
│   └── class_names.json               # List of animal class names for classification
├── .gitignore
├── pyproject.toml
├── README.md                          # (This file)
└── requirements.txt
```
## Dataset

This project uses the Animals10 dataset, which is a collection of images representing 10 different animal categories. The dataset contains a variety of images and is intended for tasks such as image classification. It is publicly available on Kaggle and can be downloaded using the following link:
https://www.kaggle.com/datasets/alessiocorrado99/animals10/code?datasetId=59760&sortBy=voteCount&searchQuery=pyt

**Classes**: "butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"

---

**To use this dataset you should download raw images, divide it into train val and test, ypu can do it by running 'data_loading.py'**

---

## Usage

### 1. Training the Models
- **NER Model Training:**

To train the NER model, run the following script:
```python src/train_ner.py```

- **Image Classification Model Training:**

To train the image classification model, run:
```python src/train_img.py```

### 2. Running the Inference

- **NER Model Inference:**
To run inference for the NER model and extract animal names from text, use the following command:

```python src/infer_ner.py --text "There is a cow in the picture.```

- **Image Classification Inference:**
To run inference for the image classification model, use:

```python src/infer_img.py --image path/to/image.jpg```

### The Full Pipeline
The main script that ties both models together and outputs the result (True or False) is pipeline.py. This script takes a text input and an image input, processes both models, and outputs whether the text description matches the image.

```python src/pipeline.py --text "There is a cow in the picture." --image path/to/image.jpg```
