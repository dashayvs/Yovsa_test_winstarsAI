from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / 'data'
DATA_IMG_DIR = DATA_DIR / 'data_img'
DATA_NER_DIR = DATA_DIR / 'data_ner'

TRAIN_DIR = DATA_IMG_DIR / 'train'
TEST_DIR = DATA_IMG_DIR / 'test'
VAL_DIR = DATA_IMG_DIR / 'val'

TRAIN_NER = DATA_NER_DIR / 'train_data_ner.json'

MODEL_DIR = ROOT_DIR / 'models'
IMG_MODEL_PATH = MODEL_DIR / 'img_model.pth'
NER_MODEL_PATH = MODEL_DIR / 'ner_model'

UTILS_DIR = ROOT_DIR / 'utils'
CLASSES_PATH = UTILS_DIR / 'class_names.json'

