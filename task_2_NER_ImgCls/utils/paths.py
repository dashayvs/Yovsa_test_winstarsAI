from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
VAL_DIR = DATA_DIR / 'val'

MODEL_DIR = ROOT_DIR / 'models'
IMG_MODEL_PATH = MODEL_DIR / 'img_model.pth'

UTILS_DIR = ROOT_DIR / 'utils'
CLASSES_PATH = UTILS_DIR / 'class_names.json'
