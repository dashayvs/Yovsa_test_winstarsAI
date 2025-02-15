from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

MODELS_DIR = ROOT_DIR / "models"

RF_CLS_PATH = MODELS_DIR / "rf_cls.joblib"
FNN_CLS_PATH = MODELS_DIR / "fnn_cls.pth"
CNN_CLS_PATH = MODELS_DIR / "cnn_cls.pth"
