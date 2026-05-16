from pathlib import Path
import logging

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

TRAIN_DATA = DATA_DIR / "news_train_dataset.csv"
TEST_DATA = DATA_DIR / "news_test_dataset.csv"

LOGISTIC_REGRESSION_MODEL = MODEL_DIR / "logistic_regression_model.pkl"

CLASS_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}
