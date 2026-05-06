from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "IMDB_dataset.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_NAME = "sentiment_analysis_model.pkl"
