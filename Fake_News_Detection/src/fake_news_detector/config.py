from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"


# Dataset paths
TRAIN_DATA_PATH = DATA_DIR / "train.csv"

# Randomness controll
RANDOM_STATE = 42

# TFIDF Vectorizer parameters
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)

# Test train split
TEST_SIZE = 0.2
