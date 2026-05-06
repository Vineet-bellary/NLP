import joblib
from pathlib import Path

from src.processing import preprocess_text

MODEL_DIR_PATH = Path(__file__).resolve().parent / "models"
MODEL_NAME = "spam_classifier_model.pkl"

load = joblib.load(MODEL_DIR_PATH / MODEL_NAME)

model = load["model"]
vectorizer = load["vectorizer"]

x_test_raw = "Call when you get home"

x_test_cleaned = preprocess_text(x_test_raw)

x_test = vectorizer.transform([x_test_cleaned])

y = model.predict(x_test)

print(f"Message: {x_test_raw}")
print(f"Predicted Label: {y[0]}")
