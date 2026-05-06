import joblib
from pathlib import Path

from util.preprocessing import preprocess_text

MODEL_DIR_PATH = Path(__file__).resolve().parent / "models"
MODEL_NAME = "sentiment_analysis_model.pkl"

load = joblib.load(MODEL_DIR_PATH / MODEL_NAME)

model = load["model"]
vectorizer = load["vectorizer"]

x_test_raw = "What a horrible movie! I wasted my time watching it."

x_test_cleaned = preprocess_text(x_test_raw)

x_test = vectorizer.transform([x_test_cleaned])

y = model.predict(x_test)

print(f"Review: {x_test_raw}")
print(f"Predicted Label: {y[0]}")
