import joblib
from pathlib import Path

from util.preprocessing import preprocess_text
import config

MODEL_DIR_PATH = config.MODEL_DIR
MODEL_NAME = config.MODEL_NAME

load = joblib.load(MODEL_DIR_PATH / MODEL_NAME)

model = load["model"]
vectorizer = load["vectorizer"]

x_test_raw = "What a horrible movie! I wasted my time watching it."

x_test_cleaned = preprocess_text(x_test_raw)

x_test = vectorizer.transform([x_test_cleaned])

y = model.predict(x_test)

print(f"{'Review:':>16} {x_test_raw}")
print(f"{'Predicted Label:':>16} {y[0]}")
