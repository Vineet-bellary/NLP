from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

from util.eval import evaluate_model, get_confusion_matrix
from util.preprocessing import preprocess_pipeline
from util.vectorizer import vectorize_texts

DIR_DATA = Path("data").resolve()
MODEL_DIR = Path("models").resolve()
DATA_FILE = DIR_DATA / "IMDB_dataset.csv"

sentiment_analysis = preprocess_pipeline(DATA_FILE)

vectorizer, x_train, x_test, y_train, y_test = vectorize_texts(sentiment_analysis)

model = LogisticRegression(max_iter=1000)
print("Training the model...")
model.fit(x_train, y_train)
print("Model training completed.")

print("Evaluating the model...")
y_pred = model.predict(x_test)

evaluate_model(y_test, y_pred)
get_confusion_matrix(y_test, y_pred)

model_save = {"model": model, "vectorizer": vectorizer}

joblib.dump(model_save, MODEL_DIR / "sentiment_analysis_model.pkl") 
print(f"Model saved: {MODEL_DIR}\\sentiment_analysis_model.pkl")
