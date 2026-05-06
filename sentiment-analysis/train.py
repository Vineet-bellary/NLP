from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import logging


from util.eval import evaluate_model, get_confusion_matrix
from util.preprocessing import preprocess_pipeline
from util.vectorizer import vectorize_texts
import config

# DIR_DATA = Path("data").resolve()
# MODEL_DIR = Path("models").resolve()
# DATA_FILE = DIR_DATA / "IMDB_dataset.csv"
DIR_DATA = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR
DATA_FILE = config.DATA_FILE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

sentiment_analysis = preprocess_pipeline(DATA_FILE)

vectorizer, x_train, x_test, y_train, y_test = vectorize_texts(sentiment_analysis)

model = LogisticRegression(max_iter=1000)
logging.info("Training the model...")
model.fit(x_train, y_train)
logging.info("Model training completed.")

logging.info("Evaluating the model...")
y_pred = model.predict(x_test)

evaluate_model(y_test, y_pred)
get_confusion_matrix(y_test, y_pred)

model_save = {"model": model, "vectorizer": vectorizer}

joblib.dump(model_save, MODEL_DIR / config.MODEL_NAME)
logging.info(f"Model saved: {MODEL_DIR}\\{config.MODEL_NAME}")
