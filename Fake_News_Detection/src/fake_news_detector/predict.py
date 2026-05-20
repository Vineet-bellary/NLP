import joblib
from pathlib import Path

from fake_news_detector.config import (
    MODELS_DIR,
    LOGS_DIR,
    LOGISTIC_REGRESSION_MODEL,
    CLASS_MAP,
)
from fake_news_detector.util.logger import setup_logger
from fake_news_detector.util.preprocessing import preprocess_pipeline

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", mode="w")


def prediction_pipeline(texts: list[str]) -> list[str]:
    """A pipeline to preprocess text data and make predictions using the trained model"""
    model_save = joblib.load(MODELS_DIR / LOGISTIC_REGRESSION_MODEL)
    prediction = []

    if "model" not in model_save or "vectorizer" not in model_save:
        logger.error("Model or vectorizer not found in the saved file.")
        raise ValueError("Model or vectorizer not found in the saved file.")

    logger.info("Model and vectorizer loaded successfully.")
    model = model_save["model"]
    logger.info(f"Model Classes: {model.classes_}")
    vectorizer = model_save["vectorizer"]

    for text in texts:
        cleaned_text = preprocess_pipeline(text)
        text_vec = vectorizer.transform([cleaned_text])

        prediction_label = model.predict(text_vec)[0]
        prediction.append(CLASS_MAP[prediction_label])
        logger.info(f"\nText: {text[:45]}... -> prediction: {prediction[-1]:>10}\n")
    logger.info("Predictions made successfully...")

    return prediction


if __name__ == "__main__":
    sample_text = [
        "Breaking news: Scientists discover a new species of bird in the Amazon rainforest!",
        "The U.S. Department of Labor reported Thursday that unemployment claims fell slightly last week, signaling continued stability in the labor market despite ongoing concerns about inflation. According to government data, initial jobless claims decreased by 8,000 to a seasonally adjusted 215,000. Economists had expected claims to remain relatively unchanged. Analysts said the figures suggest employers are still retaining workers even as interest rates remain elevated.",
    ]
    prediction = prediction_pipeline(sample_text)
