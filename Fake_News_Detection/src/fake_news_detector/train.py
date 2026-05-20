from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from fake_news_detector.config import (
    LOGS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
    LOGISTIC_REGRESSION_MODEL,
)
from fake_news_detector.util.logger import setup_logger
from fake_news_detector.util.data_audit import clean_and_combine_datasets
from fake_news_detector.util.preprocessing import preprocess_pipeline
from fake_news_detector.util.vectorizer import vectorize_data
from fake_news_detector.util.models import build_logistic_regression_model
from fake_news_detector.util.evaluate import evaluate_model
from fake_news_detector.util.data_prep import data_preparation, split_data

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", mode="w")


def main():
    df = data_preparation()

    logger.info(
        f"Data preparation completed. Sample data:\n{df[['text', 'cleaned_text']].head(3)}"
    )

    x_train, x_test, y_train, y_test = split_data(df)
    logger.info(f"X Train Shape: {x_train.shape}")
    logger.info(f"X Test Shape: {x_test.shape}")
    logger.info(f"Y Train Shape: {y_train.shape}")
    logger.info(f"Y Test Shape: {y_test.shape}")

    x_train_vec, x_test_vec, vectorizer = vectorize_data(x_train, x_test)
    logger.info(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")

    model = build_logistic_regression_model()

    logger.info("Starting model training...")
    model.fit(x_train_vec, y_train)
    logger.info("Model training completed...")

    logger.info("Evaluating model performance...")
    evaluate_model(model, x_test_vec, y_test)
    logger.info(f"Model evaluation completed and saved to {LOGS_DIR / 'evaluate.log'}")

    save_model = {
        "model": model,
        "vectorizer": vectorizer,
    }

    joblib.dump(save_model, MODELS_DIR / LOGISTIC_REGRESSION_MODEL)
    logger.info(
        f"Model and Vectorizer saved to {MODELS_DIR / LOGISTIC_REGRESSION_MODEL}"
    )


if __name__ == "__main__":
    main()
