from pathlib import Path
import joblib

from fake_news_detector.config import LOGS_DIR, MODELS_DIR
from fake_news_detector.util.models import (
    build_linear_svm_model,
    build_logistic_regression_model,
    build_naive_bayes_model,
)
from fake_news_detector.util.evaluate import evaluate_model
from fake_news_detector.util.data_prep import split_data, data_preparation
from fake_news_detector.util.vectorizer import vectorize_data
from fake_news_detector.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", mode="w")


def prepare_dataset():
    df = data_preparation()
    x_train, x_test, y_train, y_test = split_data(df)

    x_train_vec, x_test_vec, vectorizer = vectorize_data(x_train, x_test)

    return x_train_vec, x_test_vec, y_train, y_test, vectorizer


def compare_models():
    x_train_vec, x_test_vec, y_train, y_test, vectorizer = prepare_dataset()

    models = {
        "logistic_regression": build_logistic_regression_model(),
        "naive_bayes": build_naive_bayes_model(),
        "linear_svm": build_linear_svm_model(),
    }

    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(x_train_vec, y_train)
        logger.info(f"Evaluating {name}...")
        metrics = evaluate_model(model, x_test_vec, y_test)
        results[name] = metrics
        model_save = {
            "model": model,
            "vectorizer": vectorizer,
        }
        joblib.dump(model_save, MODELS_DIR / f"{name}.joblib")

    logger.info("Model comparison completed...")

    return results


if __name__ == "__main__":
    comparison = compare_models()
    for model_name, metrics in comparison.items():
        report = metrics["classification_report"]
        conf_matrix = metrics["confusion_matrix"]
        logger.info(f"Model: {model_name}")
        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
