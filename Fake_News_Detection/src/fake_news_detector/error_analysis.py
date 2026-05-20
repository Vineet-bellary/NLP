import pandas as pd
from pathlib import Path
import joblib

from fake_news_detector.config import MODELS_DIR, LOGS_DIR
from fake_news_detector.util.logger import setup_logger
from fake_news_detector.util.data_prep import prepare_dataset
from fake_news_detector.util.vectorizer import vectorize_data

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", "w")

logistic_regression_model_path = MODELS_DIR / "logistic_regression.joblib"
naive_bayes_model_path = MODELS_DIR / "naive_bayes.joblib"
linear_svm_model_path = MODELS_DIR / "linear_svm.joblib"

model_paths = [
    logistic_regression_model_path,
    naive_bayes_model_path,
    linear_svm_model_path,
]


def load_model(model_path: Path) -> object:
    """Load a model from the specified path."""
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        model_save = joblib.load(model_path)
        model = model_save["model"]
        vectorizer = model_save["vectorizer"]
    else:
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return model, vectorizer


def analyze_errors(model, X_test, X_test_vec, y_test):
    """Analyze the errors made by the model on the test set."""
    y_pred = model.predict(X_test_vec)

    # Create a DataFrame to analyze errors
    error_analysis_df = pd.DataFrame(
        {"text": X_test, "actual": y_test, "predicted": y_pred}
    )

    # Identify misclassified samples
    misclassified_df = error_analysis_df[
        error_analysis_df["actual"] != error_analysis_df["predicted"]
    ]

    false_positives = misclassified_df[
        (misclassified_df["actual"] == 0) & (misclassified_df["predicted"] == 1)
    ]

    false_negatives = misclassified_df[
        (misclassified_df["actual"] == 1) & (misclassified_df["predicted"] == 0)
    ]

    return false_positives, false_negatives


def main():
    _unused_x_train, x_test, _unused_y_train, y_test = prepare_dataset()

    for model_path in model_paths:
        model, vectorizer = load_model(model_path)
        x_test_vec = vectorizer.transform(x_test)
        false_positives, false_negatives = analyze_errors(
            model, x_test, x_test_vec, y_test
        )
        logger.info(f"Error analysis for Model: {model_path.stem}")
        logger.info(f"False Positives:\n{false_positives.head()}")
        logger.info(f"False Negatives:\n{false_negatives.head()}\n\n")


if __name__ == "__main__":
    main()
