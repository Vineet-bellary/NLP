import config
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from util.data_prep import prepare_data
from util.logger import setup_logging

logger = setup_logging("eval")


def evaluate_model():
    model_state_path = config.LOGISTIC_REGRESSION_MODEL
    if not model_state_path.exists():
        logger.error(
            f"Model file not found at {model_state_path}. Please train the model first."
        )
        return

    model_save = joblib.load(model_state_path)
    logger.info(f"Model loaded from {model_state_path}")
    model = model_save["model"]
    vectorizer = model_save["vectorizer"]

    x_test, y_test = prepare_data(config.TEST_DATA)
    x_test_vectorized = vectorizer.transform(x_test)

    y_pred = model.predict(x_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")

    logger.info("Classification Report:")
    logger.info(
        "\n"
        + classification_report(y_test, y_pred, target_names=config.CLASS_MAP.values())
    )

    logger.info("Confusion Matrix:")
    logger.info("\n" + str(confusion_matrix(y_test, y_pred)))


def main():
    evaluate_model()


if __name__ == "__main__":
    main()
