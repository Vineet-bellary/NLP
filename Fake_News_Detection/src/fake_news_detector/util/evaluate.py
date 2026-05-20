from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from fake_news_detector.config import LOGS_DIR
from fake_news_detector.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", mode="w")


def evaluate_model(model, x_test_vec, y_test) -> dict:
    """Evaluate the model using classification report and confusion matrix"""
    y_pred = model.predict(x_test_vec)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Real News", "Fake News"],
    )
    conf_matrix = confusion_matrix(y_test, y_pred)

    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Real News", "Fake News"],
        columns=["Predicted Real", "Predicted Fake"],
    )
    logger.info("Classification Report:\n%s", report)

    logger.info("Confusion Matrix:\n%s", conf_matrix_df)

    return {
        "classification_report": report,
        "confusion_matrix": conf_matrix_df,
    }
