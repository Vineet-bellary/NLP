from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib

from src.processing import preprocess_pipeline
from src.vectorizer import vectorize_texts
from src.eval import evaluate_model, get_confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_DIR_PATH = BASE_DIR / "models"


def main():

    spam_data = preprocess_pipeline(DATA_PATH)

    vectorizer, x_train, x_test, y_train, y_test = vectorize_texts(spam_data)

    # model = MultinomialNB()
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    evaluate_model(y_test, y_pred)

    get_confusion_matrix(y_test, y_pred)

    # Save the trained model and vectorizer
    saved_model = {
        "model": model,
        "vectorizer": vectorizer,
    }
    joblib.dump(saved_model, MODEL_DIR_PATH / "spam_classifier_model.pkl")


if __name__ == "__main__":
    main()
