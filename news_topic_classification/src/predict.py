import config
import joblib

from util.preprocess import preprocess_text


def predict(text: str):
    model_state_path = config.LOGISTIC_REGRESSION_MODEL
    if not model_state_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_state_path}. Please train the model first."
        )

    model_save = joblib.load(model_state_path)
    model = model_save["model"]
    vectorizer = model_save["vectorizer"]

    processed_text = preprocess_text(title=text, description="")
    text_vectorized = vectorizer.transform([processed_text])

    prediction_index = model.predict(text_vectorized)[0]
    predicted_class = config.CLASS_MAP[prediction_index]

    return predicted_class


def main():
    sample_text = "Apple launches new AI-powered iPhone chipset"
    predicted_class = predict(sample_text)
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
