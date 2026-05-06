from pathlib import Path
import joblib
import config

MODEL_DIR = config.MODEL_DIR

saved = joblib.load(MODEL_DIR / config.MODEL_NAME)
vectorizer = saved["vectorizer"]
model = saved["model"]


def feature_importance(vectorizer, model, top_n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_positive = coefs.argsort()[-top_n:]
    top_negative = coefs.argsort()[:top_n]

    print("\nTop Positive Features:")
    print("=" * 60)
    for i in top_positive:
        print(f"{feature_names[i]}: {coefs[i]:.4f}")
    print("=" * 60)

    print("\nTop Negative Features:")
    print("=" * 60)
    for i in top_negative:
        print(f"{feature_names[i]}: {coefs[i]:.4f}")
    print("=" * 60)


top_n = 10
feature_importance(vectorizer, model, top_n)
