from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

from fake_news_detector.config import LOGS_DIR
from fake_news_detector.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", mode="w")


def build_vectorizer() -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=100000,
    )

    return vectorizer


def vectorize_data(x_train, x_test):
    vectorizer = build_vectorizer()
    logger.info(f"Vectorizer built successfully: {vectorizer}")

    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    logger.info(f"X Train Vectorized Shape: {x_train_vec.shape}")
    logger.info(f"X Test Vectorized Shape: {x_test_vec.shape}")

    return x_train_vec, x_test_vec, vectorizer
