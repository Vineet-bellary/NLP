from sklearn.model_selection import train_test_split
from pathlib import Path

from fake_news_detector.config import LOGS_DIR, RANDOM_STATE, TEST_SIZE
from fake_news_detector.util.logger import setup_logger
from fake_news_detector.util.data_audit import clean_and_combine_datasets
from fake_news_detector.util.preprocessing import preprocess_pipeline
from fake_news_detector.util.vectorizer import vectorize_data

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", mode="a")


def data_preparation():
    combined_df = clean_and_combine_datasets()
    combined_df["cleaned_text"] = combined_df["text"].apply(preprocess_pipeline)

    return combined_df


def split_data(df):
    x = df["cleaned_text"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=RANDOM_STATE, test_size=TEST_SIZE, stratify=y
    )

    logger.info(f"X Train Shape: {x_train.shape}")
    logger.info(f"X Test Shape: {x_test.shape}")
    logger.info(f"Y Train Shape: {y_train.shape}")
    logger.info(f"Y Test Shape: {y_test.shape}")
    return x_train, x_test, y_train, y_test


def main():
    df = data_preparation()

    logger.info(
        f"Data preparation completed. Sample data:\n{df[['text', 'cleaned_text']].head(3)}"
    )

    x_train, x_test, y_train, y_test = split_data(df)

    x_train_vec, x_test_vec, vectorizer = vectorize_data(x_train, x_test)
    logger.info(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
