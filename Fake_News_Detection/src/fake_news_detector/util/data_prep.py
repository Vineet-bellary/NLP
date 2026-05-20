from sklearn.model_selection import train_test_split

from fake_news_detector.util.data_audit import clean_and_combine_datasets
from fake_news_detector.util.preprocessing import preprocess_pipeline
from fake_news_detector.config import RANDOM_STATE, TEST_SIZE


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

    return x_train, x_test, y_train, y_test


def prepare_dataset() -> tuple:
    df = data_preparation()
    x_train, x_test, y_train, y_test = split_data(df)

    # x_train_vec, x_test_vec, vectorizer = vectorize_data(x_train, x_test)

    return x_train, x_test, y_train, y_test
