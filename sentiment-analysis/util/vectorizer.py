from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

RANDOM_STATE = 42
TEST_SIZE = 0.2


def vectorize_texts(dataframe: pd.DataFrame) -> tuple:

    x_messages = dataframe["cleaned_review"]
    y = dataframe["sentiment"]

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_messages, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)

    x_train = vectorizer.fit_transform(x_train_raw)
    x_test = vectorizer.transform(x_test_raw)

    return vectorizer, x_train, x_test, y_train, y_test
