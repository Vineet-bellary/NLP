import string
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd

from src.load_data import load_data

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def process_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())

    return text


def remove_stop_words(tokens: list) -> list:
    return [token for token in tokens if token not in STOP_WORDS]


def preprocess_text(text: str) -> str:
    text = process_text(text)
    tokens = text.split()
    tokens = remove_stop_words(tokens)
    return " ".join(tokens)


def preprocess_pipeline(path: Path = None) -> pd.DataFrame:
    spam_data = None
    try:
        spam_data = load_data(path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Invalid data format: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    if spam_data is None or spam_data.empty:
        print("No data to process.")
        return
    else:
        assert set(spam_data["label"].unique()).issubset(
            {"ham", "spam"}
        ), "Unexpected labels found"

        # print("Before preprocessing:\n")
        # data_summary(spam_data)

        spam_data["cleaned_message"] = spam_data["message"].apply(preprocess_text)

        # print("\nAfter preprocessing:")
        # data_summary(spam_data, 10)

        return spam_data
