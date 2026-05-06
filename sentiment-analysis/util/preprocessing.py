import string
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import re

from util.load_data import load_data

STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"not", "no"}


def process_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text) 

    punctuation_table = string.punctuation.replace("?", "").replace("!", "")
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation_table + string.digits))
    text = " ".join(text.split())

    return text


def remove_stop_words(tokens: list) -> list:
    return [token for token in tokens if token not in STOP_WORDS]


def preprocess_text(text: str) -> str:
    text = process_text(text)
    tokens = text.split()
    tokens = remove_stop_words(tokens)
    return " ".join(tokens)


def preprocess_pipeline(path: Path) -> pd.DataFrame:
    review_analysis = None
    try:
        review_analysis = load_data(path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Invalid data format: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    if review_analysis is None or review_analysis.empty:
        print("No data to process.")
        return
    else:
        assert set(review_analysis["sentiment"].unique()).issubset(
            {"positive", "negative"}
        ), "Unexpected labels found"

        review_analysis["cleaned_review"] = review_analysis["review"].apply(
            preprocess_text
        )

        return review_analysis
