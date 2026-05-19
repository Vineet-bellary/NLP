import re
from pathlib import Path

from fake_news_detector.config import LOGS_DIR
from fake_news_detector.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", mode="w")

"""
    Preprocess:
    1. Lower
    2. Remove punctuation
    3. Remove extra spaces
    4. Remove URLs
    5. Remove HTML tags
"""


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def test():
    """Testing the clean_text function"""
    sample_text = [
        "Check out this link: https://example.com! <b>Bold Text</b>   Extra spaces.",
        "THIS IS ALL CAPS!!!",
        "100% GUARANTEED!!!",
        "Breaking-News---TODAY",
        "Email me at abc@gmail.com",
    ]

    cleaned_text = [clean_text(text) for text in sample_text]

    results = {
        "Original": sample_text,
        "Cleaned": cleaned_text,
    }

    return results

def preprocess_pipeline(text: str) -> str:
    """A pipeline to preprocess text data"""
    

    return clean_text(text)

if __name__ == "__main__":
    results = test()
    for key, value in results.items():
        logger.info(f"{key}: {value}")
