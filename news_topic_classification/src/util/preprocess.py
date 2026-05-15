import re


def preprocess_text(title: str, description: str) -> str:
    text = f"{title} {description}"

    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
