import string

STOP_WORDS = {"the", "is", "in", "and", "to", "of", "a", "that", "it", "with", "for"}


def process_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    text = " ".join(text.split())

    return text


def tokenize(text) -> list:
    return text.split()


def count_words(tokens) -> dict:
    word_counts = {}
    for token in tokens:
        if token.strip() == "" or token in STOP_WORDS:
            continue
        word_counts[token] = word_counts.get(token, 0) + 1

    return word_counts


def top_n_words(word_counts: dict, n: int) -> list:
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

    return sorted_words[:n]
