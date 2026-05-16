from pathlib import Path

from util.load_data import load_data, normalize_columns
from util.preprocess import preprocess_text

def prepare_data(data_path: Path):
    df = load_data(data_path)
    df = normalize_columns(df)
    df["text"] = df.apply(
        lambda row: preprocess_text(title=row["title"], description=row["description"]),
        axis=1,
    )

    X = df["text"]
    y = df["class_index"]
    return X, y
