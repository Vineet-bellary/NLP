import pandas as pd
from pathlib import Path


def load_data(data_path: Path) -> pd.DataFrame:

    df = pd.read_csv(data_path)

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df


def get_class_map() -> dict:
    from config import CLASS_MAP

    return CLASS_MAP


def get_columns(df: pd.DataFrame) -> list:
    return df.columns.tolist()
