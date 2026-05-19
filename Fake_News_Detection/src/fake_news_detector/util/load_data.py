import pandas as pd
from pathlib import Path

from fake_news_detector.util.logger import setup_logger
from fake_news_detector.config import DATA_DIR, LOGS_DIR

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log")


def load_data(file_path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(file_path)
    file_name = file_path.stem
    return df, file_name


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df


def get_columns(df: pd.DataFrame) -> list:
    return df.columns.tolist()


def get_null_counts(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def get_duplicate_count(df: pd.DataFrame) -> int:
    return df.duplicated().sum()


def log_data_summary(df: pd.DataFrame, dataset_name: str) -> None:
    logger.info(f"{dataset_name} - Data Summary:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {get_columns(df)}")
    logger.info(f"Null Counts:\n{get_null_counts(df)}")
    logger.info(f"Duplicate Count: {get_duplicate_count(df)}")

    return None


if __name__ == "__main__":
    datasets = ["Fake.csv", "True.csv"]
    for file in datasets:
        df, file_name = load_data(DATA_DIR / file)
        df = normalize_columns(df)
        logger.info(f"{file_name} loaded{'...'*5}")
        log_data_summary(df, file_name)
