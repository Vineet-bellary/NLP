import pandas as pd
from pathlib import Path

from fake_news_detector.util.logger import setup_logger
from fake_news_detector.util.load_data import load_data
from fake_news_detector.config import LOGS_DIR, DATA_DIR

logger = setup_logger(LOGS_DIR / f"{Path(__file__).stem}.log", mode="w")


def combine_datasets(fake_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine fake and true news datasets and assign labels.

    Fake news -> 1
    True news -> 0
    """

    fake_df = fake_df.copy()
    true_df = true_df.copy()

    fake_df["label"] = 1
    true_df["label"] = 0

    # Combining
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)

    return combined_df


def log_combined_data_summary(df: pd.DataFrame) -> None:
    logger.info("Combined Data Summary:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Null Counts:\n{df.isnull().sum()}")
    logger.info(f"Duplicate Count: {df.duplicated().sum()}")
    logger.info(f"Label Distribution:\n{df['label'].value_counts()}")

    return None


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["title", "text"])


def check_for_conflicting_duplicates(combined_df: pd.DataFrame) -> bool:
    duplicates = combined_df[combined_df.duplicated(subset=["title", "text"])]
    if not duplicates.empty:
        logger.info(f"Duplicate articles based on title + text:\n{len(duplicates)}")

    conflicting_duplicates = combined_df[
        combined_df.duplicated(subset=["title", "text"], keep=False)
    ]
    conflicting_duplicates = conflicting_duplicates.groupby(["title", "text"])[
        "label"
    ].nunique()
    conflicting_duplicates = conflicting_duplicates[conflicting_duplicates > 1]
    logger.info(f"Conflicting duplicate articles: " f"{len(conflicting_duplicates)}")

    if conflicting_duplicates.empty:
        safe_to_remove_duplicates = True
    else:
        safe_to_remove_duplicates = False
    return safe_to_remove_duplicates


def text_length_analysis(df: pd.DataFrame) -> None:
    text_length = df["text"].astype(str).str.split().str.len()
    stats = text_length.describe().round(2)
    logger.info(f"Text Length Statistics:\n{stats}")
    return None


def empty_text_analysis(df: pd.DataFrame) -> None:
    empty_texts = df[df["text"].str.strip() == ""]
    logger.info(f"Empty Text Articles: {len(empty_texts)}")
    return None


def subject_distribution_analysis(df: pd.DataFrame) -> None:
    if "subject" in df.columns:
        subject_counts = df.groupby("label")["subject"].value_counts()
        logger.info(f"Subject Distribution:\n{subject_counts}")
    else:
        logger.warning("No 'subject' column found for analysis.")
    return None


def clean_and_combine_datasets():
    fake_df, _unused = load_data(DATA_DIR / "Fake.csv")
    true_df, _unused = load_data(DATA_DIR / "True.csv")

    combined_df = combine_datasets(fake_df, true_df)
    log_combined_data_summary(combined_df)

    safe_to_remove_duplicates = check_for_conflicting_duplicates(combined_df)
    if safe_to_remove_duplicates:
        logger.info(
            "No conflicting duplicates found."
            " Data is clean."
            " Safely removing duplicates."
        )
        combined_df = remove_duplicates(combined_df)
    else:
        logger.warning(
            "Duplicates not removed due to conflicts."
            " Please review the log for details."
        )

    text_length_analysis(combined_df)
    empty_text_analysis(combined_df)
    subject_distribution_analysis(combined_df)
    logger.info(f"Final shape after removing duplicates: {combined_df.shape}")

    return combined_df


if __name__ == "__main__":
    clean_and_combine_datasets()
