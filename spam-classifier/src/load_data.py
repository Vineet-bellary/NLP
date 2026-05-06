import pandas as pd
from pathlib import Path


def load_data(file_path: Path | str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(
        path, sep=",", encoding="latin-1", usecols=[0, 1], header=0
    ).rename(columns={"v1": "label", "v2": "message"})

    return df


def data_summary(df: pd.DataFrame, preview_rows: int = 3) -> None:
    print("Data Summary")
    print("=" * 60)

    if df is None:
        print("Input is None")
        print("=" * 60)
        return

    if df.empty:
        print("DataFrame is empty")
        print(f"Columns: {df.columns.tolist()}")
        print("=" * 60)
        return

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print("-" * 60)

    print(f"Preview (top {preview_rows}):")
    print(df.head(preview_rows))
    print("-" * 60)

    missing_counts = df.isna().sum()
    print("Missing values per column:")
    print(
        missing_counts[missing_counts > 0]
        if (missing_counts > 0).any()
        else "No missing values"
    )
    print("-" * 60)

    duplicate_rows = int(df.duplicated().sum())
    print(f"Duplicate rows: {duplicate_rows}")
    print("-" * 60)

    if "label" in df.columns:
        label_counts = df["label"].value_counts(dropna=False)
        label_ratio = (label_counts / len(df) * 100).round(2)
        print("Label distribution (count):")
        print(label_counts)
        print("Label distribution (percent):")
        print(label_ratio.astype(str) + "%")
        print(
            f"Unique labels: {df['label'].nunique()} -> {df['label'].dropna().unique().tolist()}"
        )
        print("-" * 60)
    else:
        print("Column 'label' not found")
        print("-" * 60)

    if "message" in df.columns:
        msg_len = df["message"].fillna("").astype(str).str.len()
        print("Message length stats:")
        print(msg_len.describe().round(2))
        print("-" * 60)
    else:
        print("Column 'message' not found")
        print("-" * 60)

    print("Summary complete")
    print("=" * 60)
