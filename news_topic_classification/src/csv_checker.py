import pandas as pd

import config as config
from util.load_data import load_data, normalize_columns, get_class_map, get_columns
from util.preprocess import preprocess_text

train_df_path = config.TRAIN_DATA
test_df_path = config.TEST_DATA
CLASS_MAP = config.CLASS_MAP

train = load_data(train_df_path)
train = normalize_columns(train)

train["text"] = train.apply(
    lambda row: preprocess_text(title=row["title"], description=row["description"]),
    axis=1,
)

test = load_data(test_df_path)
test = normalize_columns(test)

test["text"] = test.apply(
    lambda row: preprocess_text(title=row["title"], description=row["description"]),
    axis=1,
)


def check_records(index: int = 0, split: str = "train") -> None:
    if split == "train":
        print(f"Record at index {index} in Train data:")
        print("-" * 120)
        print(train.iloc[index])
        print("=" * 120)
    elif split == "test":
        print(f"Record at index {index} in Test data:")
        print("-" * 120)
        print(test.iloc[index])
        print("=" * 120)

    return None


print("=" * 120)
print("Train columns: ", get_columns(train))
print("Test columns: ", get_columns(test))
print("=" * 120)


def missing_values_summary(df: pd.DataFrame) -> None:
    print(f"Missing values in {df}: ")
    print(df.isna().sum())
    print("-" * 120)

    return None


def main():
    check_records(index=0, split="train")
    check_records(index=0, split="test")

    class_maps = get_class_map()
    print("Class mapping:")
    for class_id, class_name in class_maps.items():
        print(f"{class_id} -> {class_name}")

    missing_values_summary(train)
    missing_values_summary(test)


if __name__ == "__main__":
    main()
