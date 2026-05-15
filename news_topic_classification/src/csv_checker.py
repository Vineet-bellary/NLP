import pandas as pd

import config as config
from util.load_data import load_data
from util.preprocess import preprocess_text

train_df_path = config.TRAIN_DATA
test_df_path = config.TEST_DATA
CLASS_MAP = config.CLASS_MAP

train = load_data(train_df_path).rename(
    columns={
        "Class Index": "class_index",
        "Title": "title",
        "Description": "description",
    }
)

train["text"] = train.apply(
    lambda row: preprocess_text(title=row["title"], description=row["description"]),
    axis=1,
)
test = load_data(test_df_path).rename(
    columns={
        "Class Index": "class_index",
        "Title": "title",
        "Description": "description",
    }
)

test["text"] = test.apply(
    lambda row: preprocess_text(title=row["title"], description=row["description"]),
    axis=1,
)

print("=" * 120)
print("Class Map:", CLASS_MAP)
print("=" * 120)

print("Checking CSV files...")

print("=" * 120)
print("TRAIN_DATA:")
print("-" * 120)
print(train.iloc[0])
print("=" * 120)

print("TEST_DATA:")
print("-" * 120)
print(test.iloc[0])
print("=" * 120)

print("=" * 120)
print("Train columns: ", train.columns.tolist())
print("Test columns: ", test.columns.tolist())
print("=" * 120)

print("Missing values in Train data:")
print(train.isna().sum())
print("-" * 120)
print("Missing values in Test data:")
print(test.isna().sum())
print("=" * 120)
