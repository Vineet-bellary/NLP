import pandas as pd
from pathlib import Path


def load_data(data_path: Path) -> pd.DataFrame:

    df = pd.read_csv(data_path)

    return df
