import pandas as pd


def load_raw_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)
