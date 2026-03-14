import os
import pandas as pd


def test_data_exists():
    path = "data/prepared/train_full.csv"

    assert os.path.exists(path), "Prepared dataset not found"


def test_data_schema():
    df = pd.read_csv("data/prepared/train_full.csv")
    required = {"Store", "Date", "Weekly_Sales"}
    missing = required - set(df.columns)

    assert not missing, f"Missing columns {missing}"
    assert df.shape[0] > 100