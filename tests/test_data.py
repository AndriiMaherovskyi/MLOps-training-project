import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data/prepared/train_full.csv"

if os.getenv("CI", "false").lower() == "true":
    DATA_PATH = BASE_DIR / "data/prepared_sample/train_sample.csv"


def test_data_exists():
    assert DATA_PATH.exists(), f"Prepared dataset not found at {DATA_PATH}"


def test_data_schema():
    df = pd.read_csv(DATA_PATH)
    expected_cols = ["Date", "Store", "Dept", "Weekly_Sales", "Type", "Size"]
    for column in expected_cols:
        assert column in df.columns, f"Missing column {column}"
