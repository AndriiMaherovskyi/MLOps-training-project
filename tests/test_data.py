import os
from pathlib import Path
import pandas as pd
import pytest

# Base path
BASE_DIR = Path(__file__).resolve().parent.parent

# Default dataset path
DATA_PATH = BASE_DIR / "data/prepared/train_full.csv"

# Use sample in CI
if os.getenv("CI", "false").lower() == "true":
    DATA_PATH = BASE_DIR / "data/prepared_sample/train_sample.csv"


def test_data_exists():
    assert DATA_PATH.exists(), f"Prepared dataset not found at {DATA_PATH}"


def test_data_schema():
    df = pd.read_csv(DATA_PATH)
    expected_cols = ["Date", "Store", "Dept", "Weekly_Sales", "Type", "Size"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c}"