import os
import pandas as pd
import pytest

path = "data/prepared/train_full.csv"

# Use sample in CI
if os.getenv("CI", "false").lower() == "true":
    path = "data/prepared/train_sample.csv"

def test_data_exists():
    assert os.path.exists(path), "Prepared dataset not found"

def test_data_schema():
    df = pd.read_csv(path)
    expected_cols = ["Date", "Store", "Dept", "Weekly_Sales", "Type", "Size"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c}"