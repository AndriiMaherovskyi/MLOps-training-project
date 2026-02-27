import pandas as pd
from pathlib import Path


def load_data(data_path="../data/raw/"):

    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "raw"

    features = pd.read_csv(data_path / "features.csv.zip")
    train = pd.read_csv(data_path / "train.csv.zip")
    stores = pd.read_csv(data_path / "stores.csv")

    return features, train, stores


def merge_data(features, train, stores):
    feature_store = features.merge(stores, how="inner", on="Store")

    feature_store["Date"] = pd.to_datetime(feature_store["Date"])
    train["Date"] = pd.to_datetime(train["Date"])

    feature_store["Day"] = feature_store["Date"].dt.day
    feature_store["Week"] = feature_store["Date"].dt.isocalendar().week.astype(int)
    feature_store["Month"] = feature_store["Date"].dt.month
    feature_store["Year"] = feature_store["Date"].dt.year

    train_df = train.merge(
        feature_store,
        how="inner",
        on=["Store", "Date", "IsHoliday"]
    )

    return train_df