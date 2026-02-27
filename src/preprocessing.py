import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def prepare_features(df):
    df = df.sort_values(["Store", "Dept", "Date"])

    # Target & features
    y = df["Weekly_Sales"]
    X = df.drop(columns=["Weekly_Sales", "Date"])

    # Gap filling
    markdown_cols = [col for col in X.columns if "MarkDown" in col]
    X[markdown_cols] = X[markdown_cols].fillna(0)

    X["CPI"] = X["CPI"].fillna(X["CPI"].median())
    X["Unemployment"] = X["Unemployment"].fillna(X["Unemployment"].median())

    # Seasonality features
    X["Quarter"] = X["Month"].apply(lambda x: (x-1)//3 + 1)
    X["WeekOfYear"] = X["Week"]

    # Lag features – last 1–4 weeks of sales
    lags = [1, 2, 3, 4]
    for lag in lags:
        X[f"lag_{lag}"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag).fillna(0)

    # holiday features
    X["IsHoliday"] = X["IsHoliday"].astype(int)  # binary
    # week before holiday
    X["HolidayPrevWeek"] = df.groupby(["Store", "Dept"])["IsHoliday"].shift(1).fillna(0)
    # week after holiday
    X["HolidayNextWeek"] = df.groupby(["Store", "Dept"])["IsHoliday"].shift(-1).fillna(0)

    # Categorical encoding
    categorical_cols = ["Type"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ]
    )

    return X, y, preprocessor


def time_series_split(X, y):
    train_mask = X["Year"] < 2012
    val_mask = X["Year"] == 2012

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    return X_train, X_val, y_train, y_val