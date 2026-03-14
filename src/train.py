import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
import yaml
import json
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import prepare_features, time_series_split

# Load params from params.yaml
def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

# Training function
def train():
    # Load parameters
    params = load_params()

    model_type = params["model"]["model_type"]
    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]

    experiment_name = params["experiment"]["experiment_name"]
    author = params["experiment"]["author"]
    dataset_version = params["experiment"]["dataset_version"]

    # Base path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Use small sample for CI
    if os.getenv("CI", "false").lower() == "true":
        data_path = os.path.join(BASE_DIR, "../data/prepared/train_sample.csv")
    else:
        data_path = os.path.join(BASE_DIR, "../data/prepared/train_full.csv")

    train_df = pd.read_csv(data_path)
    train_df["Date"] = pd.to_datetime(train_df["Date"])

    # Feature engineering
    X, y, preprocessor = prepare_features(train_df)

    # Time series split
    X_train, X_val, y_train, y_val = time_series_split(X, y)

    # Model selection
    if model_type == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    elif model_type == "LightGBM":
        model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    elif model_type == "XGBoost":
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            use_label_encoder=False,
            eval_metric="rmse"
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Build pipeline
    pipeline_model = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # MLflow setup
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log tags
        mlflow.set_tag("author", author)
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("model_type", model_type)

        # Log hyperparameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train
        pipeline_model.fit(X_train, y_train)

        # Predict
        y_train_pred = pipeline_model.predict(X_train)
        y_val_pred = pipeline_model.predict(X_val)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        metrics = {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "val_r2": float(val_r2)
        }

        metrics_path = os.path.join(BASE_DIR, "../metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)

        # Log model
        model_name = f"{model_type.lower()}_timeseries_pipeline"
        mlflow.sklearn.log_model(pipeline_model, name=model_name)

        # Save model locally
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(BASE_DIR, f"../models/{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        mlflow.sklearn.save_model(pipeline_model, path=model_dir)

        model_path = os.path.join(BASE_DIR, "../model.pkl")
        joblib.dump(pipeline_model, model_path)

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(y_val, y_val_pred, alpha=0.3)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title("True vs Predicted Weekly Sales")
        plt.plot([y_val.min(), y_val.max()],
                 [y_val.min(), y_val.max()], 'r--')
        plt.tight_layout()
        plot_path = os.path.join(BASE_DIR, "../confusion_matrix.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # Feature importance
        model_obj = pipeline_model.named_steps["model"]
        preproc = pipeline_model.named_steps["preprocessor"]

        cat_cols = []
        if "cat" in preproc.named_transformers_:
            cat_cols = preproc.named_transformers_["cat"] \
                .get_feature_names_out().tolist()

        num_cols = [c for c in X_train.columns if c not in ["Type"]]
        all_cols = cat_cols + num_cols

        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
        else:
            importances = np.zeros(len(all_cols))

        feat_imp = dict(zip(all_cols, importances))
        sorted_feat = dict(sorted(feat_imp.items(),
                                  key=lambda x: x[1],
                                  reverse=True))

        plt.figure(figsize=(12, 8))
        plt.barh(
            list(sorted_feat.keys())[:20][::-1],
            list(sorted_feat.values())[:20][::-1]
        )
        plt.title("Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

    print(
        f"Train RMSE: {train_rmse:.2f}, "
        f"Val RMSE: {val_rmse:.2f}, "
        f"Val R2: {val_r2:.4f}"
    )


if __name__ == "__main__":
    train()