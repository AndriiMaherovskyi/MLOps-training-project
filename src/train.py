import datetime
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from preprocessing import prepare_features, time_series_split


def load_params():
    with open("params.yaml", encoding="utf-8") as params_file:
        return yaml.safe_load(params_file)


def get_mlflow_tracking_uri():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri

    base_dir = Path(os.getcwd())
    return (base_dir / "mlruns").as_uri()


def get_mlflow_artifact_root():
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")
    if artifact_root:
        if "://" in artifact_root:
            return artifact_root
        return Path(artifact_root).resolve().as_uri()

    base_dir = Path(os.getcwd())
    return (base_dir / "mlruns").as_uri()


def write_latest_run_info(base_dir: Path, run_id: str, model_name: str):
    latest_run_path = base_dir / "latest_run.json"
    latest_run_info = {
        "run_id": run_id,
        "model_name": model_name,
    }
    with latest_run_path.open("w", encoding="utf-8") as run_file:
        json.dump(latest_run_info, run_file, indent=2)


def build_model(model_type, n_estimators, max_depth):
    if model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
    if model_type == "LightGBM":
        return LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
    if model_type == "XGBoost":
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            eval_metric="rmse",
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def train():
    params = load_params()
    model_type = params["model"]["model_type"]
    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    experiment_name = params["experiment"]["experiment_name"]
    author = params["experiment"]["author"]
    dataset_version = params["experiment"]["dataset_version"]

    base_dir = Path(os.getcwd())
    if os.getenv("CI", "false").lower() == "true":
        data_path = base_dir / "data/prepared_sample/train_sample.csv"
    else:
        data_path = base_dir / "data/prepared/train_full.csv"

    train_df = pd.read_csv(data_path)
    train_df["Date"] = pd.to_datetime(train_df["Date"])

    X, y, preprocessor = prepare_features(train_df)
    X_train, X_val, y_train, y_val = time_series_split(X, y)

    pipeline_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("model", build_model(model_type, n_estimators, max_depth)),
        ]
    )

    tracking_uri = get_mlflow_tracking_uri()
    artifact_root = get_mlflow_artifact_root()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root,
        )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id
        mlflow.set_tag("author", author)
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("model_type", model_type)

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        pipeline_model.fit(X_train, y_train)

        y_train_pred = pipeline_model.predict(X_train)
        y_val_pred = pipeline_model.predict(X_val)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        metrics = {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "val_r2": float(val_r2),
        }

        metrics_path = base_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_artifact(str(metrics_path))

        model_name = f"{model_type.lower()}_timeseries_pipeline"
        mlflow.sklearn.log_model(
            sk_model=pipeline_model,
            artifact_path=model_name,
        )
        write_latest_run_info(base_dir, run_id, model_name)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = base_dir / "models" / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(pipeline_model, path=str(model_dir))

        model_path = base_dir / "model.pkl"
        joblib.dump(pipeline_model, model_path)

        plt.figure(figsize=(12, 6))
        plt.scatter(y_val, y_val_pred, alpha=0.3)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title("True vs Predicted Weekly Sales")
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
        plt.tight_layout()
        plot_path = base_dir / "true_vs_predicted_plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(str(plot_path))
        plt.close()

        model_obj = pipeline_model.named_steps["model"]
        preproc = pipeline_model.named_steps["preprocessor"]

        cat_cols = []
        if "cat" in preproc.named_transformers_:
            cat_cols = (
                preproc.named_transformers_["cat"]
                .get_feature_names_out()
                .tolist()
            )

        num_cols = [column for column in X_train.columns if column != "Type"]
        all_cols = cat_cols + num_cols

        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
        else:
            importances = np.zeros(len(all_cols))

        feat_imp = dict(zip(all_cols, importances))
        sorted_feat = dict(
            sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)
        )

        plt.figure(figsize=(12, 8))
        plt.barh(
            list(sorted_feat.keys())[:20][::-1],
            list(sorted_feat.values())[:20][::-1],
        )
        plt.title("Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        feature_path = base_dir / "feature_importance.png"
        plt.savefig(feature_path)
        mlflow.log_artifact(str(feature_path))
        plt.close()

    print(
        f"Train RMSE: {train_rmse:.2f}, "
        f"Val RMSE: {val_rmse:.2f}, "
        f"Val R2: {val_r2:.4f}"
    )


if __name__ == "__main__":
    train()
