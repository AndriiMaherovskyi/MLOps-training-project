import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import load_data, merge_data
from preprocessing import prepare_features, time_series_split

def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest on Walmart Time Series")
    parser.add_argument( "--model_type", type=str, default="RandomForest", choices=["RandomForest", "LightGBM", "XGBoost"], help="Model type to train")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in RF")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth of trees")
    parser.add_argument("--experiment_name", type=str, default="Walmart_TimeSeries", help="MLflow experiment name")
    parser.add_argument("--author", type=str, default="Andrii", help="Author tag for MLflow run")
    parser.add_argument("--dataset_version", type=str, default="v1", help="Dataset version tag")
    args = parser.parse_args()
    return args

def train(args):
    # Load & merge data
    features, train_data, stores = load_data()
    train_df = merge_data(features, train_data, stores)

    # Prepare features
    X, y, preprocessor = prepare_features(train_df)

    # Time series split
    X_train, X_val, y_train, y_val = time_series_split(X, y)
        
    if args.model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=42)
    elif args.model_type == "LightGBM":
        model = LGBMRegressor(n_estimators=args.n_estimators,
                            max_depth=args.max_depth,
                            random_state=42)
    elif args.model_type == "XGBoost":
        model = XGBRegressor(n_estimators=args.n_estimators,
                            max_depth=args.max_depth,
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric="rmse")

    # Build pipeline
    pipeline_model = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # MLflow experiment
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        # Log tags
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("model_type", args.model_type)

        # Log hyperparameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Train model
        pipeline_model.fit(X_train, y_train)

        # Predict
        y_train_pred = pipeline_model.predict(X_train)
        y_val_pred = pipeline_model.predict(X_val)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)

        # Log model
        model_name = f"{args.model_type.lower()}_timeseries_pipeline"
        mlflow.sklearn.log_model(pipeline_model, name=model_name)

        # Save model in global models folder with timestamp
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(BASE_DIR, f"../models/{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        mlflow.sklearn.save_model(pipeline_model, path=model_dir)

        # Plot y_true vs y_pred
        plt.figure(figsize=(12,6))
        plt.scatter(y_val, y_val_pred, alpha=0.3)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title("True vs Predicted Weekly Sales")
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
        plt.tight_layout()
        plt.savefig("y_true_vs_pred.png")
        mlflow.log_artifact("y_true_vs_pred.png")

        # Feature Importance
        model = pipeline_model.named_steps["model"]
        preproc = pipeline_model.named_steps["preprocessor"]

        # Categorical columns
        cat_cols = []
        if "cat" in preproc.named_transformers_:
            cat_cols = preproc.named_transformers_["cat"].get_feature_names_out().tolist()

        # Numeric columns
        num_cols = [c for c in X_train.columns if c not in ["Type"]]
        all_cols = cat_cols + num_cols

        # importances = model.feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(all_cols))
        feat_imp = dict(zip(all_cols, importances))
        sorted_feat = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

        plt.figure(figsize=(12,8))
        plt.barh(list(sorted_feat.keys())[:20][::-1], list(sorted_feat.values())[:20][::-1])
        plt.title("Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

    print(f"Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}, Val R2: {val_r2:.4f}")

if __name__ == "__main__":
    args = parse_args()
    train(args)