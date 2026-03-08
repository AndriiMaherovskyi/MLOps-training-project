import os
import mlflow
import optuna
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from preprocessing import prepare_features, time_series_split
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///C:/mlops_lab_1/mlflow.db")

@hydra.main(config_path="../config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    
    df = pd.read_csv("data/prepared/train_full.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    X, y, preprocessor = prepare_features(df)
    X_train, X_val, y_train, y_val = time_series_split(X, y)

    mlflow.set_experiment("XGBoost_HPO")

    sampler_class = getattr(optuna.samplers, cfg.hpo.sampler)
    sampler = sampler_class()
    study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", cfg.model.n_estimators.min, cfg.model.n_estimators.max),
            "max_depth": trial.suggest_int("max_depth", cfg.model.max_depth.min, cfg.model.max_depth.max),
            "learning_rate": trial.suggest_float("learning_rate", cfg.model.learning_rate.min, cfg.model.learning_rate.max),
            "subsample": trial.suggest_float("subsample", cfg.model.subsample.min, cfg.model.subsample.max)
        }

        trial.set_user_attr("sampler", cfg.hpo.sampler)

        model = XGBRegressor(**params, random_state=cfg.seed, eval_metric="rmse", use_label_encoder=False)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.set_tag("sampler", cfg.hpo.sampler)

        return val_rmse

    study.optimize(objective, n_trials=cfg.hpo.n_trials, n_jobs=1)

    best_params = study.best_params
    print("Best params:", best_params)

    final_model = XGBRegressor(**best_params, random_state=cfg.seed, eval_metric="rmse", use_label_encoder=False)
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", final_model)
    ])
    final_pipeline.fit(X, y)

    model_name = cfg.mlflow.model_name
    with mlflow.start_run(run_name="Final_Model") as run:
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(final_pipeline, artifact_path="best_model")

        model_uri = f"runs:/{run.info.run_id}/best_model"
        registered_model = mlflow.register_model(model_uri, model_name)

        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging",
            archive_existing_versions=False
        )

        print(f"Final model '{model_name}' version {registered_model.version} registered in Staging")
        print(f"MLflow run_id: {run.info.run_id}")

if __name__ == "__main__":
    main()