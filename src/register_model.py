import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from train import get_mlflow_tracking_uri, load_params


def load_latest_run_info():
    latest_run_path = Path("latest_run.json")
    if not latest_run_path.exists():
        return None

    with latest_run_path.open(encoding="utf-8") as run_file:
        return json.load(run_file)


def register_latest_model(stage: str) -> tuple[str, str]:
    params = load_params()
    model_type = params["model"]["model_type"]
    experiment_name = params["experiment"]["experiment_name"]
    model_name = f"{model_type.lower()}_timeseries_pipeline"

    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' does not exist in MLflow."
        )

    latest_run_info = load_latest_run_info()
    if latest_run_info:
        run_id = latest_run_info["run_id"]
        model_name = latest_run_info.get("model_name", model_name)
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(
                f"No runs found in experiment '{experiment_name}' for registration."
            )
        run_id = runs[0].info.run_id

    model_uri = f"runs:/{run_id}/{model_name}"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage=stage,
        archive_existing_versions=False,
    )

    return model_name, registered_model.version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()

    model_name, version = register_latest_model(stage=args.stage)
    print(
        f"Registered model '{model_name}' version {version} "
        f"and moved it to stage '{args.stage}'."
    )


if __name__ == "__main__":
    main()
