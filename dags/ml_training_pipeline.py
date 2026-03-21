from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.python import PythonSensor


PROJECT_ROOT = Path("/opt/airflow/project")
RAW_FILES = (
    PROJECT_ROOT / "data/raw/features.csv.zip",
    PROJECT_ROOT / "data/raw/train.csv.zip",
    PROJECT_ROOT / "data/raw/stores.csv",
    PROJECT_ROOT / "dvc.yaml",
)
METRICS_PATH = PROJECT_ROOT / "metrics.json"


def _data_is_ready() -> bool:
    return all(path.exists() for path in RAW_FILES)


def _choose_next_step(threshold: float = 0.5) -> str:
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    val_r2 = float(metrics["val_r2"])
    if val_r2 >= threshold:
        return "register_model"
    return "skip_registration"


@dag(
    dag_id="ml_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "dvc", "mlflow"],
)
def ml_training_pipeline():
    data_ready = PythonSensor(
        task_id="check_data_ready",
        python_callable=_data_is_ready,
        poke_interval=30,
        timeout=300,
        mode="poke",
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /opt/airflow/project && dvc repro prepare",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/project && dvc repro train",
    )

    evaluate_model = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=_choose_next_step,
        op_kwargs={"threshold": 0.5},
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/register_model.py --stage Staging"
        ),
    )

    skip_registration = EmptyOperator(task_id="skip_registration")
    pipeline_complete = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed_min_one_success",
    )

    data_ready >> prepare_data >> train_model >> evaluate_model
    evaluate_model >> register_model >> pipeline_complete
    evaluate_model >> skip_registration >> pipeline_complete


ml_training_pipeline()
