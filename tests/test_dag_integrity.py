import os
from pathlib import Path

from airflow.models import DagBag

os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")


def test_airflow_dagbag_imports_cleanly():
    dags_path = Path(__file__).resolve().parent.parent / "dags"
    dag_bag = DagBag(dag_folder=str(dags_path), include_examples=False)

    assert dag_bag.import_errors == {}
    assert "ml_training_pipeline" in dag_bag.dags
