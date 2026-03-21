import json
import os


def test_artifacts_exist():
    assert os.path.exists("model.pkl"), "model.pkl not found"
    assert os.path.exists("metrics.json"), "metrics.json not found"
    assert os.path.exists("true_vs_predicted_plot.png"), "plot not found"


def test_latest_run_metadata_when_present():
    if not os.path.exists("latest_run.json"):
        return

    with open("latest_run.json", encoding="utf-8") as run_file:
        run_info = json.load(run_file)

    assert "run_id" in run_info and run_info["run_id"], "run_id is missing"
    assert "model_name" in run_info and run_info["model_name"], "model_name is missing"


def test_quality_gate():
    threshold = float(os.getenv("R2_THRESHOLD", "0.5"))
    with open("metrics.json", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    r2 = metrics["val_r2"]
    assert r2 >= threshold, f"Quality gate failed: r2={r2}"
