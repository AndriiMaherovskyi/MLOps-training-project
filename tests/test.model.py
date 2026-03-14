import json
import os


def test_artifacts_exist():
    assert os.path.exists("model.pkl"), "model.pkl not found"
    assert os.path.exists("metrics.json"), "metrics.json not found"
    assert os.path.exists("confusion_matrix.png"), "plot not found"


def test_quality_gate():
    threshold = float(os.getenv("R2_THRESHOLD", "0.5"))
    with open("metrics.json") as f:
        metrics = json.load(f)
    r2 = metrics["val_r2"]

    assert r2 >= threshold, f"Quality gate failed: r2={r2}"