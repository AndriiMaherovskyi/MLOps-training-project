import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///C:/mlops_lab_1/mlflow.db")

EXPERIMENT_NAME = "XGBoost_HPO"

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    client = MlflowClient()
    experiments = client.search_experiments()
    print("All experiments:", [e.name for e in experiments])
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")

runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
if runs_df.empty:
    raise ValueError(f"No runs found in experiment '{EXPERIMENT_NAME}'")

metric_col = [c for c in runs_df.columns if "val_rmse" in c.lower()]
if not metric_col:
    raise ValueError("Not found column of metric 'val_rmse' in runs")
metric_col = metric_col[0]

def get_sampler(run):
    return run["tags.sampler"] if "tags.sampler" in run else "unknown"

runs_df["sampler"] = runs_df.apply(get_sampler, axis=1)

runs_df = runs_df[runs_df[metric_col].notna()]

summary = runs_df.groupby("sampler")[metric_col].agg(
    best="min",
    mean="mean",
    median="median",
    std="std",
    count="count"
)
print("\nSampler summary:\n")
print(summary)

plt.figure(figsize=(8, 5))
for sampler in runs_df["sampler"].dropna().unique():
    df_sampler = runs_df[runs_df["sampler"] == sampler].sort_values("start_time")
    values = df_sampler[metric_col].values
    best_so_far = pd.Series(values).cummin()
    plt.plot(best_so_far, label=sampler, marker="o")

plt.xlabel("Trial")
plt.ylabel("Best RMSE so far")
plt.title("Convergence comparison of samplers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
for sampler in runs_df["sampler"].dropna().unique():
    df_sampler = runs_df[runs_df["sampler"] == sampler].sort_values("start_time")
    plt.scatter(range(len(df_sampler)), df_sampler[metric_col], label=sampler)

plt.xlabel("Trial")
plt.ylabel("Validation RMSE")
plt.title("All trials per sampler")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()