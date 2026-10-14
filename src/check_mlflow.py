
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

ROOT_DIR = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)

client = MlflowClient()

for experiment in client.search_experiments():
    print(f"\n{'=' * 60}")
    print("Experiment:", experiment.name)
    print("=" * 60)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=100,
    )

    print("Total runs:", len(runs))

    for run in runs:
        print(
            run.info.run_name,
            "| Status:", run.info.status,
            "| Metrics:", run.data.metrics
        )

