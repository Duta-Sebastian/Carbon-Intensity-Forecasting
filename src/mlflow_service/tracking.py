from typing import Any, Literal

import pandas as pd
from matplotlib.figure import Figure
from mlflow.data.pandas_dataset import from_pandas

import mlflow


class MLTrackingService:
    """
    Singleton service to manage MLflow tracking configuration and operations.
    Ensures that the tracking URI is only set once per application lifecycle.
    """

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "MLTrackingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, tracking_uri: str = "http://localhost:5000") -> None:
        if not getattr(self, "initialized", False):
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.autolog(log_models=True, log_datasets=False, exclusive=False)
            self.initialized = True

    def set_experiment(self, experiment_name: str) -> None:
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_figure(self, fig: Figure, file_name: str) -> None:
        mlflow.log_figure(fig, file_name)

    def set_tag(self, key: str, value: Any) -> None:
        """Sets a tag for the current active run."""
        mlflow.set_tag(key, value)

    def log_pandas_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        context: Literal["training", "testing", "evaluation"] = "training",
    ) -> None:
        dataset = from_pandas(df, name=name)
        mlflow.log_input(dataset, context=context)
