from typing import Any, Protocol

import pandas as pd


class WindForecastingModelProtocol(Protocol):
    """Protocol defining the required interface for all wind forecasting models."""

    @property
    def model_name(self) -> str:
        """Returns the human-readable name of the model."""
        ...

    @property
    def params(self) -> dict[str, Any]:
        """Returns the hyperparameters used for tracking."""
        ...

    def train(self, series: pd.Series) -> None:
        """Trains the model on the provided time series."""
        ...

    def forecast(self, steps: int) -> pd.Series:
        """Forecasts future values for the specified number of steps."""
        ...

    def log_model_to_mlflow(self, name: str = "model") -> None:
        """Saves the trained model artifact to the MLflow registry."""
        ...
