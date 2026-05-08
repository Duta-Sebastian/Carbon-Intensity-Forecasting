from typing import Any, cast

import mlflow.statsmodels
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper


class WindSARIMAModel:
    """SARIMAX implementation conforming to the WindForecastingModelProtocol."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 96),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit: SARIMAXResultsWrapper | None = None
        self._current_step = 0

    @property
    def model_name(self) -> str:
        return "SARIMAX"

    @property
    def params(self) -> dict[str, Any]:
        return {"order": str(self.order), "seasonal_order": str(self.seasonal_order)}

    def _iteration_callback(self, xk: Any) -> None:
        """
        Instance method callback to avoid local variable pickling issues
        when MLflow's autolog attempts to serialize the statsmodels fit.
        """
        self._current_step += 1
        print(f"[{self.model_name}] Training iteration: {self._current_step}...")

        if mlflow.active_run():
            mlflow.log_metric(
                key="optimization_step",
                value=self._current_step,
                step=self._current_step,
            )

    def train(self, series: pd.Series) -> None:
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self._current_step = 0

        raw_fit = model.fit(disp=True, callback=self._iteration_callback)

        self.model_fit = cast(SARIMAXResultsWrapper, raw_fit)

        if hasattr(self.model_fit, "mlefit") and hasattr(
            self.model_fit.mlefit, "mle_settings"
        ):
            self.model_fit.mlefit.mle_settings["callback"] = None

        print(f"[{self.model_name}] Training complete in {self._current_step} steps!")

    def forecast(self, steps: int) -> pd.Series:
        if not self.model_fit:
            raise ValueError("Model must be trained before forecasting.")
        return self.model_fit.forecast(steps=steps)

    def log_model_to_mlflow(self, name: str = "model") -> None:
        if not self.model_fit:
            raise ValueError("Cannot log an untrained model.")

        mlflow.statsmodels.log_model(
            statsmodels_model=self.model_fit,
            name=name,
            pip_requirements=["statsmodels", "pandas", "mlflow"],
            metadata={
                "description": "Wind Generation Forecaster",
                "model_type": self.model_name,
                "note": "Generates 15-min interval predictions for country scale generation.",
            },
        )
