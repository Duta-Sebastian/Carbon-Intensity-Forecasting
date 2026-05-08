from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error

from core.types import CountryCode
from forecasting.wind.model_protocol import WindForecastingModelProtocol
from forecasting.wind.plotting import ForecastPlotter
from forecasting.wind.repository import WindForecastingRepository
from mlflow_service.tracking import MLTrackingService


class WindForecastingService:
    def __init__(
        self,
        repository: WindForecastingRepository,
        tracker: MLTrackingService,
        plotter: type[ForecastPlotter] = ForecastPlotter,
    ) -> None:
        self.repository = repository
        self.tracker = tracker
        self.plotter = plotter
        self.tracker.set_experiment("Wind_Generation_Forecasting")

    async def run_train_test_pipeline(
        self,
        model: WindForecastingModelProtocol,
        country_code: CountryCode,
        start_date: datetime,
        end_date: datetime,
        save_path: str | None = None,
    ) -> Figure | str:

        records = await self.repository.get_historical_wind_generation(
            country_code=country_code, start_date=start_date, end_date=end_date
        )
        if not records:
            raise ValueError("No data found for the selected period.")

        df = pd.DataFrame(
            [{"ts": r.timestamp, "val": r.generation_mw} for r in records]
        )
        df.set_index(pd.to_datetime(df["ts"]), inplace=True)
        series = df["val"].asfreq("15min").ffill()

        # split_idx = int(len(series) * 0.8)
        train = series.iloc[:-96]
        test = series.iloc[-96:]

        # Extract precise start and end times for dataset transparency
        train_start, train_end = train.index.min(), train.index.max()
        test_start, test_end = test.index.min(), test.index.max()

        run_name = (
            f"{model.model_name}_{country_code.value}_{start_date.strftime('%Y%m%d')}"
        )

        with self.tracker.start_run(run_name=run_name):
            # 1. Add an overarching Markdown note to the MLflow run
            run_note = (
                f"### Wind Generation Forecasting: {country_code.value}\n"
                f"**Model Type**: {model.model_name}\n\n"
                f"**Training Period**: {train_start} to {train_end}\n"
                f"**Testing Period**: {test_start} to {test_end}\n"
                f"**Frequency**: 15-minute intervals"
            )
            self.tracker.set_tag("mlflow.note.content", run_note)

            self.tracker.log_pandas_dataset(
                pd.DataFrame(train), f"{country_code.value}_train"
            )
            self.tracker.log_pandas_dataset(
                pd.DataFrame(test), f"{country_code.value}_test", context="testing"
            )

            # 2. Log exact boundaries to parameters for easy querying
            self.tracker.log_params(
                {
                    "model_type": model.model_name,
                    "country_code": country_code.value,
                    "train_size": len(train),
                    "test_size": len(test),
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                }
            )
            self.tracker.log_params(model.params)

            model.train(train)
            predictions = model.forecast(steps=len(test))
            predictions.index = test.index

            metrics = {
                "mse": mean_squared_error(test, predictions),
                "rmse": np.sqrt(mean_squared_error(test, predictions)),
                "mae": mean_absolute_error(test, predictions),
            }
            self.tracker.log_metrics(metrics)

            fig = self.plotter.plot_train_test_validation(
                train=train,
                test=test,
                predictions=predictions,
                model_name=model.model_name,
                country_code=country_code.value,
                metrics=metrics,
            )
            self.tracker.log_figure(fig, f"{model.model_name}_predicted_vs_actual.png")

            model.log_model_to_mlflow(name="model")

            if save_path:
                fig.savefig(save_path, dpi=300)
                plt.close(fig)
                return save_path

            return fig
