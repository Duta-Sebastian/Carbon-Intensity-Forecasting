import datetime
import json
import os
import pickle

import pandas as pd

from core.types import CITY_POPULATION_MAP
from forecasting.load.arimax import LoadForecastingARIMAX
from forecasting.load.data_processor import LoadForecastingDataProcessor
from forecasting.load.prediction_visualizer import (
    LoadForecastingPredictionVisualizer,
)
from forecasting.load.repository import LoadForecastingRepository


class LoadForecastingARIMAXPredictionService:
    def __init__(
        self,
        repository: LoadForecastingRepository,
        processor: LoadForecastingDataProcessor,
        start_date_train: datetime.datetime,
        end_date_train: datetime.datetime,
        start_date_test: datetime.datetime | None = None,
        end_date_test: datetime.datetime | None = None,
        fixed_k_params: tuple = (2, 1, 1, 1),
        output_dir: str = "output",
        with_scalar: bool = False,
        with_fit: bool = False,
        with_dropped_features: bool = True,
    ):
        self.repository = repository
        self.processor = processor
        self.start_date_train = start_date_train
        self.end_date_train = end_date_train
        self.start_date_test = start_date_test
        self.end_date_test = end_date_test
        self.k_params = fixed_k_params
        self.with_scalar = with_scalar
        self.with_fit = with_fit
        self.with_dropped_features = with_dropped_features

        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.models_dir = os.path.join(output_dir, "models")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.visualizer = LoadForecastingPredictionVisualizer(
            output_dir=os.path.join(output_dir, "plots")
        )

    async def _fetch_and_process_data(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> pd.DataFrame:

        buffer_start = start_date - datetime.timedelta(days=7)

        load_res = await self.repository.get_historical_load_generation(
            start_date=buffer_start, end_date=end_date
        )
        weather_res = await self.repository.get_historical_weather_data(
            start_date=buffer_start, end_date=end_date
        )

        if not load_res:
            raise ValueError(f"No load data found between {start_date} and {end_date}.")

        df_load = pd.DataFrame(
            [{"timestamp": row.timestamp, "load_mw": row.load_mw} for row in load_res]
        )

        df_weather = pd.DataFrame(
            [
                {
                    "timestamp": row.timestamp,
                    "city": row.city,
                    "temperature_2m": row.temperature_2m,
                    "apparent_temperature": row.apparent_temperature,
                    "shortwave_radiation": row.shortwave_radiation,
                    "wind_speed_10m": row.wind_speed_10m,
                    "relative_humidity_2m": row.relative_humidity_2m,
                    "precipitation": row.precipitation,
                }
                for row in weather_res
            ]
        )

        weather_cols = [
            "apparent_temperature",
            "shortwave_radiation",
            "wind_speed_10m",
            "relative_humidity_2m",
            "precipitation",
        ]

        df_weather_national = self.processor.aggregate_population_weighted_weather(
            df_weather, CITY_POPULATION_MAP, weather_cols
        )

        df = pd.merge(df_load, df_weather_national, on="timestamp", how="inner")

        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        k_days, k_weeks, k_months, k_years = self.k_params
        processed_df = self.processor.apply_feature_engineering_pipeline(
            df,
            "load_mw",
            k_days,
            k_weeks,
            k_months,
            k_years,
            with_scaling=self.with_scalar,
            fit_scaler=self.with_fit,
            with_autoregressive_features=False,
        )

        if self.with_dropped_features:
            processed_df = processed_df.drop(
                columns=[
                    "month_sin_1",
                    "month_cos_1",
                    "year_sin_1",
                    "year_cos_1",
                    "is_holiday",
                    "dt_since_prev_holiday_hrs",
                    "dt_until_next_holiday_hrs",
                ]
            )

        processed_df.set_index("timestamp", inplace=True)

        processed_df = processed_df.asfreq(f"{self.processor.freq}min", method="ffill")

        final_df = processed_df[processed_df.index >= start_date].copy()

        return final_df

    async def fit_and_save(
        self,
        model_filename: str = "arimax_base.pkl",
        method: str = "lbfgs",
        maxiter: int = 500,
        **fit_kwargs,
    ) -> str:
        """
        Orchestrates the full training pipeline: data fetching, engineering,
        model fitting with convergence optimization, and persistence.

        Args:
            model_filename: Filename for the pickled model.
            method: Optimization solver (e.g., 'lbfgs', 'cg').
            maxiter: Maximum iterations to force convergence.
            **fit_kwargs: Extra args for the model.fit() call.

        Returns:
            str: The filesystem path to the saved model.
        """
        print(
            f"Fetching training data ({self.start_date_train} to {self.end_date_train})..."
        )
        train_processed = await self._fetch_and_process_data(
            self.start_date_train, self.end_date_train
        )

        exog_cols = [col for col in train_processed.columns if col not in "load_mw"]

        print(f"Training model with {method} (maxiter={maxiter})...")
        model = LoadForecastingARIMAX(order=(5, 1, 1), seasonal_order=(0, 0, 0, 0))

        model.train(
            y_train=train_processed["load_mw"],
            exog_train=train_processed[exog_cols],
            method=method,
            maxiter=maxiter,
            **fit_kwargs,
        )

        print(model.summary())

        model_path = os.path.join(self.models_dir, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Model saved successfully to {model_path}")
        return model_path

    async def optimize_and_save_model(self, model_filename: str = "arimax_base.pkl"):
        print(
            f"Fetching training data ({self.start_date_train} to {self.end_date_train})..."
        )
        train_processed = await self._fetch_and_process_data(
            self.start_date_train, self.end_date_train
        )
        exog_cols = [
            col
            for col in train_processed.columns
            if col not in ["timestamp", "load_mw"]
        ]

        print("Optimizing and Training Model...")
        model = LoadForecastingARIMAX()
        model.optimize_and_train(
            y_train=train_processed["load_mw"], exog_train=train_processed[exog_cols]
        )

        model_path = os.path.join(self.models_dir, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Model optimized and saved to {model_path}")
        return model_path

    async def predict_range(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        model_filename: str = "arimax_scaled_with_dropped_features_with_lagsspecific.pkl",
    ) -> dict:
        """
        Predicts load for a given date range, compares to actuals,
        computes evaluation metrics, and plots results.

        Args:
            start_date: First timestamp to predict (inclusive).
            end_date:   Last timestamp to predict (inclusive).
            model_filename: Pickled model to load.

        Returns:
            dict with keys: 'forecast_df', 'metrics', 'plot_path', 'csv_path'
        """
        model_path = os.path.join(self.models_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Run fit_and_save first."
            )

        print(f"Loading trained model from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"Fetching data ({start_date} to {end_date})...")
        processed = await self._fetch_and_process_data(start_date, end_date)

        exog_cols = [col for col in processed.columns if col != "load_mw"]

        steps = len(processed)
        print(f"Forecasting {steps} steps ({start_date.date()} → {end_date.date()})...")
        predictions = model.forecast(steps=steps, exog_forecast=processed[exog_cols])

        y_true = processed["load_mw"].values
        y_pred = predictions.values

        print("Computing metrics...")
        metrics = model.evaluate(y_true, y_pred)
        print(f"Metrics: {metrics}")

        forecast_df = pd.DataFrame(
            {
                "timestamp": processed.index,
                "actual_mw": y_true,
                "forecast_mw": y_pred,
                "error_mw": y_pred - y_true,
            }
        )

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = os.path.join(self.results_dir, f"predict_range_{timestamp_str}.csv")
        forecast_df.to_csv(csv_path, index=False)
        print(f"Forecast saved to {csv_path}")

        metrics_path = os.path.join(
            self.metrics_dir, f"predict_range_metrics_{timestamp_str}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "model": model_filename,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "steps": steps,
                    "metrics": metrics,
                },
                f,
                indent=4,
            )

        plot_save_name = f"predict_range_{timestamp_str}.png"
        self.visualizer.plot_forecast_vs_actual(
            y_true,  # type: ignore
            predictions,
            title=f"ARIMAX Forecast vs Actual: {start_date.date()} to {end_date.date()}",
            save_name=plot_save_name,
        )
        plot_path = os.path.join(self.output_dir, "plots", plot_save_name)

        return {
            "forecast_df": forecast_df,
            "metrics": metrics,
            "plot_path": plot_path,
            "csv_path": csv_path,
        }

    async def walk_forward_validate(
        self, test_days: int, model_filename: str = "arimax_base.pkl"
    ) -> dict:
        model_path = os.path.join(self.models_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Run optimize_and_save_model first."
            )

        print(f"Loading trained base model from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        freq_mins = self.processor.freq
        steps_per_day = int((24 * 60) / freq_mins)
        total_test_steps = test_days * steps_per_day

        validation_start = self.end_date_train + datetime.timedelta(minutes=freq_mins)
        validation_end = validation_start + datetime.timedelta(days=test_days)

        print(f"Fetching validation data ({validation_start} to {validation_end})...")
        test_processed = await self._fetch_and_process_data(
            validation_start, validation_end
        )

        exog_cols = [
            col for col in test_processed.columns if col not in ["timestamp", "load_mw"]
        ]

        if len(test_processed) < total_test_steps:
            raise ValueError(
                f"Insufficient telemetry in database to walk forward {test_days} days."
            )

        print(f"Starting Walk-Forward loop for {test_days} consecutive days...")
        all_predictions = []
        all_actuals = []

        model_res = model.model_res

        for day in range(test_days):
            start_idx = day * steps_per_day
            end_idx = start_idx + steps_per_day

            current_test_chunk = test_processed.iloc[start_idx:end_idx]
            current_exog = current_test_chunk[exog_cols]
            current_actuals = current_test_chunk["load_mw"].values

            pred_chunk = model_res.forecast(steps=steps_per_day, exog=current_exog)

            all_predictions.extend(pred_chunk)
            all_actuals.extend(current_actuals)

            model_res = model_res.append(
                endog=current_actuals, exog=current_exog, refit=False
            )
            print(f"Extrapolated Day {day + 1}/{test_days}")

        print("Evaluating Backtest Results...")
        metrics = model.evaluate(all_actuals, all_predictions)
        print(f"Final Walk-Forward Metrics: {metrics}")

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_path = os.path.join(
            self.metrics_dir, f"walkforward_metrics_{timestamp_str}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(
                {"order": model.order, "test_days": test_days, "metrics": metrics},
                f,
                indent=4,
            )

        self.visualizer.plot_forecast_vs_actual(
            all_actuals,
            all_predictions,
            title=f"Walk-Forward Validation ({test_days} Days)",
            save_name=f"walkforward_{timestamp_str}.png",
        )

        return metrics
