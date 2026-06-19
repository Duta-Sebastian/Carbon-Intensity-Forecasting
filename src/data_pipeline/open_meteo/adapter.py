from typing import Any, cast

import numpy as np
import pandas as pd

from core.types import RomanianCity, WeatherDataProvider
from database.schemas.open_meteo import WeatherDataSchema


class OpenMeteoAdapter:
    def transform(
        self, responses: list[Any], cities: list[RomanianCity]
    ) -> list[WeatherDataSchema]:
        """
        Transforms Open-Meteo raw array responses into strict Pydantic models.
        """
        if not responses:
            return []

        all_schemas: list[WeatherDataSchema] = []

        for response, city_enum in zip(responses, cities):
            minutely_15 = response.Minutely15()

            data = {
                "temperature_2m": minutely_15.Variables(0).ValuesAsNumpy(),
                "wind_speed_10m": minutely_15.Variables(1).ValuesAsNumpy(),
                "shortwave_radiation": minutely_15.Variables(2).ValuesAsNumpy(),
                "precipitation": minutely_15.Variables(3).ValuesAsNumpy(),
                "apparent_temperature": minutely_15.Variables(4).ValuesAsNumpy(),
                "relative_humidity_2m": minutely_15.Variables(5).ValuesAsNumpy(),
            }

            timestamps = pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=minutely_15.Interval()),
                inclusive="left",
                name="timestamp",
            )

            df = pd.DataFrame(data=data, index=timestamps)

            expected_range = pd.date_range(
                start=df.index.min(), end=df.index.max(), freq="15min", name="timestamp"
            )
            df = df.reindex(expected_range)

            weather_columns = [
                "temperature_2m",
                "wind_speed_10m",
                "shortwave_radiation",
                "precipitation",
                "apparent_temperature",
                "relative_humidity_2m",
            ]
            for col in weather_columns:
                df[col] = self._smooth_weather_anomalies(df, col, city_enum.city)

            self._validate_15_min_intervals(df)

            df = df.reset_index()
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            df["city"] = pd.Series([city_enum] * len(df), dtype=object)
            df["provider"] = WeatherDataProvider.OPENMETEO

            df = df.where(pd.notnull(df), None)

            records = cast(list[dict[str, Any]], df.to_dict(orient="records"))
            all_schemas.extend([WeatherDataSchema(**r) for r in records])

        return all_schemas

    def _smooth_weather_anomalies(
        self, df: pd.DataFrame, col: str, city_name: str
    ) -> pd.Series:
        """
        Detects and interpolates weather anomalies using the 99.9th percentile
        variance and rolling median absolute deviation.
        """
        series = df[col].copy()

        diffs = series.diff().abs()

        valid_diffs = diffs[diffs > 0]

        p999 = 0.0
        if not valid_diffs.empty:
            p999 = valid_diffs.quantile(0.999)

        missing_mask = series.isna()

        rolling_median = series.rolling(window=5, center=True, min_periods=1).median()

        min_thresholds = {
            "temperature_2m": 3.0,
            "wind_speed_10m": 15.0,
            "shortwave_radiation": 250.0,
            "precipitation": 10.0,
            "apparent_temperature": 3.0,
            "relative_humidity_2m": 15.0,
        }

        actual_threshold = max(p999, min_thresholds.get(col, 2.0))

        sudden_change_mask = abs(series - rolling_median) > actual_threshold

        anomaly_mask = missing_mask | sudden_change_mask

        if anomaly_mask.any():
            print(
                f"--- Interpolating {anomaly_mask.sum()} anomalies for {city_name} in {col} ---"
            )

            series.loc[anomaly_mask] = np.nan
            series = self._fill_missing_intervals(series)

        return series

    def _fill_missing_intervals(self, series: pd.Series) -> pd.Series:
        """
        Applies the business logic for missing data:
        - Pchip Interpolation (Excellent for smooth weather curves like temperature)
        - Forward/Backward Fill for edge cases.
        """
        return series.interpolate(method="pchip").ffill().bfill()

    def _validate_15_min_intervals(self, df: pd.DataFrame) -> None:
        """Checks if the timestamp sequence is strictly 15-minute intervals."""
        if len(df) < 2:
            return

        ts = df.index.to_series()
        diffs = ts.diff().dropna()
        expected = pd.Timedelta(minutes=15)

        if not (diffs == expected).all():
            bad_indices = diffs[diffs != expected].index
            raise ValueError(
                f"Irregular time intervals detected at: {bad_indices.tolist()}"
            )
