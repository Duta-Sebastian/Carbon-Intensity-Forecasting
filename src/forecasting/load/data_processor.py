import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class LoadForecastingDataProcessor:
    """
    Handles the transformation of raw time-series telemetry into a structured format
    suitable for statistical (ARIMAX) and deep learning (LSTM) forecasting models.
    """

    def __init__(self, frequency_minutes: int = 15):
        """
        Initializes the processing engine with a specific sampling frequency.

        Args:
            frequency_minutes (int): The data resolution in minutes.
                                     Standard ENTSO-E resolution is 15 or 60 minutes.
        """
        self.freq = frequency_minutes
        self.scaler = StandardScaler()
        self.daily_period = (24 * 60) / self.freq
        self.weekly_period = (7 * 24 * 60) / self.freq

    def encode_seasonal_fourier_features(
        self,
        df: pd.DataFrame,
        k_daily: int = 2,
        k_weekly: int = 1,
        k_monthly: int = 1,
        k_yearly: int = 1,
    ) -> pd.DataFrame:
        """
        Encodes daily, weekly, monthly, and annual seasonalities into a
        continuous Fourier space using sine and cosine transformations.
        """
        df = df.copy()

        if self.freq == 60:
            time_index = df["timestamp"].dt.hour
        else:
            time_index = df["timestamp"].dt.hour * (60 // self.freq) + (
                df["timestamp"].dt.minute // self.freq
            )

        for k in range(1, k_daily + 1):
            df[f"day_sin_{k}"] = np.sin(2 * np.pi * k * time_index / self.daily_period)
            df[f"day_cos_{k}"] = np.cos(2 * np.pi * k * time_index / self.daily_period)

        week_progress = (
            df["timestamp"].dt.dayofweek + time_index / self.daily_period
        ) / 7
        for k in range(1, k_weekly + 1):
            df[f"week_sin_{k}"] = np.sin(2 * np.pi * k * week_progress)
            df[f"week_cos_{k}"] = np.cos(2 * np.pi * k * week_progress)

        days_in_month = df["timestamp"].dt.days_in_month
        month_progress = (
            df["timestamp"].dt.day - 1 + time_index / self.daily_period
        ) / days_in_month
        for k in range(1, k_monthly + 1):
            df[f"month_sin_{k}"] = np.sin(2 * np.pi * k * month_progress)
            df[f"month_cos_{k}"] = np.cos(2 * np.pi * k * month_progress)

        days_in_year = np.where(df["timestamp"].dt.is_leap_year, 366, 365)
        year_progress = (
            df["timestamp"].dt.dayofyear - 1 + time_index / self.daily_period
        ) / days_in_year
        for k in range(1, k_yearly + 1):
            df[f"year_sin_{k}"] = np.sin(2 * np.pi * k * year_progress)
            df[f"year_cos_{k}"] = np.cos(2 * np.pi * k * year_progress)

        return df

    def encode_binary_calendar_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts discrete socioeconomic indicators that represent structural breaks
        in the load profile, such as weekend transitions and legal holidays.
        """
        df = df.copy()
        df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

        years = df["timestamp"].dt.year.unique().tolist()
        ro_holidays = holidays.Romania(years=years)

        df["is_holiday"] = df["timestamp"].dt.date.apply(
            lambda x: 1 if x in ro_holidays else 0
        )

        return df

    def encode_continuous_holiday_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ts_utc = (
            df["timestamp"].dt.tz_convert("UTC")
            if df["timestamp"].dt.tz is not None
            else df["timestamp"].dt.tz_localize("UTC")
        )

        years = df["timestamp"].dt.year.unique().tolist()
        buffered_years = [min(years) - 1] + years + [max(years) + 1]

        ro_holidays = holidays.Romania(years=buffered_years)

        holiday_dates = pd.to_datetime(list(ro_holidays.keys())).tz_localize("UTC")
        holiday_ns = holiday_dates.astype("int64")
        ts_ns = ts_utc.astype("int64")

        idx_next = np.searchsorted(holiday_ns, ts_ns)
        idx_next = np.clip(idx_next, 0, len(holiday_dates) - 1)
        idx_prev = np.clip(idx_next - 1, 0, len(holiday_dates) - 1)

        prev_holiday_series = pd.Series(holiday_dates[idx_prev], index=df.index)
        next_holiday_series = pd.Series(holiday_dates[idx_next], index=df.index)

        df["dt_since_prev_holiday_hrs"] = (ts_utc - prev_holiday_series) / pd.Timedelta(
            hours=1
        )
        df["dt_until_next_holiday_hrs"] = (next_holiday_series - ts_utc) / pd.Timedelta(
            hours=1
        )

        return df

    def create_autoregressive_lags(
        self, df: pd.DataFrame, target_col: str = "load"
    ) -> pd.DataFrame:
        """
        Augments the feature space with the historical lag vector (L_t) derived
        from ACF analysis. Represents exact load values from 24h, 48h, and 7d prior.
        """
        df = df.copy()

        steps_24h = int((24 * 60) / self.freq)
        steps_48h = int((48 * 60) / self.freq)
        steps_7d = int((7 * 24 * 60) / self.freq)

        df[f"{target_col}_lag_24h"] = df[target_col].shift(steps_24h)
        df[f"{target_col}_lag_48h"] = df[target_col].shift(steps_48h)
        df[f"{target_col}_lag_7d"] = df[target_col].shift(steps_7d)

        return df

    def aggregate_population_weighted_weather(
        self, weather_df: pd.DataFrame, population_map: dict, weather_cols: list
    ) -> pd.DataFrame:
        df = weather_df.copy()

        df["P_i"] = df["city"].map(population_map)

        for col in weather_cols:
            df[f"{col}_weighted"] = df[col] * df["P_i"]

        sum_cols = [f"{col}_weighted" for col in weather_cols] + ["P_i"]

        national_df = df.groupby("timestamp")[sum_cols].sum().reset_index()

        for col in weather_cols:
            national_df[col] = national_df[f"{col}_weighted"] / national_df["P_i"]

        return national_df[["timestamp"] + weather_cols]

    def scale_exogenous_features(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        df = df.copy()

        cols_to_scale = [c for c in df.columns if c not in ["timestamp", "load_mw"]]

        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        return df

    def apply_feature_engineering_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str = "load",
        k_days: int = 2,
        k_weeks: int = 1,
        k_months: int = 1,
        k_years: int = 1,
        with_autoregressive_features: bool = False,
        with_scaling: bool = False,
        fit_scaler: bool = False,
    ) -> pd.DataFrame:
        """
        Orchestrates the full feature engineering process, combining cyclical
        harmonics, discrete calendar effects, continuous holiday proximity, and
        autoregressive lags into a single dataset.

        Args:
            df (pd.DataFrame): Cleaned input DataFrame with 'timestamp' and target.
            target_col (str): The column representing the national electrical load.

        Returns:
            pd.DataFrame: Complete feature matrix ready for model training.
        """
        df = self.encode_seasonal_fourier_features(
            df, k_days, k_weeks, k_months, k_years
        )

        df = self.encode_binary_calendar_effects(df)

        df = self.encode_continuous_holiday_proximity(df)

        if with_autoregressive_features:
            df = self.create_autoregressive_lags(df, target_col=target_col)

        if with_scaling:
            df = self.scale_exogenous_features(df, fit=fit_scaler)

        return df
