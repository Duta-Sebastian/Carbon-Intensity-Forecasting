import asyncio
import datetime
import json
import os
import pickle

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.types import CITY_POPULATION_MAP
from forecasting.load.data_processor import LoadForecastingDataProcessor
from forecasting.load.repository import LoadForecastingRepository

WEATHER_COLS = [
    "apparent_temperature",
    "shortwave_radiation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "precipitation",
]
# DROP_COLS = [
#     "month_sin_1",
#     "month_cos_1",
#     "year_sin_1",
#     "year_cos_1",
#     "is_holiday",
#     "dt_since_prev_holiday_hrs",
#     "dt_until_next_holiday_hrs",
# ]


async def fetch_raw(
    repository: LoadForecastingRepository,
    processor: LoadForecastingDataProcessor,
    start: datetime.datetime,
    end: datetime.datetime,
) -> pd.DataFrame:
    load_res = await repository.get_historical_load_generation(
        start_date=start, end_date=end
    )
    weather_res = await repository.get_historical_weather_data(
        start_date=start, end_date=end
    )

    df_load = pd.DataFrame(
        [{"timestamp": r.timestamp, "load_mw": r.load_mw} for r in load_res]
    )
    df_weather = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "city": r.city,
                "apparent_temperature": r.apparent_temperature,
                "shortwave_radiation": r.shortwave_radiation,
                "wind_speed_10m": r.wind_speed_10m,
                "relative_humidity_2m": r.relative_humidity_2m,
                "precipitation": r.precipitation,
            }
            for r in weather_res
        ]
    )

    df_weather_nat = processor.aggregate_population_weighted_weather(
        df_weather, CITY_POPULATION_MAP, WEATHER_COLS
    )
    df = pd.merge(df_load, df_weather_nat, on="timestamp", how="inner")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_features(
    df_raw: pd.DataFrame,
    processor: LoadForecastingDataProcessor,
    fit_scaler: bool,
    start_dt: datetime.datetime,
) -> pd.DataFrame:
    df_feat = processor.apply_feature_engineering_pipeline(
        df_raw,
        "load_mw",
        2,
        1,
        1,
        1,
        with_scaling=False,
        fit_scaler=fit_scaler,
        with_autoregressive_features=False,
    )
    # df_feat.drop(columns=DROP_COLS, inplace=True)
    df_feat.set_index("timestamp", inplace=True)
    df_feat = df_feat.asfreq("15min", method="ffill")
    df_feat = df_feat[df_feat.index >= start_dt]
    return df_feat


def plot_day(
    date: datetime.date,
    timestamps,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    save_path: str,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(f"ARIMAX Forecast vs Actual — {date}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(timestamps, y_true, label="Actual", color="#60a5fa", linewidth=1.8)
    ax.plot(
        timestamps,
        y_pred,
        label="Forecast",
        color="#f87171",
        linewidth=1.8,
        linestyle="--",
    )
    ax.fill_between(timestamps, y_true, y_pred, alpha=0.12, color="#facc15")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.2)
    ax.set_title(
        "   ".join(f"{k.upper()}: {v:.2f}" for k, v in metrics.items()),
        fontsize=10,
        color="#555",
        pad=6,
    )

    error = y_pred - y_true
    axes[1].bar(
        timestamps,
        error,
        width=0.008,
        color=["#f87171" if e > 0 else "#60a5fa" for e in error],
        alpha=0.7,
    )
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("Error (MW)")
    axes[1].set_xlabel("Hour")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {save_path}")


async def walk_forward_month(
    year: int,
    month: int,
    model_path: str,
    repository: LoadForecastingRepository,
    processor: LoadForecastingDataProcessor,
    train_start: datetime.datetime,
    train_end: datetime.datetime,
    output_dir: str = "pipeline_output/walk_forward",
):
    plots_dir = os.path.join(output_dir, "plots")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model_res = model.model_res

    print(
        f"Fitting scaler on training data ({train_start.date()} → {train_end.date()})..."
    )
    train_buf = train_start - datetime.timedelta(days=7)
    df_train_raw = await fetch_raw(repository, processor, train_buf, train_end)
    processor.apply_feature_engineering_pipeline(
        df_train_raw,
        "load_mw",
        2,
        1,
        1,
        1,
        with_scaling=False,
        fit_scaler=False,
        with_autoregressive_features=False,
    )
    print("Scaler fitted.\n")

    first_day = datetime.date(year, month, 1)
    if month == 12:
        last_day = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        last_day = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

    days = [
        first_day + datetime.timedelta(days=i)
        for i in range((last_day - first_day).days + 1)
    ]

    all_metrics = []
    all_y_true = []
    all_y_pred = []
    all_timestamps = []

    for day in days:
        print(f"--- {day} ---")

        start_dt = datetime.datetime(
            day.year, day.month, day.day, 0, 0, tzinfo=datetime.timezone.utc
        )
        end_dt = datetime.datetime(
            day.year, day.month, day.day, 23, 45, tzinfo=datetime.timezone.utc
        )
        buf_start = start_dt - datetime.timedelta(days=7)

        df_raw = await fetch_raw(repository, processor, buf_start, end_dt)
        df_feat = build_features(df_raw, processor, fit_scaler=False, start_dt=start_dt)

        exog_cols = [c for c in df_feat.columns if c != "load_mw"]
        steps = len(df_feat)

        preds = model_res.forecast(steps=steps, exog=df_feat[exog_cols])

        y_true = df_feat["load_mw"].values
        y_pred = preds.values
        timestamps = df_feat.index

        metrics = model.evaluate(y_true, y_pred)
        print("  " + "   ".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items()))

        all_metrics.append({"date": str(day), **metrics})
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_timestamps.extend(timestamps)

        plot_path = os.path.join(plots_dir, f"{day}.png")
        plot_day(day, timestamps, y_true, y_pred, metrics, plot_path)  # type: ignore

        model_res = model_res.append(
            endog=y_true,
            exog=df_feat[exog_cols],
            refit=False,
        )
        print(f"  Model extended with {steps} actuals.\n")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    monthly_metrics = {
        "RMSE": float(np.sqrt(np.mean((all_y_pred - all_y_true) ** 2))),
        "MAE": float(np.mean(np.abs(all_y_pred - all_y_true))),
        "MAPE": float(np.mean(np.abs((all_y_pred - all_y_true) / all_y_true)) * 100),
    }

    print("=" * 50)
    print(f"APRIL {year} — MONTHLY SUMMARY")
    for k, v in monthly_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)

    metrics_path = os.path.join(metrics_dir, f"{year}-{month:02d}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"monthly": monthly_metrics, "daily": all_metrics}, f, indent=4)
    print(f"Metrics saved → {metrics_path}")

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle(
        f"ARIMAX Walk-Forward — April {year}  |  Monthly MAPE: {monthly_metrics['MAPE']:.2f}%  RMSE: {monthly_metrics['RMSE']:.1f} MW",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(
        all_timestamps,
        all_y_true,
        label="Actual",
        color="#60a5fa",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.plot(
        all_timestamps,
        all_y_pred,
        label="Forecast",
        color="#f87171",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
    )
    ax.set_ylabel("Load (MW)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    day_labels = [m["date"][5:] for m in all_metrics]  # MM-DD
    day_mapes = [m["MAPE"] for m in all_metrics]
    colors = ["#f87171" if m > 5 else "#4ade80" for m in day_mapes]
    ax2.bar(range(len(day_mapes)), day_mapes, color=colors, alpha=0.85)
    ax2.axhline(
        monthly_metrics["MAPE"],
        color="#facc15",
        linewidth=1.2,
        linestyle="--",
        label=f"Mean MAPE {monthly_metrics['MAPE']:.2f}%",
    )
    ax2.set_xticks(range(len(day_labels)))
    ax2.set_xticklabels(day_labels, rotation=45, fontsize=8)
    ax2.set_ylabel("MAPE (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    summary_path = os.path.join(output_dir, f"summary_{year}-{month:02d}.png")
    plt.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary plot saved → {summary_path}")

    return {"monthly": monthly_metrics, "daily": all_metrics}


if __name__ == "__main__":
    from core.config import AppSettings
    from database.manager import DatabaseManager

    async def main():
        app_settings = AppSettings()
        timescale_database = DatabaseManager(app_settings.db)
        timescale_database.initialize()

        train_start = datetime.datetime(2023, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        train_end = datetime.datetime(2026, 3, 31, 23, 45, tzinfo=datetime.timezone.utc)

        async for session in timescale_database.get_session():
            repository = LoadForecastingRepository(session)
            processor = LoadForecastingDataProcessor()

            await walk_forward_month(
                year=2026,
                month=4,
                model_path="pipeline_output_xgboost/models/arimax_base_specific.pkl",
                repository=repository,
                processor=processor,
                train_start=train_start,
                train_end=train_end,
                output_dir="pipeline_output/final_arimax_models/walk_forward/arimax_base_specific",
            )

    asyncio.run(main())
