
# OpenRoGrid

Open dataset and reproducible ML benchmark for **short-term load forecasting (STLF)** of the Romanian national grid at 15-minute resolution, built entirely from open sources. Companion code for the paper *OpenRoGrid: A Public Dataset and Machine Learning Benchmark for Romanian Electricity Load Forecasting* (under review, SYNASC 2026).

## What we did

An end-to-end day-ahead forecasting pipeline:

- **Acquisition** — aggregated national load (ENTSO-E), weather (Open-Meteo), Romanian bank-holiday calendar
- **Cleaning** — uniform 15-min resampling, anomaly detection (7.5% rolling-median volatility threshold), PCHIP imputation
- **Features** — population-weighted weather aggregates, continuous Fourier temporal encodings, holiday-proximity vectors, autoregressive lags (24h / 48h / 7-day)
- **Models** — a stateful ARIMAX baseline vs. XGBoost, compared under a walk-forward rolling-inference framework over April 2026

XGBoost reaches **5.16% MAPE** (356 MW RMSE) day-ahead, beating the ARIMAX baseline (8.47%, 554 MW) while cutting training from ~20 min to ~1 min.

## Data

`romania_grid_data_with_weather_25-12-2022_30-04-2026.csv`

- **Range:** 2022-12-25 → 2026-04-30, **15-min** resolution (~117k rows)
- **Target:** aggregated national load (MW)
- **Covariates:** 2 m & apparent temperature, wind speed, shortwave radiation, precipitation, humidity, plus bank-holiday calendar features

Weather columns are **archived day-ahead forecasts**, not realized measurements, so the dataset stays leakage-free for inference-time use.

## Setup

### Dependencies — [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

### Services — Docker

TimescaleDB (storage) and pgAdmin (inspection):

```bash
docker compose up -d
```

| Service     | Address            | Credentials                     |
| ----------- | ------------------ | ------------------------------- |
| TimescaleDB | `localhost:5432` | `sebastian` / `admin`       |
| pgAdmin     | `localhost:5050` | `admin@local.dev` / `admin` |

### Schema — Alembic

```bash
uv run alembic upgrade head
```

Migrations set up the TimescaleDB hypertable. Then load the CSV and run training.
