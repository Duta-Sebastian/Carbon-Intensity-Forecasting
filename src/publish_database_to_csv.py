import asyncio
import datetime

import pandas as pd
from sqlalchemy import select

from core.config import AppSettings
from database.manager import DatabaseManager
from database.models import EnergyLoad
from database.models.WeatherData import WeatherData


async def main():
    app_settings = AppSettings()
    timescale_database = DatabaseManager(app_settings.db)
    timescale_database.initialize()

    start_train = datetime.datetime(2022, 12, 25, 0, 0, tzinfo=datetime.timezone.utc)
    end_train = datetime.datetime(2026, 4, 30, 23, 45, tzinfo=datetime.timezone.utc)

    async for session in timescale_database.get_session():
        load_query = (
            select(EnergyLoad.timestamp, EnergyLoad.load_mw)
            .where(
                EnergyLoad.timestamp >= start_train, EnergyLoad.timestamp <= end_train
            )
            .order_by(EnergyLoad.timestamp.asc())
        )

        load_result = await session.execute(load_query)
        df_load = pd.DataFrame(load_result.all(), columns=["timestamp", "load_mw"])
        df_load.set_index("timestamp", inplace=True)

        weather_query = (
            select(
                WeatherData.timestamp,
                WeatherData.city,
                WeatherData.temperature_2m,
                WeatherData.wind_speed_10m,
                WeatherData.shortwave_radiation,
                WeatherData.precipitation,
                WeatherData.apparent_temperature,
                WeatherData.relative_humidity_2m,
            )
            .where(
                WeatherData.timestamp >= start_train, WeatherData.timestamp <= end_train
            )
            .order_by(WeatherData.timestamp.asc())
        )

        weather_result = await session.execute(weather_query)
        df_weather = pd.DataFrame(weather_result.all())

        if not df_weather.empty:
            df_weather["city"] = df_weather["city"].apply(
                lambda x: x.name if hasattr(x, "name") else x
            )

        df_weather_wide = df_weather.pivot_table(
            index="timestamp",
            columns="city",
            values=[
                "temperature_2m",
                "wind_speed_10m",
                "shortwave_radiation",
                "precipitation",
                "apparent_temperature",
                "relative_humidity_2m",
            ],
        )

        df_weather_wide.columns = [
            f"{city}_{feature}" for feature, city in df_weather_wide.columns
        ]
        df_final = df_load.join(df_weather_wide, how="inner")

        df_final.index = pd.to_datetime(df_final.index)
        df_final.index.name = "timestamp"
        df_final.sort_index(ascending=True, inplace=True)

        csv_filename = "romania_grid_data_with_weather_25-12-2022_30-04-2026.csv"
        df_final.to_csv(
            f"{csv_filename}.zst", compression={"method": "zstd", "level": 22}
        )
        print(
            f"Data successfully exported to {csv_filename} with shape: {df_final.shape}"
        )


if __name__ == "__main__":
    asyncio.run(main())
