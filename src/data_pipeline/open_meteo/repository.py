from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.WeatherData import WeatherData
from database.schemas.open_meteo import WeatherDataSchema

BATCH_SIZE = 2000


class OpenMeteoRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def insert_weather_data(self, data: list[WeatherDataSchema]) -> None:
        if not data:
            return

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            values = [item.model_dump(mode="python") for item in batch]

            stmt = insert(WeatherData).values(values)

            stmt = stmt.on_conflict_do_update(
                index_elements=["timestamp", "city", "provider"],
                set_={
                    "temperature_2m": stmt.excluded.temperature_2m,
                    "wind_speed_10m": stmt.excluded.wind_speed_10m,
                    "shortwave_radiation": stmt.excluded.shortwave_radiation,
                    "precipitation": stmt.excluded.precipitation,
                    "apparent_temperature": stmt.excluded.apparent_temperature,
                    "relative_humidity_2m": stmt.excluded.relative_humidity_2m,
                    "updated_at": func.now(),
                },
            )

            await self._session.execute(stmt)

        await self._session.commit()
