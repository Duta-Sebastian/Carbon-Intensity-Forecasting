from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from core.types import RomanianCity
from data_pipeline.open_meteo.adapter import OpenMeteoAdapter
from data_pipeline.open_meteo.config import OpenMeteoSettings
from data_pipeline.open_meteo.manager import OpenMeteoManager
from data_pipeline.open_meteo.repository import OpenMeteoRepository
from database.schemas.open_meteo import WeatherDataSchema


class OpenMeteoService:
    """Orchestrates fetching, transforming, and persisting Open-Meteo data."""

    def __init__(
        self,
        manager: OpenMeteoManager,
        session: AsyncSession,
        settings: OpenMeteoSettings,
    ) -> None:
        self.manager = manager
        self.settings = settings
        self.adapter = OpenMeteoAdapter()
        self.repository = OpenMeteoRepository(session)

    async def sync_historical_forecast(
        self,
        start_date: datetime,
        end_date: datetime,
        cities: list[RomanianCity] | None = None,
    ) -> list[WeatherDataSchema] | None:
        target_cities = cities or list(RomanianCity)

        params = {
            "latitude": [city.lat for city in target_cities],
            "longitude": [city.lon for city in target_cities],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "minutely_15": [
                "temperature_2m",
                "wind_speed_10m",
                "shortwave_radiation",
                "precipitation",
                "apparent_temperature",
                "relative_humidity_2m",
            ],
        }

        responses = await self.manager.fetch_weather_data(
            url=self.settings.forecast_url, params=params
        )

        if not responses:
            return None

        schemas: list[WeatherDataSchema] = self.adapter.transform(
            responses=responses, cities=target_cities
        )

        await self.repository.insert_weather_data(schemas)

        return schemas
