from datetime import datetime

from pydantic import BaseModel

from core.types import RomanianCity, WeatherDataProvider


class WeatherDataSchema(BaseModel):
    timestamp: datetime
    city: RomanianCity
    provider: WeatherDataProvider
    temperature_2m: float | None
    wind_speed_10m: float | None
    shortwave_radiation: float | None
    precipitation: float | None
    apparent_temperature: float | None
    relative_humidity_2m: float | None
