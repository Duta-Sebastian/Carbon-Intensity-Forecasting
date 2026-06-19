from sqlalchemy import Column, DateTime, Float, Index
from sqlalchemy import Enum as SQLEnum

from core.types import RomanianCity, WeatherDataProvider
from database.models.Base import Base


class WeatherData(Base):
    __tablename__ = "weather_data"

    timestamp = Column(DateTime(timezone=True), primary_key=True, index=True)
    city = Column(SQLEnum(RomanianCity), primary_key=True)
    provider = Column(SQLEnum(WeatherDataProvider), primary_key=True)
    updated_at = Column(DateTime(timezone=True))
    temperature_2m = Column(Float, nullable=False)
    wind_speed_10m = Column(Float, nullable=False)
    shortwave_radiation = Column(Float, nullable=False)
    precipitation = Column(Float, nullable=False)
    apparent_temperature = Column(Float, nullable=False)
    relative_humidity_2m = Column(Float, nullable=False)

    __table_args__ = (
        Index(
            "idx_weather_provider_time_city",
            "provider",
            "city",
            timestamp.desc(),
            unique=True,
        ),
    )
