from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenMeteoSettings(BaseSettings):
    """Configuration for the Open-Meteo API."""

    forecast_url: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="OPENMETEO_",
    )
