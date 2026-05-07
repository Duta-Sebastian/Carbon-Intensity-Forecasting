from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from data_pipeline.entsoe.config import EntsoeSettings
from database.config import TimescaleDatabaseSettings


class AppSettings(BaseSettings):
    """Master Settings that delegates to sub-configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db: TimescaleDatabaseSettings = Field(default_factory=TimescaleDatabaseSettings)
    entsoe: EntsoeSettings = Field(default_factory=EntsoeSettings)


settings = AppSettings()
