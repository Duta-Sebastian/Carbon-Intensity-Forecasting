from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EntsoeSettings(BaseSettings):
    """Configuration for the ENTSO-E API."""

    api_key: SecretStr = Field(
        description="API key for ENTSO-E. You can obtain it from https://transparency.entsoe.eu/ by registering an account.",
    )
    """
    API key for ENTSO-E. 
    You can obtain it from https://transparency.entsoe.eu/ by registering and requesting one by email.
    """

    model_config = SettingsConfigDict(
        env_prefix="ENTSOE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
