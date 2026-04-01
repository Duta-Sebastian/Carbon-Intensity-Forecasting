from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TimescaleDatabaseSettings(BaseSettings):
    """Isolated settings exclusively for Database configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="TIMESCALE_DB_",
    )

    USER: str = Field(default="sebastian", description="Database username")
    PASSWORD: str = Field(default="admin", description="Database password")
    HOST: str = Field(default="localhost", description="Database host")
    PORT: int = Field(default=5432, description="Database port")
    DB: str = Field(default="carbon_intensity_forecasting", description="Database name")

    DB_USE_SSL: bool = Field(
        default=False, description="Whether to use SSL for the database connection"
    )

    DB_SSL_CA_CERT_PATH: str | None = Field(
        default=None, description="Path to the SSL CA certificate"
    )

    DB_SSL_CLIENT_CERT_PATH: str | None = Field(
        default=None, description="Path to the SSL client certificate"
    )
    DB_SSL_CLIENT_KEY_PATH: str | None = Field(
        default=None, description="Path to the SSL client key"
    )

    @computed_field
    @property
    def async_database_url(self) -> str:
        """Constructs the asyncpg connection string."""
        return (
            f"postgresql+asyncpg://{self.USER}:{self.PASSWORD}"
            f"@{self.HOST}:{self.PORT}/{self.DB}"
        )

    @computed_field
    @property
    def sync_database_url(self) -> str:
        """
        Constructs a synchronous-compatible URL for Alembic.
        Uses the standard postgresql:// scheme without the +asyncpg driver.
        """
        return (
            f"postgresql://{self.USER}:{self.PASSWORD}"
            f"@{self.HOST}:{self.PORT}/{self.DB}"
        )
