import ssl
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import TimescaleDatabaseSettings


class DatabaseManager:
    def __init__(self, settings: TimescaleDatabaseSettings):
        self.settings = settings
        self._engine: AsyncEngine | None = None
        self._session_maker: async_sessionmaker[AsyncSession] | None = None

    def _get_ssl_context(self) -> ssl.SSLContext | None:
        """Builds the SSL context based on the Pydantic settings."""
        if not self.settings.DB_USE_SSL:
            return None

        ssl_context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH, cafile=self.settings.DB_SSL_CA_CERT_PATH
        )

        if (
            self.settings.DB_SSL_CLIENT_CERT_PATH
            and self.settings.DB_SSL_CLIENT_KEY_PATH
        ):
            ssl_context.load_cert_chain(
                certfile=self.settings.DB_SSL_CLIENT_CERT_PATH,
                keyfile=self.settings.DB_SSL_CLIENT_KEY_PATH,
            )

        ssl_context.set_alpn_protocols(["postgresql"])
        return ssl_context

    def initialize(self) -> None:
        """Initializes the engine and session factory."""
        connect_args = {}
        if self.settings.DB_USE_SSL:
            connect_args["ssl"] = self._get_ssl_context()

        self._engine = create_async_engine(
            self.settings.async_database_url,
            echo=False,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

        self._session_maker = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    async def close(self) -> None:
        """Safely disposes of the engine pool. Call this on app shutdown."""
        if self._engine is None:
            return
        await self._engine.dispose()

    @property
    def engine(self) -> AsyncEngine:
        """Exposes the engine for Alembic and testing."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._engine

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Dependency to be injected into routes/services."""
        if self._session_maker is None:
            raise RuntimeError(
                "Database connection is not initialized. Call initialize() first."
            )

        async with self._session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
