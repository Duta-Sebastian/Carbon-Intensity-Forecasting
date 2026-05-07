import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection

from alembic import context
from src.database.config import TimescaleDatabaseSettings
from src.database.manager import DatabaseManager
from src.database.models import Base, EnergyGeneration  # noqa: F401

settings = TimescaleDatabaseSettings()
db_manager = DatabaseManager(settings=settings)

target_metadata = Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Used for generating SQL scripts without a live DB connection.
    """
    url = settings.sync_database_url

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Synchronous core logic.
    Alembic's internal engine works synchronously even in async projects.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Online mode: Connects to the DB using your AsyncEngine.
    """
    db_manager.initialize()
    connectable = db_manager.engine

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await db_manager.close()


def run_migrations_online() -> None:
    """Entry point for online migrations."""
    asyncio.run(run_async_migrations())


# Execution logic
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
