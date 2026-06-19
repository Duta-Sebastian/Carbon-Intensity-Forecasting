from datetime import datetime

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from core.types import CountryCode, MetricType
from data_pipeline.entsoe.adapter import EntsoeAdapter
from data_pipeline.entsoe.manager import EntsoeManager
from data_pipeline.entsoe.repository import EntsoeRepository
from database.schemas.entsoe import EnergyGenerationSchema, EnergyLoadSchema


class EntsoeService:
    def __init__(self, manager: EntsoeManager, session: AsyncSession) -> None:
        self.manager: EntsoeManager = manager
        self.adapter: EntsoeAdapter = EntsoeAdapter()
        self.repository: EntsoeRepository = EntsoeRepository(session)

    async def sync_generation_data(
        self, country_code: CountryCode, start: datetime, end: datetime
    ) -> list[EnergyGenerationSchema] | None:
        """
        Orchestrates fetching, transforming, and persisting ENTSO-E generation data.
        """
        ts_start: pd.Timestamp = pd.Timestamp(start)
        ts_end: pd.Timestamp = pd.Timestamp(end)

        df: pd.DataFrame = await self.manager.query_generation(
            country_code=country_code, start=ts_start, end=ts_end
        )

        if df.empty:
            return None

        schemas: list[EnergyGenerationSchema] = self.adapter.transform(
            df=df, country_code=country_code, metric_type=MetricType.GENERATION
        )

        await self.repository.insert_generation(schemas)

        return schemas

    async def sync_load_data(
        self, country_code: CountryCode, start: datetime, end: datetime
    ) -> list[EnergyLoadSchema] | None:
        """
        Orchestrates fetching, transforming, and persisting ENTSO-E generation data.
        """
        ts_start: pd.Timestamp = pd.Timestamp(start)
        ts_end: pd.Timestamp = pd.Timestamp(end)

        df: pd.DataFrame = await self.manager.query_load(
            country_code=country_code, start=ts_start, end=ts_end
        )

        if df.empty:
            return None

        schemas: list[EnergyLoadSchema] = self.adapter.transform(
            df=df, country_code=country_code, metric_type=MetricType.LOAD
        )

        await self.repository.insert_load(schemas)

        return schemas
