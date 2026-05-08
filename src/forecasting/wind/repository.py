from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.types import CountryCode, EnergySource
from database.models import EnergyGeneration


class WindForecastingRepository:
    """
    Repository dedicated to reading and aggregating data specifically for Wind ML Forecasting.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_historical_wind_generation(
        self,
        country_code: CountryCode,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[EnergyGeneration]:
        """
        Retrieves raw historical onshore wind generation data.
        """
        stmt = (
            select(EnergyGeneration)
            .where(
                EnergyGeneration.country_code == country_code,
                EnergyGeneration.energy_source == EnergySource.WIND_ONSHORE,
                EnergyGeneration.timestamp >= start_date,
                EnergyGeneration.timestamp <= end_date,
            )
            .order_by(EnergyGeneration.timestamp.asc())
        )

        result = await self._session.execute(stmt)
        return result.scalars().all()
