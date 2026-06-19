from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.types import (
    CountryCode,
    EnergyDataProvider,
    RomanianCity,
    WeatherDataProvider,
)
from database.models import EnergyLoad
from database.models.WeatherData import WeatherData


class LoadForecastingRepository:
    """
    Repository dedicated to reading and aggregating data specifically for Load ML Forecasting.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_historical_load_generation(
        self,
        country_code: CountryCode = CountryCode.ROMANIA,
        provider: EnergyDataProvider = EnergyDataProvider.ENTSOE,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[EnergyLoad]:
        """
        Retrieves raw historical onshore wind generation data.
        """

        stmt = select(EnergyLoad).where(
            EnergyLoad.country_code == country_code and EnergyLoad.provider == provider
        )

        if start_date is not None:
            stmt = stmt.where(EnergyLoad.timestamp >= start_date)

        if end_date is not None:
            stmt = stmt.where(EnergyLoad.timestamp <= end_date)

        stmt = stmt.order_by(EnergyLoad.timestamp.asc())

        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def find_irregular_intervals(self) -> list[dict[str, Any]] | None:
        """
        Scans the entire energy_load table and returns any records
        where the gap between it and the previous record is NOT 15 minutes.
        """
        query = text("""
            WITH time_diffs AS (
                SELECT 
                    timestamp,
                    country_code,
                    provider,
                    load_mw,
                    timestamp - LAG(timestamp) OVER (
                        PARTITION BY country_code, provider 
                        ORDER BY timestamp
                    ) AS time_delta
                FROM energy_load
            )
            SELECT 
                timestamp, 
                country_code, 
                provider, 
                time_delta
            FROM time_diffs
            WHERE time_delta != INTERVAL '15 minutes'
              AND time_delta IS NOT NULL;
        """)

        result = await self._session.execute(query)

        bad_rows = result.mappings().all()

        if bad_rows:
            print(f"FOUND {len(bad_rows)} IRREGULAR GAPS IN THE DATABASE!")
            for row in bad_rows:
                print(
                    f"Jumped {row['time_delta']} at {row['timestamp']} ({row['provider']} - {row['country_code']})"
                )
            return [dict(row) for row in bad_rows]

        return None

    async def get_historical_load_aggregated_hourly(
        self,
        country_code: CountryCode = CountryCode.ROMANIA,
        provider: EnergyDataProvider = EnergyDataProvider.ENTSOE,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[EnergyLoad]:
        hour_expr = func.date_trunc(text("'hour'"), EnergyLoad.timestamp)

        stmt = select(
            hour_expr.label("timestamp"),
            EnergyLoad.country_code,
            EnergyLoad.provider,
            func.avg(EnergyLoad.load_mw).label("load_mw"),
        ).where(
            EnergyLoad.country_code == country_code, EnergyLoad.provider == provider
        )

        if start_date is not None:
            stmt = stmt.where(EnergyLoad.timestamp >= start_date)

        if end_date is not None:
            stmt = stmt.where(EnergyLoad.timestamp <= end_date)

        stmt = stmt.group_by(
            hour_expr, EnergyLoad.country_code, EnergyLoad.provider
        ).order_by(hour_expr.asc())

        result = await self._session.execute(stmt)

        aggregated_loads = []
        for row in result.mappings().all():
            aggregated_loads.append(
                EnergyLoad(
                    timestamp=row["timestamp"],
                    country_code=row["country_code"],
                    provider=row["provider"],
                    load_mw=row["load_mw"],
                )
            )

        return aggregated_loads

    async def get_historical_weather_data(
        self,
        cities: list[RomanianCity] | None = None,
        provider: WeatherDataProvider = WeatherDataProvider.OPENMETEO,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[WeatherData]:
        """
        Retrieves raw historical weather data, optionally filtered by specific cities and dates.
        """
        stmt = select(WeatherData).where(WeatherData.provider == provider)

        if cities:
            stmt = stmt.where(WeatherData.city.in_(cities))

        if start_date is not None:
            stmt = stmt.where(WeatherData.timestamp >= start_date)

        if end_date is not None:
            stmt = stmt.where(WeatherData.timestamp <= end_date)
        stmt = stmt.order_by(WeatherData.timestamp.asc(), WeatherData.city.asc())

        result = await self._session.execute(stmt)
        return result.scalars().all()
