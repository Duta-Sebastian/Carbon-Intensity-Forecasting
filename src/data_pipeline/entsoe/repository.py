from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import EnergyGeneration
from database.schemas.entsoe import EnergyGenerationSchema


class EntsoeRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def insert_generation(self, data: list[EnergyGenerationSchema]):
        if not data:
            return

        values = [item.model_dump(mode="python") for item in data]

        stmt = insert(EnergyGeneration).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                "timestamp",
                "country_code",
                "provider",
                "energy_source",
            ],
            set_={
                "generation_mw": stmt.excluded.generation_mw,
                "updated_at": func.now(),
            },
        )

        await self._session.execute(stmt)
        await self._session.commit()
