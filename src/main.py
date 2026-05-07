import asyncio
from datetime import datetime, timedelta, timezone

from core.config import AppSettings
from data_pipeline.entsoe import (
    EntsoeGateway,
    EntsoeManager,
)
from database.manager import DatabaseManager


async def main():
    app_settings = AppSettings()
    entsoe_gw = EntsoeGateway().initialize(app_settings.entsoe)
    entsoe_manager = EntsoeManager(entsoe_gw)

    timescale_database = DatabaseManager(app_settings.db)
    timescale_database.initialize()

    start_date = datetime.now(timezone.utc) - timedelta(365)
    # print(start_date)
    # for _ in range(0, 1095):
    #     start_date = (start_date - timedelta(days=1)).replace(
    #         hour=0, minute=0, second=0, microsecond=0
    #     )
    #     end_date = start_date + timedelta(
    #         hours=23, minutes=59, seconds=59, microseconds=999, milliseconds=999
    #     )
    #     async for session in timescale_database.get_session():
    #         entsoe_service = EntsoeService(entsoe_manager, session)
    #         await entsoe_service.sync_generation_data(
    #             country_code=CountryCode.ROMANIA,
    #             start=start_date,
    #             end=end_date,
    #         )


if __name__ == "__main__":
    asyncio.run(main())
