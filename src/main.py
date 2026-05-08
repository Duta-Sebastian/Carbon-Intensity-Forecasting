import asyncio
from datetime import datetime, timedelta, timezone

from core.config import AppSettings
from core.types import CountryCode
from data_pipeline.entsoe.gateway import EntsoeGateway
from data_pipeline.entsoe.manager import EntsoeManager
from database.manager import DatabaseManager
from services.entsoe_service import EntsoeService


async def main():
    app_settings = AppSettings()
    entsoe_gw = EntsoeGateway().initialize(app_settings.entsoe)
    entsoe_manager = EntsoeManager(entsoe_gw)

    timescale_database = DatabaseManager(app_settings.db)
    timescale_database.initialize()

    current_date = datetime.now(timezone.utc)
    start_date = current_date - timedelta(days=730)
    end_date = current_date
    print(start_date)

    async for session in timescale_database.get_session():
        entsoe_service = EntsoeService(entsoe_manager, session)
        await entsoe_service.sync_load_data(
            country_code=CountryCode.ROMANIA,
            start=start_date,
            end=end_date,
        )

    # app_settings = AppSettings()
    # db_manager = DatabaseManager(app_settings.db)
    # db_manager.initialize()

    # timescale_database = DatabaseManager(app_settings.db)
    # timescale_database.initialize()

    # end_date = datetime.now(timezone.utc) - timedelta(days=1)
    # start_date = end_date - timedelta(days=365)

    # tracker = MLTrackingService()

    # async for session in timescale_database.get_session():
    #     repository = WindForecastingRepository(session)
    #     service = WindForecastingService(repository, tracker)

    #     my_model = WindSARIMAModel(order=(2, 1, 2), seasonal_order=(0, 0, 0, 96))
    #     await service.run_train_test_pipeline(
    #         my_model, CountryCode.ROMANIA, start_date, end_date
    #     )


if __name__ == "__main__":
    asyncio.run(main())
