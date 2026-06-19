import asyncio
import datetime

from core.config import AppSettings
from database.manager import DatabaseManager
from forecasting.load.arimax_prediction_service import (
    LoadForecastingARIMAXPredictionService,
)
from forecasting.load.data_processor import LoadForecastingDataProcessor
from forecasting.load.repository import LoadForecastingRepository


async def main():
    app_settings = AppSettings()
    timescale_database = DatabaseManager(app_settings.db)
    timescale_database.initialize()

    # Define strict temporal boundaries
    start_train = datetime.datetime(2023, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    end_train = datetime.datetime(2026, 3, 31, 23, 45, tzinfo=datetime.timezone.utc)

    FIXED_FOURIER_K = (2, 1, 1, 1)

    async for session in timescale_database.get_session():
        repository = LoadForecastingRepository(session)
        processor = LoadForecastingDataProcessor(frequency_minutes=15)

        service = LoadForecastingARIMAXPredictionService(
            repository=repository,
            processor=processor,
            start_date_train=start_train,
            end_date_train=end_train,
            fixed_k_params=FIXED_FOURIER_K,
            output_dir="pipeline_output",
            with_fit=True,
            with_scalar=True,
            with_dropped_features=True,
        )

        metrics = await service.fit_and_save(
            "arimax_scaled_dropped_features_specific.pkl"
        )

        print(metrics)


if __name__ == "__main__":
    asyncio.run(main())
