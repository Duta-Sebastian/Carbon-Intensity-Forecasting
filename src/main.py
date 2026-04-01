import asyncio
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from adapters.entsoe_adapter import EntsoeAdapter
from clients.entsoe_client import EntsoeManager
from core.config.entsoe_settings import EntsoeSettings
from core.types import CountryCode, MetricType
from gateways.entsoe_gateway import EntsoeGateway


async def main():
    now = datetime.now(timezone.utc)
    start_dt = (now - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_dt = start_dt + timedelta(hours=23, minutes=59)
    yesterday_start = pd.Timestamp(start_dt)
    yesterday_end = pd.Timestamp(end_dt)

    entsoe_settings = EntsoeSettings.model_validate({})
    entsoe_gw = EntsoeGateway().initialize(entsoe_settings)
    entsoe_manager = EntsoeManager(entsoe_gw)
    entsoe_adapter = EntsoeAdapter()
    df = await entsoe_manager.query_generation(
        CountryCode.ROMANIA, yesterday_start, yesterday_end
    )
    print(df)
    results = entsoe_adapter.transform(df, CountryCode.ROMANIA, MetricType.GENERATION)
    with open("test.out", "w") as f:
        json.dump([result.model_dump_json() for result in results], f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
