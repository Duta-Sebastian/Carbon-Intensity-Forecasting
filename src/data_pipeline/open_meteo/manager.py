import asyncio
import logging
from typing import Any, Callable

from data_pipeline.open_meteo.gateway import OpenMeteoGateway

logger = logging.getLogger(__name__)


class OpenMeteoManager:
    """
    Async Proxy Manager for Open-Meteo.
    Responsibility: Bridging Sync-to-Async using ThreadPoolExecutors
    and ensuring linearizable access through the Gateway's lock.
    """

    def __init__(self, gateway: OpenMeteoGateway) -> None:
        self._gateway = gateway

    async def _execute[T](self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()

        def locked_call():
            with self._gateway.lock:
                return func(*args, **kwargs)

        return await loop.run_in_executor(None, locked_call)

    async def fetch_weather_data(self, url: str, params: dict[str, Any]) -> list[Any]:
        """Proxy for openmeteo client.weather_api"""
        return await self._execute(
            self._gateway.client.weather_api, url, params=params, timeout=90
        )
