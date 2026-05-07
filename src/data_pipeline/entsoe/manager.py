import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

if TYPE_CHECKING:
    from data_pipeline.entsoe.gateway import EntsoeGateway

logger = logging.getLogger(__name__)


class EntsoeManager:
    """
    Async Proxy Manager for ENTSO-E.
    Responsibility: Bridging Sync-to-Async using ThreadPoolExecutors
    and ensuring linearizable access through the Gateway's lock.
    """

    def __init__(self, gateway: EntsoeGateway) -> None:
        self._gateway = gateway

    async def _execute[T](self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Offloads the blocking sync call to a worker thread while
        enforcing the thread-lock from the gateway.
        """
        loop = asyncio.get_running_loop()

        def locked_call():
            with self._gateway.lock:
                return func(*args, **kwargs)

        return await loop.run_in_executor(None, locked_call)

    def __getattr__(self, name: str) -> Any:
        """
        Magic method that intercepts calls to methods not defined in this class.
        """
        attr = getattr(self._gateway.client, name)

        if callable(attr):

            async def wrapper(*args, **kwargs):
                return await self._execute(attr, *args, **kwargs)

            return wrapper

        return attr

    async def query_load(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Proxy for query_load."""
        return await self._execute(
            self._gateway.client.query_load, country_code, start=start, end=end
        )

    async def query_generation(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Proxy for query_generation."""
        return await self._execute(
            self._gateway.client.query_generation,
            country_code,
            start=start,
            end=end,
        )
