import logging
import threading

from entsoe.entsoe import EntsoePandasClient

from data_pipeline.entsoe.config import EntsoeSettings

logger = logging.getLogger(__name__)


class EntsoeGateway:
    """
    Infrastructure Gateway for ENTSO-E.
    Responsibility: Lifecycle management of the synchronous SDK.
    """

    def __init__(self) -> None:
        self._client: EntsoePandasClient | None = None
        self._lock = threading.Lock()

    def initialize(self, settings: EntsoeSettings) -> "EntsoeGateway":
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = EntsoePandasClient(
                        api_key=settings.api_key.get_secret_value()
                    )
                    logger.info("ENTSO-E Gateway successfully initialized.")
        return self

    @property
    def client(self) -> EntsoePandasClient:
        if self._client is None:
            raise RuntimeError("Gateway not initialized. Call initialize() first.")
        return self._client

    @property
    def lock(self) -> threading.Lock:
        return self._lock
