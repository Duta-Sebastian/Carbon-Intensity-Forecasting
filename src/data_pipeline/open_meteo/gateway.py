import logging
import threading

import openmeteo_requests

logger = logging.getLogger(__name__)


class OpenMeteoGateway:
    """
    Infrastructure Gateway for Open-Meteo.
    Responsibility: Thread-safe lifecycle management of the synchronous SDK.
    """

    def __init__(self) -> None:
        self._client: openmeteo_requests.Client | None = None
        self._lock = threading.Lock()

    def initialize(self) -> "OpenMeteoGateway":
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = openmeteo_requests.Client()
                    logger.info("Open-Meteo Gateway successfully initialized.")
        return self

    @property
    def client(self) -> openmeteo_requests.Client:
        if self._client is None:
            raise RuntimeError("Gateway not initialized. Call initialize() first.")
        return self._client

    @property
    def lock(self) -> threading.Lock:
        return self._lock
