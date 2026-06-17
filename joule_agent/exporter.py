"""Batch event exporter for joule-agent."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

import httpx

from joule_agent import __version__
from joule_agent.schema import BaseEvent, EventBatch

logger = logging.getLogger(__name__)


class Exporter:
    """Buffers events and flushes them to a Joule ingest endpoint."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        cluster_id: str,
        *,
        tenant_id: str | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        timeout: float = 10.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.cluster_id = cluster_id
        self.tenant_id = tenant_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @classmethod
    def from_env(cls) -> Exporter:
        endpoint = os.environ.get("JOULE_ENDPOINT")
        api_key = os.environ.get("JOULE_API_KEY")
        cluster_id = os.environ.get("JOULE_CLUSTER_ID")

        missing = [
            name
            for name, value in [
                ("JOULE_ENDPOINT", endpoint),
                ("JOULE_API_KEY", api_key),
                ("JOULE_CLUSTER_ID", cluster_id),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            endpoint=endpoint,  # type: ignore[arg-type]
            api_key=api_key,  # type: ignore[arg-type]
            cluster_id=cluster_id,  # type: ignore[arg-type]
            tenant_id=os.environ.get("JOULE_TENANT_ID"),
            batch_size=int(os.environ.get("JOULE_BATCH_SIZE", "100")),
            flush_interval=float(os.environ.get("JOULE_FLUSH_INTERVAL", "5")),
        )

    def start(self) -> None:
        if self._flush_thread and self._flush_thread.is_alive():
            return
        self._stop_event.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=self.flush_interval + 1)
        self.flush()

    def emit(self, event: BaseEvent) -> None:
        payload = event.to_export()
        if payload.get("cluster_id") is None:
            payload["cluster_id"] = self.cluster_id
        if payload.get("tenant_id") is None:
            payload["tenant_id"] = self.tenant_id

        with self._lock:
            self._buffer.append(payload)
            should_flush = len(self._buffer) >= self.batch_size

        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = EventBatch(
                agent_version=__version__,
                events=self._buffer.copy(),
            )
            self._buffer.clear()

        try:
            asyncio.run(self._send_batch(batch))
        except Exception:
            logger.exception("Failed to export joule event batch")

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(self.flush_interval):
            self.flush()

    async def _send_batch(self, batch: EventBatch) -> None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint,
                json=batch.to_export(),
                headers=headers,
            )
            response.raise_for_status()
        logger.debug("Exported %d events to %s", len(batch.events), self.endpoint)
