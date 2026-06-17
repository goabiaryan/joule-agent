"""NVIDIA Triton integration hooks for joule-agent."""

from __future__ import annotations

import logging
import socket
import time
from typing import Any
from uuid import uuid4

from joule_agent.exporter import Exporter
from joule_agent.schema import InferencePhase, InferenceRequestEvent

logger = logging.getLogger(__name__)


class JouleTritonCallback:
    """
    Callback helper for Triton Python backend models.

    Call `on_execute` from your Triton Python model's `execute` method.
    """

    def __init__(
        self,
        exporter: Exporter,
        model_id: str,
        *,
        gpu_index: int | None = None,
        node_id: str | None = None,
    ) -> None:
        self.exporter = exporter
        self.model_id = model_id
        self.gpu_index = gpu_index
        self.node_id = node_id or socket.gethostname()

    def on_execute(
        self,
        *,
        request_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        batch_size: int = 1,
        latency_ms: float | None = None,
        phase: InferencePhase = InferencePhase.DECODE,
    ) -> str:
        request_id = request_id or str(uuid4())
        self.exporter.emit(
            InferenceRequestEvent(
                cluster_id=self.exporter.cluster_id,
                tenant_id=self.exporter.tenant_id,
                node_id=self.node_id,
                gpu_index=self.gpu_index,
                request_id=request_id,
                model_id=self.model_id,
                phase=phase,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                batch_size=batch_size,
                latency_ms=latency_ms,
                engine="triton",
            )
        )
        return request_id

    def timed_execute(self, fn: Any, *, tokens_in: int = 0, batch_size: int = 1) -> Any:
        """Wrap a Triton execute callable and emit latency automatically."""
        request_id = str(uuid4())
        started = time.perf_counter()
        self.on_execute(
            request_id=request_id,
            phase=InferencePhase.PREFILL,
            tokens_in=tokens_in,
            batch_size=batch_size,
        )
        result = fn()
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.on_execute(
            request_id=request_id,
            phase=InferencePhase.DECODE,
            tokens_out=max(1, tokens_in),
            batch_size=batch_size,
            latency_ms=latency_ms,
        )
        return result
