"""SGLang integration hooks for joule-agent."""

from __future__ import annotations

import logging
import socket
import time
from uuid import uuid4

from joule_agent.exporter import Exporter
from joule_agent.schema import InferencePhase, InferenceRequestEvent

logger = logging.getLogger(__name__)


class JouleSGLangCallback:
    """
    Callback hooks for SGLang serving runtimes.

    SGLang deployments vary — this class exposes explicit phase methods
    you can call from middleware or custom endpoints. See
    examples/sglang_plugin/ for a FastAPI-style wiring example.
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
        self._active: dict[str, float] = {}

    def track_request(
        self,
        *,
        phase: InferencePhase,
        tokens_in: int = 0,
        tokens_out: int = 0,
        batch_size: int = 1,
        latency_ms: float | None = None,
        request_id: str | None = None,
    ) -> str:
        request_id = request_id or str(uuid4())
        if latency_ms is None and request_id in self._active:
            latency_ms = (time.perf_counter() - self._active[request_id]) * 1000.0

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
                engine="sglang",
            )
        )
        return request_id

    def begin(self, *, tokens_in: int, batch_size: int = 1, request_id: str | None = None) -> str:
        request_id = request_id or str(uuid4())
        self._active[request_id] = time.perf_counter()
        return self.track_request(
            phase=InferencePhase.PREFILL,
            tokens_in=tokens_in,
            batch_size=batch_size,
            request_id=request_id,
        )

    def complete(self, *, request_id: str, tokens_out: int, batch_size: int = 1) -> None:
        self.track_request(
            phase=InferencePhase.DECODE,
            tokens_out=tokens_out,
            batch_size=batch_size,
            request_id=request_id,
        )
        self._active.pop(request_id, None)
