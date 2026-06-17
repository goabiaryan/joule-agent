"""vLLM integration hooks for joule-agent."""

from __future__ import annotations

import logging
import socket
import time
from typing import Any
from uuid import uuid4

from joule_agent.exporter import Exporter
from joule_agent.schema import InferencePhase, InferenceRequestEvent

logger = logging.getLogger(__name__)


class JouleVLLMCallback:
    """
    Lightweight callback object for vLLM request lifecycle hooks.

    Wire this into your vLLM deployment by calling the phase methods from
    scheduler / engine callbacks. The exact hook points depend on your
    vLLM version — see examples/vllm_sidecar/ for a reference layout.

    Usage:
        exporter = Exporter.from_env()
        callback = JouleVLLMCallback(exporter=exporter, model_id="llama-3.1-70b")
        exporter.start()

        callback.on_prefill_start(request_id="abc", tokens_in=512, batch_size=8)
        callback.on_prefill_end(request_id="abc", latency_ms=42.5)
        callback.on_decode_start(request_id="abc")
        callback.on_decode_end(request_id="abc", tokens_out=128, latency_ms=2100.0)
    """

    def __init__(
        self,
        exporter: Exporter,
        model_id: str,
        *,
        gpu_index: int | None = None,
        node_id: str | None = None,
        attach_power: bool = True,
    ) -> None:
        self.exporter = exporter
        self.model_id = model_id
        self.gpu_index = gpu_index
        self.node_id = node_id or socket.gethostname()
        self.attach_power = attach_power
        self._active: dict[str, float] = {}

    def _maybe_power(self) -> tuple[float | None, int | None]:
        if not self.attach_power:
            return None, None
        try:
            from joule_agent.sampler import GPUSampler

            sampler = GPUSampler(self.exporter, gpu_indices=[self.gpu_index or 0])
            sample = sampler.sample_once()[0]
            return sample.power_watts, sample.clock_sm_mhz
        except Exception:
            logger.debug("Power attachment unavailable", exc_info=True)
            return None, None

    def _emit(
        self,
        *,
        request_id: str,
        phase: InferencePhase,
        tokens_in: int = 0,
        tokens_out: int = 0,
        batch_size: int = 1,
        latency_ms: float | None = None,
    ) -> None:
        power_watts, clock_mhz = self._maybe_power() if latency_ms is not None else (None, None)
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
                power_watts=power_watts,
                clock_mhz=clock_mhz,
                engine="vllm",
            )
        )

    def on_request_start(self, *, tokens_in: int, batch_size: int = 1, request_id: str | None = None) -> str:
        request_id = request_id or str(uuid4())
        self._active[request_id] = time.perf_counter()
        self._emit(
            request_id=request_id,
            phase=InferencePhase.PREFILL,
            tokens_in=tokens_in,
            batch_size=batch_size,
        )
        return request_id

    def on_prefill_start(self, *, request_id: str, tokens_in: int, batch_size: int = 1) -> None:
        self._active[request_id] = time.perf_counter()
        self._emit(
            request_id=request_id,
            phase=InferencePhase.PREFILL,
            tokens_in=tokens_in,
            batch_size=batch_size,
        )

    def on_prefill_end(self, *, request_id: str, tokens_in: int, batch_size: int = 1, latency_ms: float | None = None) -> None:
        latency_ms = latency_ms or self._elapsed_ms(request_id)
        self._emit(
            request_id=request_id,
            phase=InferencePhase.PREFILL,
            tokens_in=tokens_in,
            batch_size=batch_size,
            latency_ms=latency_ms,
        )

    def on_decode_start(self, *, request_id: str, batch_size: int = 1) -> None:
        self._active[request_id] = time.perf_counter()
        self._emit(request_id=request_id, phase=InferencePhase.DECODE, batch_size=batch_size)

    def on_decode_end(
        self,
        *,
        request_id: str,
        tokens_out: int,
        batch_size: int = 1,
        latency_ms: float | None = None,
    ) -> None:
        latency_ms = latency_ms or self._elapsed_ms(request_id)
        self._emit(
            request_id=request_id,
            phase=InferencePhase.DECODE,
            tokens_out=tokens_out,
            batch_size=batch_size,
            latency_ms=latency_ms,
        )
        self._active.pop(request_id, None)

    def _elapsed_ms(self, request_id: str) -> float | None:
        started = self._active.get(request_id)
        if started is None:
            return None
        return (time.perf_counter() - started) * 1000.0


def wrap_vllm_engine(engine: Any, callback: JouleVLLMCallback) -> Any:
    """
    Best-effort monkey-patch for simple vLLM deployments.

    This wraps common engine methods when present. For production, prefer
    explicit hooks in your serving layer.
    """
    if hasattr(engine, "generate"):
        original = engine.generate

        def generate(*args, **kwargs):
            request_id = callback.on_request_start(tokens_in=_guess_tokens_in(args, kwargs), batch_size=1)
            try:
                result = original(*args, **kwargs)
                callback.on_decode_end(request_id=request_id, tokens_out=_guess_tokens_out(result))
                return result
            except Exception:
                raise

        engine.generate = generate  # type: ignore[method-assign]
        logger.info("joule-agent: wrapped engine.generate")

    return engine


def _guess_tokens_in(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    prompt = kwargs.get("prompt") or (args[0] if args else "")
    if isinstance(prompt, str):
        return max(1, len(prompt.split()))
    if isinstance(prompt, list):
        return max(1, len(prompt))
    return 1


def _guess_tokens_out(result: Any) -> int:
    if result is None:
        return 0
    text = getattr(result, "text", None) or str(result)
    return max(1, len(text.split()))
