"""Open event schema for joule-agent telemetry."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InferencePhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class BaseEvent(BaseModel):
    """Common fields for all exported events."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    timestamp: datetime = Field(default_factory=utcnow)
    cluster_id: str | None = None
    tenant_id: str | None = None
    node_id: str | None = None
    gpu_index: int | None = None

    def to_export(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class InferenceRequestEvent(BaseEvent):
    """Per-request inference telemetry."""

    event_type: Literal["inference.request"] = "inference.request"
    request_id: str
    model_id: str
    phase: InferencePhase
    tokens_in: int = 0
    tokens_out: int = 0
    batch_size: int = 1
    latency_ms: float | None = None
    power_watts: float | None = None
    clock_mhz: int | None = None
    engine: str | None = None


class GPUSampleEvent(BaseEvent):
    """Periodic GPU hardware sample."""

    event_type: Literal["gpu.sample"] = "gpu.sample"
    power_watts: float
    clock_sm_mhz: int | None = None
    clock_mem_mhz: int | None = None
    utilization_pct: float | None = None
    memory_utilization_pct: float | None = None
    temperature_c: float | None = None
    gpu_name: str | None = None


class EventBatch(BaseModel):
    """Batch payload sent to the ingest endpoint."""

    schema_version: str = "1.0"
    agent_version: str
    events: list[dict[str, Any]]

    def to_export(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
