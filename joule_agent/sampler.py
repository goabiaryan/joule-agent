"""GPU hardware sampler using NVML (pynvml)."""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Iterator

from joule_agent.exporter import Exporter
from joule_agent.schema import GPUSampleEvent

logger = logging.getLogger(__name__)


def _load_nvml():
    try:
        import pynvml
    except ImportError as exc:
        raise ImportError(
            "GPU sampling requires nvidia-ml-py. Install with: pip install 'joule-agent[gpu]'"
        ) from exc
    return pynvml


class GPUSampler:
    """Polls NVML and emits gpu.sample events."""

    def __init__(
        self,
        exporter: Exporter,
        *,
        interval: float = 2.0,
        gpu_indices: list[int] | None = None,
        node_id: str | None = None,
    ) -> None:
        self.exporter = exporter
        self.interval = interval
        self.gpu_indices = gpu_indices
        self.node_id = node_id or socket.gethostname()
        self._pynvml = None
        self._handles: dict[int, object] = {}

    def _init_nvml(self) -> None:
        if self._pynvml is not None:
            return

        pynvml = _load_nvml()
        pynvml.nvmlInit()
        self._pynvml = pynvml

        count = pynvml.nvmlDeviceGetCount()
        indices = self.gpu_indices if self.gpu_indices is not None else list(range(count))
        for index in indices:
            self._handles[index] = pynvml.nvmlDeviceGetHandleByIndex(index)

    def sample_once(self) -> list[GPUSampleEvent]:
        self._init_nvml()
        assert self._pynvml is not None
        pynvml = self._pynvml

        events: list[GPUSampleEvent] = []
        for index, handle in self._handles.items():
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except pynvml.NVMLError:
                sm_clock = mem_clock = None

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_pct = float(util.gpu)
                mem_util_pct = float(util.memory)
            except pynvml.NVMLError:
                util_pct = mem_util_pct = None

            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError:
                temp = None

            try:
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode()
            except pynvml.NVMLError:
                gpu_name = None

            events.append(
                GPUSampleEvent(
                    cluster_id=self.exporter.cluster_id,
                    tenant_id=self.exporter.tenant_id,
                    node_id=self.node_id,
                    gpu_index=index,
                    power_watts=power_mw / 1000.0,
                    clock_sm_mhz=sm_clock,
                    clock_mem_mhz=mem_clock,
                    utilization_pct=util_pct,
                    memory_utilization_pct=mem_util_pct,
                    temperature_c=temp,
                    gpu_name=gpu_name,
                )
            )
        return events

    def run(self) -> None:
        self.exporter.start()
        logger.info("Starting GPU sampler (interval=%ss, node=%s)", self.interval, self.node_id)
        try:
            while True:
                for event in self.sample_once():
                    self.exporter.emit(event)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("Sampler interrupted, flushing remaining events")
        finally:
            self.exporter.stop()
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()

    def iter_samples(self) -> Iterator[GPUSampleEvent]:
        """Yield samples once per call — useful in tests or custom loops."""
        for event in self.sample_once():
            yield event


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exporter = Exporter.from_env()
    interval = float(os.environ.get("JOULE_SAMPLER_INTERVAL", "2"))
    sampler = GPUSampler(exporter, interval=interval)
    sampler.run()


if __name__ == "__main__":
    main()
