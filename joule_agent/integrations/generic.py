"""Generic log-line parser fallback for unsupported inference engines."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator, TextIO

from joule_agent.exporter import Exporter
from joule_agent.schema import InferencePhase, InferenceRequestEvent

logger = logging.getLogger(__name__)

# Example log line:
# joule inference phase=decode model=llama-70b request_id=abc latency_ms=2100 tokens_out=128 batch=8
LOG_PATTERN = re.compile(
    r"phase=(?P<phase>prefill|decode)\s+"
    r"model=(?P<model>\S+)\s+"
    r"request_id=(?P<request_id>\S+)"
    r"(?:\s+latency_ms=(?P<latency_ms>\d+(?:\.\d+)?))?"
    r"(?:\s+tokens_in=(?P<tokens_in>\d+))?"
    r"(?:\s+tokens_out=(?P<tokens_out>\d+))?"
    r"(?:\s+batch=(?P<batch>\d+))?"
)


class GenericLogParser:
    """Parse structured log lines and emit inference.request events."""

    def __init__(self, exporter: Exporter, *, engine: str = "generic") -> None:
        self.exporter = exporter
        self.engine = engine

    def parse_line(self, line: str) -> InferenceRequestEvent | None:
        if "joule inference" not in line:
            return None

        match = LOG_PATTERN.search(line)
        if not match:
            logger.debug("Unrecognized joule log line: %s", line.strip())
            return None

        groups = match.groupdict()
        return InferenceRequestEvent(
            cluster_id=self.exporter.cluster_id,
            tenant_id=self.exporter.tenant_id,
            request_id=groups["request_id"],
            model_id=groups["model"],
            phase=InferencePhase(groups["phase"]),
            latency_ms=float(groups["latency_ms"]) if groups.get("latency_ms") else None,
            tokens_in=int(groups["tokens_in"] or 0),
            tokens_out=int(groups["tokens_out"] or 0),
            batch_size=int(groups["batch"] or 1),
            engine=self.engine,
        )

    def emit_line(self, line: str) -> bool:
        event = self.parse_line(line)
        if event is None:
            return False
        self.exporter.emit(event)
        return True

    def tail_file(self, path: str | Path) -> Iterator[bool]:
        """Tail a log file and emit parsed events."""
        log_path = Path(path)
        with log_path.open() as handle:
            handle.seek(0, 2)
            while True:
                line = handle.readline()
                if not line:
                    continue
                yield self.emit_line(line)

    def ingest_jsonl(self, stream: TextIO) -> int:
        """Ingest newline-delimited JSON inference events."""
        count = 0
        for line in stream:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            event = InferenceRequestEvent(
                cluster_id=self.exporter.cluster_id,
                tenant_id=self.exporter.tenant_id,
                engine=self.engine,
                **payload,
            )
            self.exporter.emit(event)
            count += 1
        return count
