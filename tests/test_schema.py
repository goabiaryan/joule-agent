from joule_agent.schema import InferencePhase, InferenceRequestEvent, GPUSampleEvent, EventBatch
from joule_agent.exporter import Exporter
from joule_agent.integrations.generic import GenericLogParser


def test_inference_request_event_export():
    event = InferenceRequestEvent(
        cluster_id="test-cluster",
        request_id="req-1",
        model_id="llama-70b",
        phase=InferencePhase.DECODE,
        tokens_out=128,
        latency_ms=2100.0,
    )
    payload = event.to_export()
    assert payload["event_type"] == "inference.request"
    assert payload["phase"] == "decode"
    assert payload["tokens_out"] == 128


def test_gpu_sample_event_export():
    event = GPUSampleEvent(
        cluster_id="test-cluster",
        gpu_index=0,
        power_watts=412.5,
        clock_sm_mhz=1980,
    )
    payload = event.to_export()
    assert payload["event_type"] == "gpu.sample"
    assert payload["power_watts"] == 412.5


def test_event_batch_export():
    batch = EventBatch(agent_version="0.1.0", events=[{"event_type": "gpu.sample"}])
    payload = batch.to_export()
    assert payload["schema_version"] == "1.0"
    assert len(payload["events"]) == 1


def test_generic_log_parser():
    exporter = Exporter(
        endpoint="https://example.com/v1/events",
        api_key="test-key",
        cluster_id="test-cluster",
    )
    parser = GenericLogParser(exporter)
    line = "joule inference phase=decode model=llama-70b request_id=abc latency_ms=2100 tokens_out=128 batch=8"
    event = parser.parse_line(line)
    assert event is not None
    assert event.model_id == "llama-70b"
    assert event.phase == InferencePhase.DECODE
    assert event.tokens_out == 128
