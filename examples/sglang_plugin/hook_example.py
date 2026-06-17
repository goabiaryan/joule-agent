"""
Example: SGLang-style FastAPI middleware with joule-agent hooks.

Run alongside your SGLang server and call `track_generation` from your
completion endpoint.
"""

from joule_agent.exporter import Exporter
from joule_agent.integrations.sglang import JouleSGLangCallback
from joule_agent.schema import InferencePhase

exporter = Exporter.from_env()
exporter.start()

callback = JouleSGLangCallback(
    exporter=exporter,
    model_id="qwen2.5-72b",
)


def track_generation(prompt_tokens: int, completion_tokens: int, latency_ms: float) -> None:
    request_id = callback.begin(tokens_in=prompt_tokens, batch_size=1)
    callback.track_request(
        phase=InferencePhase.PREFILL,
        tokens_in=prompt_tokens,
        latency_ms=latency_ms * 0.25,
        request_id=request_id,
    )
    callback.complete(request_id=request_id, tokens_out=completion_tokens)


if __name__ == "__main__":
    track_generation(prompt_tokens=256, completion_tokens=128, latency_ms=2200.0)
    exporter.stop()
    print("Emitted demo SGLang events")
