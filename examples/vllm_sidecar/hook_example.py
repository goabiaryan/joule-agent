"""
Example: wire joule-agent into a vLLM OpenAI-compatible server.

This is a reference layout — exact hook points depend on your vLLM version.
For most beta deployments, run the GPU sampler as a sidecar and add
explicit callback calls in your serving middleware.
"""

from joule_agent.exporter import Exporter
from joule_agent.integrations.vllm import JouleVLLMCallback

exporter = Exporter.from_env()
exporter.start()

callback = JouleVLLMCallback(
    exporter=exporter,
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_index=0,
)


def track_openai_completion(request_id: str, prompt_tokens: int, completion_tokens: int, latency_ms: float) -> None:
    callback.on_prefill_start(request_id=request_id, tokens_in=prompt_tokens)
    callback.on_prefill_end(request_id=request_id, tokens_in=prompt_tokens, latency_ms=latency_ms * 0.2)
    callback.on_decode_start(request_id=request_id)
    callback.on_decode_end(request_id=request_id, tokens_out=completion_tokens, latency_ms=latency_ms * 0.8)


if __name__ == "__main__":
    demo_id = callback.on_request_start(tokens_in=128, batch_size=4)
    callback.on_decode_end(request_id=demo_id, tokens_out=64, latency_ms=1800.0)
    exporter.stop()
    print("Emitted demo vLLM events")
