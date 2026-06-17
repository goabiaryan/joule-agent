# joule-agent

Open-source telemetry agent for GPU inference fleets. Collects power draw, clock speeds, and inference-phase metrics from vLLM, SGLang, Triton, and other engines — then exports them to your Joule ingest endpoint.

**Auditable.** No closed-source black box on your bare metal.  
**Plug and play.** `pip install` + a few lines in your inference startup.  
**IP-safe.** Collection only — waste classification and simulation live in closed-source `joule-core`.

## Install

```bash
pip install joule-agent

# GPU power sampling (requires NVIDIA drivers on the host)
pip install "joule-agent[gpu]"
```

## Quick start

### 1. Configure export

```bash
export JOULE_ENDPOINT="https://ingest.your-joule-instance.com/v1/events"
export JOULE_API_KEY="your-api-key"
export JOULE_CLUSTER_ID="lisbon-prod-1"
export JOULE_TENANT_ID="acme-labs"
```

### 2. Hook into vLLM

```python
from joule_agent.exporter import Exporter
from joule_agent.integrations.vllm import JouleVLLMCallback

exporter = Exporter.from_env()
callback = JouleVLLMCallback(exporter=exporter, model_id="llama-3.1-70b")

# Pass `callback` to your vLLM engine / scheduler hooks.
# See examples/vllm_sidecar/ for a full docker-compose setup.
```

### 3. Run the GPU sampler (sidecar or separate process)

```bash
joule-sampler
# or: python -m joule_agent.sampler
```

The sampler polls NVML for power, clocks, and utilization every few seconds and attaches readings to in-flight inference events.

## Architecture

```
[vLLM / SGLang / Triton]
        │
        ▼
joule_agent.integrations.*   ← in-process hooks (phase, tokens, latency)
        │
joule_agent.sampler          ← optional sidecar (power, clocks)
        │
        ▼
joule_agent.exporter         ← batch HTTPS export
        │
        ▼
[Joule ingest → joule-core → Dashboard]
```

## Event schema

All events follow the open `joule_agent.schema` definitions:

| Event | Purpose |
|---|---|
| `inference.request` | Per-request: phase, tokens, latency, batch size |
| `gpu.sample` | Periodic: power watts, clock MHz, utilization |
| `gpu.snapshot` | Point-in-time hardware state attached to a request |

See `joule_agent/schema.py` for the full spec.

## Integrations

| Engine | Module | Status |
|---|---|---|
| vLLM | `joule_agent.integrations.vllm` | Callback hooks |
| SGLang | `joule_agent.integrations.sglang` | Stub — PRs welcome |
| Triton | `joule_agent.integrations.triton` | Stub — PRs welcome |
| Generic | `joule_agent.integrations.generic` | Log-line parser fallback |

## Examples

- [`examples/vllm_sidecar/`](examples/vllm_sidecar/) — Docker Compose: vLLM + joule sampler
- [`examples/sglang_plugin/`](examples/sglang_plugin/) — SGLang startup with joule hooks

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `JOULE_ENDPOINT` | Yes | HTTPS URL for event ingest |
| `JOULE_API_KEY` | Yes | API key for authentication |
| `JOULE_CLUSTER_ID` | Yes | Cluster identifier |
| `JOULE_TENANT_ID` | No | Tenant / org identifier |
| `JOULE_BATCH_SIZE` | No | Events per batch (default: 100) |
| `JOULE_FLUSH_INTERVAL` | No | Seconds between flushes (default: 5) |
| `JOULE_SAMPLER_INTERVAL` | No | GPU sample interval in seconds (default: 2) |

## License

Apache 2.0 — see [LICENSE](LICENSE).
