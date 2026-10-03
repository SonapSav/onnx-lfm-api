# onnx-lfm-api

A small FastAPI service that serves [LiquidAI LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX)
on CPU via ONNX Runtime. It adapts Liquid's reference decode loop, which threads
both the GQA KV cache **and** the convolution-block state through each step —
required for the hybrid LFM2 architecture.

## Quick start (local, dedicated venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# (optional) warm the model cache up front
python scripts/download_model.py

# run the API
python -m onnx_lfm_api
# or: uvicorn onnx_lfm_api.api:app --host 0.0.0.0 --port 8383
```

First request downloads the model (a few hundred MB for `q4`).

## Endpoints

- `GET  /health` — liveness + model/quant + active ONNX Runtime providers (open, no auth)
- `POST /chat` — `{"messages":[{"role":"user","content":"..."}]}` → `{"reply":"..."}`
- `POST /chat/stream` — same body, Server-Sent-Events stream of `{"delta":"..."}` chunks

```bash
curl -X POST http://localhost:8383/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

Optional per-request overrides: `max_tokens`, `temperature`, `top_k`, `repetition_penalty`.

### Authentication

Set `LFM_API_KEY` to require a matching `X-API-Key` header on the `/chat`
endpoints (`/health` stays open for probes). Leave it unset for a fully trusted
LAN.

```bash
curl -X POST http://localhost:8383/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

## OpenAI-compatible API

Point any OpenAI client at `http://<host>:8383/v1`:

- `GET  /v1/models`
- `POST /v1/chat/completions` (supports `stream: true`)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8383/v1", api_key="change-me")  # key only needed if LFM_API_KEY is set
resp = client.chat.completions.create(
    model="lfm2.5",  # name is echoed back; the served model is fixed
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(resp.choices[0].message.content)
```

The key is accepted as either `Authorization: Bearer <key>` (OpenAI default) or
`X-API-Key: <key>`. Unimplemented OpenAI fields are ignored rather than
rejected; `temperature` and `max_tokens`/`max_completion_tokens` are honored.

### Tool calling

`/v1/chat/completions` supports OpenAI-style function calling. Pass `tools`;
when the model calls one, the response has `finish_reason: "tool_calls"` and
`message.tool_calls` (arguments as a JSON string). Append the result as a
`role: "tool"` message to continue the conversation.

```python
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

msgs = [{"role": "user", "content": "What's the weather in Paris?"}]
r = client.chat.completions.create(model="lfm2.5", messages=msgs, tools=tools)
call = r.choices[0].message.tool_calls[0]          # -> get_weather {"city": "Paris"}

msgs += [
    r.choices[0].message,                           # the assistant tool call
    {"role": "tool", "tool_call_id": call.id,
     "content": '{"temp_c": 18, "conditions": "sunny"}'},
]
final = client.chat.completions.create(model="lfm2.5", messages=msgs, tools=tools)
print(final.choices[0].message.content)             # -> "...18°C with sunny conditions."
```

Under the hood LFM2 emits calls as `<|tool_call_start|>[fn(arg=val)]<|tool_call_end|>`,
which the server parses (via `ast`, no `eval`) into OpenAI `tool_calls`.

## CLI chat client

An interactive, streaming, multi-turn chat client ships with the package. It
talks to an already-running server (e.g. the Docker container) and auto-reads
`LFM_API_KEY` from `./.env` — you don't need to start anything else.

```bash
# the CLI lives in the venv, so activate it first:
source .venv/bin/activate
lfm-chat
# ...or run it without activating:
.venv/bin/lfm-chat
# ...or as a module:
.venv/bin/python -m onnx_lfm_api.chat_cli

# point at a LAN server, set a system prompt:
lfm-chat --url http://192.168.0.125:8383/v1 --system "You are concise."
```

If the server isn't up yet, start it first with `docker compose up -d`.

In-chat commands: `/reset` (clear history), `/system <text>`, `/exit`.
Flags: `--model`, `--max-tokens`, `--temperature`, `--no-stream`.

## Configuration

All settings are env vars prefixed `LFM_` (see `.env.example`). Notably
`LFM_QUANT` selects the precision variant (default `q4`; also `q4f16`, `q8`,
`fp16`, `""` for fp32).

### Performance tuning (CPU)
Set `LFM_INTRA_OP_THREADS` to your **physical** core count. The default (`0`)
lets ONNX Runtime use every logical CPU, and the SMT siblings compete for the
same cores: slower *and* the whole machine is pegged. Measured on a Ryzen 5
5500U (6 cores / 12 threads), temperature 0, median of 3:

| `LFM_QUANT` | threads | generation | tool-calling request (706 prompt + 34 out) | CPU |
|---|---|---|---|---|
| q4f16 | 0 (=12) | 14.7 tok/s | 11.3 s | ~11.6 cores |
| q4f16 | 6 | 17.4 tok/s | 6.8 s | 6 cores |
| **q4** | **6** | **17.6 tok/s** | **6.5 s** | **6 cores** |
| q4 | 8 | 18.3 tok/s | 7.3 s | 8 cores |
| q8 | 6 | 20.2 tok/s | 9.4 s | 6 cores |

Tool-calling requests are dominated by prompt processing (tool schemas +
history), where `q4` is fastest; `q8` decodes faster but prefills ~45% slower.
More uvicorn workers don't help: one request already uses all assigned cores,
and each worker loads its own copy of the model.

### Prompt-prefix caching
Requests that start like an earlier one skip re-reading that start. Prefill is
split at message boundaries (after `<|im_end|>`), and the model state there is
kept in a small LRU (`LFM_PREFIX_CACHE_SIZE`, default 8, `0` = off). The
snapshots are the first boundary (system message + tool schemas, shared by
every call) and the last (the conversation so far, which the next agent round
extends). LFM2's conv state can't be rewound, so states are only reused at
those exact points. Greedy outputs match the uncached path (tested), and
`/health` reports `prefix_cache` hits, misses and reused tokens. Each snapshot
costs ~25 KB per prompt token for q4 (~20 MB for an agent prompt), in VRAM
under IO binding.

| tool-calling request (706 prompt + 34 out) | no cache | cached prefix |
|---|---|---|
| Ryzen 5 5500U, q4, 6 threads | 6.46 s | **2.06 s** |
| GTX 1660, q4 | 1.35 s | **0.36 s** |

Real agent runs (onnx-lfm-agent live evals, 48 runs, GTX 1660): 141 of 148
lookups hit, and calls take 0.5–0.9 s instead of 1.5–2.2 s, with the same
pass rates.

## Docker

```bash
docker build -t onnx-lfm-api .
docker run -p 8383:8383 -v "$PWD/models:/models" onnx-lfm-api
```

The model is downloaded at runtime into the mounted `/models` volume, keeping
the image small and avoiding re-downloads across restarts.

### GPU (NVIDIA/CUDA) — opt-in

The default image is CPU-only. For an NVIDIA GPU host, use the GPU variant,
which installs `onnxruntime-gpu`, defaults to the `fp16` model, and runs on the
`CUDAExecutionProvider` (falling back to CPU). Requires an NVIDIA driver + the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Verify CUDA actually engaged (ORT silently falls back to CPU if it can't load):

```bash
curl -s http://localhost:8383/health      # -> "providers": ["CUDAExecutionProvider", ...]
```

The same list is printed in the startup logs (`providers=[...]`). If you only
see `CPUExecutionProvider`, CUDA didn't load — usually a driver/toolkit issue or
a CUDA/cuDNN mismatch. The base image's CUDA/cuDNN version must match the
`onnxruntime-gpu` release (1.30 → CUDA 13.x / cuDNN 9, driver ≥ 580, Turing or
newer); adjust `Dockerfile.gpu`'s `FROM` tag if you change the ORT version.

**Pick the quant per card** (`LFM_GPU_QUANT` in `./.env`, default `fp16`). On a
GTX 1660 (6 GB, no tensor cores) ORT's fp16 math is *slower than fp32*, and `q4`
wins. Raw ORT session, 700-token prompt:

| quant | prefill 700 tok | decode |
|---|---|---|
| fp16 | 3.05 s | 56 tok/s |
| **q4** | **1.17 s** | **148 tok/s** |
| q4f16 | 3.17 s | 161 tok/s |
| fp32 | 0.51 s | 34 tok/s |

Through the API with q4, a tool-calling request (706 prompt + 34 output tokens)
takes 1.35 s, against 6.46 s on a 6-core Ryzen CPU (q4, 6 threads). Re-measure
on cards with tensor cores (and Jetson Orin), where fp16 should do better.

**IO binding** (`LFM_IO_BINDING`, default `auto`) keeps the per-token cache on the
GPU between decode steps instead of round-tripping it through host memory.
`auto` turns it on only when CUDA is active **and** the cache is fp16/bf16
(fp16, q4f16). ORT's CUDA attention kernel (`GroupQueryAttention`) is fp16/bf16
only. With an fp32 cache (q4, fp32) attention runs on the CPU, and a GPU-resident
cache just adds copies: 0.6–0.9× in the same test. `/health` reports
`"io_binding"`. Outputs are token-identical either way. GTX 1660, decode speed:

| q4f16, context | 64 | 700 | 2000 | 2300 (via the API) |
|---|---|---|---|---|
| speed-up | 1.08× | 1.24× | 1.34× | **1.36×** (41.5 → 56.5 tok/s) |

On the 1660, q4 is still fastest overall (63.8 tok/s at 2300 context), because
its fp32 matmuls are what this card does best.

### docker compose (recommended for LAN)

```bash
docker compose up -d        # builds, starts, restarts on reboot
docker compose ps           # shows health status
docker compose logs -f
```

`docker-compose.yml` publishes `8383` on all interfaces, mounts `./models`,
sets `restart: unless-stopped`, and includes a `/health` healthcheck.

To require an API key, put it in a gitignored `.env` file (compose loads it
automatically, and so does the local venv run — same file):

```bash
echo 'LFM_API_KEY=your-secret-here' >> .env
docker compose up -d
```

Leaving `.env` without `LFM_API_KEY` keeps auth disabled (trusted-LAN default).

## Tests

```bash
pytest -m integration   # downloads the model, runs a real greedy inference
```
