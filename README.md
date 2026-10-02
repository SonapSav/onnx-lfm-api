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
`LFM_QUANT` selects the precision variant — `q8` or `q4f16` are good CPU
alternatives to the default `q4`.

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
`onnxruntime-gpu` release (1.30 → CUDA 12.x / cuDNN 9); adjust `Dockerfile.gpu`'s
`FROM` tag if you change the ORT version.

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
