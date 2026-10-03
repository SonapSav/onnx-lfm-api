# CLAUDE.md

Guidance for working in this repo. Keep it short and current.

## What this is
FastAPI service serving **LiquidAI LFM2.5-1.2B-Instruct** on CPU via **raw ONNX
Runtime**. It adapts Liquid's reference decode loop, which threads **both** the
GQA KV cache and the convolution-block state each step — required for the hybrid
LFM2 architecture. Package: `src/onnx_lfm_api/`.

## Setup / run / test
- **Venv:** use `uv` (the system lacks `python3.13-venv`/ensurepip, so plain
  `python -m venv` fails):
  ```bash
  uv venv .venv --python 3.13
  uv pip install --python .venv/bin/python -e ".[dev]"
  ```
- **Run:** `.venv/bin/python -m onnx_lfm_api` (or `uvicorn onnx_lfm_api.api:app`). Port **8383**.
- **Chat CLI:** `.venv/bin/lfm-chat` (talks to a running server; reads `.env`).
- **Tests:** `.venv/bin/python -m pytest -m integration` (downloads model, real inference).
- **Docker:** `docker compose up -d` (CPU). GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`.
  GPU image = CUDA 13 base (ORT 1.30 wheel); per-card quant via `LFM_GPU_QUANT` (default fp16; **q4 on a GTX 1660**,
  where fp16 math is slower than fp32 — README GPU section). `LFM_IO_BINDING=auto|on|off` (default auto): GPU-resident
  cache between decode steps, auto = CUDA + fp16/bf16 cache only (`model.use_io_binding`; ORT's CUDA GQA is fp16/bf16
  only, so q4/fp32 run attention on CPU and binding is slower there). Token-identical; q4f16 1.36x decode at 2.3k ctx.
  Casting the q4 graph's attention to fp16 (so GQA runs on CUDA) was tried on the GTX 1660 and dropped: tokens
  identical, but decode no faster (slower at long context): Turing's CUDA GQA is no faster than the CPU one. Retry on sm_80+.

## Layout
- `config.py` — `LFM_*` env settings (pydantic-settings; reads `./.env`).
- `model.py` — downloads + loads ONNX session + tokenizer **once** at startup.
- `generate.py` — the dual-state decode loop + sampling + streaming (the core).
- `api.py` — `/health`, `/chat`, `/chat/stream`, OpenAI `/v1/*`; generation serialized with a lock.
  - Tool calling: `/v1/chat/completions` accepts OpenAI `tools`. LFM2 emits
    `<|tool_call_start|>[fn(arg=val)]<|tool_call_end|>` (token ids 10/11);
    `generate.generate_turn` + `_parse_tool_calls` (via `ast`, no eval) convert
    to OpenAI `tool_calls`. Tool/assistant history is mapped back to the
    template's string form in `api._to_template_messages`.
- `chat_cli.py` — interactive client (`lfm-chat`).

## Conventions
- All config is env vars prefixed `LFM_`. Secrets/quant live in gitignored `.env`.
  `LFM_QUANT` default `q4`; on-disk cache in `models/`. Deployed (this host's `.env`):
  `LFM_QUANT=q4`, `LFM_INTRA_OP_THREADS=6` — chosen by benchmark, see README "Performance tuning".
- Model loads once and stays resident; generation is serialized (single worker).
- GPU is opt-in only; CPU is the default build. `/health` reports active ORT providers.

## Gotchas (learned the hard way)
- **ONNX Runtime ≥1.30 rejects the HF symlink blob cache** ("external data path
  escapes model directory"). `model.py` uses `snapshot_download(local_dir=...)`
  to materialize real co-located files — keep it that way.
- **Never commit `models/` (weights, GBs) or `.env` (secret).** Already gitignored.
- **Rebuild the Docker image after source changes** — the image does not auto-update.
- **`LFM_INTRA_OP_THREADS=0` (ORT default) is slower than physical-core count** on this 6c/12t
  Ryzen: 11.3 s vs 6.5 s per tool-calling request, at double the CPU. SMT siblings contend.
  Keep the code/compose default 0 (portable); set the host value in `.env`.
- **More workers / a second instance don't help** — measured (q4, batch of 6 tool-calling requests):
  1×6 threads 38.6 s; 2 instances×3 threads concurrent 37.9 s (noise); 2×6 threads 55.7 s. Decode is
  memory-bandwidth bound (weights re-read per token), so instances split the same bandwidth. The real
  lever for agent workloads is prompt-prefix caching (tool schemas + history are re-prefilled every call).
- **LFM2.5-1.2B-Thinking** (`LFM_MODEL_REPO=LiquidAI/LFM2.5-1.2B-Thinking-ONNX`) loads and tool-calls with this
  code (same template), but: (1) its ONNX files have the **same names** as Instruct's — give it its own
  `LFM_CACHE_DIR` (q4 is downloaded in `models/lfm2.5-1.2b-thinking/`), else it overwrites the Instruct model;
  (2) it reasons ~1000–1350 tokens before a tool call, so `LFM_MAX_TOKENS=256` cuts it off (≈70–90 s/call on this
  CPU); (3) `<think>`/`</think>` are plain tokens, so the reasoning lands in `content` — split it into
  `reasoning_content` before adopting it. Not adopted: on a GTX 1660 (q4, 2048 max_tokens) it ran 10–21 s/call
  and still hit the cap; details in the agent repo's CLAUDE.md.
- On this host, port **8000** is taken (portainer) — the app uses **8383**.
- **Don't `pkill -f onnx_lfm_api`** — the pattern matches the running shell and kills it. Target the PID/container.

## Git
Branch `main`; remote `SonapSav/onnx-lfm-api`. Repo-local identity:
Panos Vasilopoulos <sonap.sav@gmail.com> (global config is intentionally different).
