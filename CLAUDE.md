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

## Layout
- `config.py` — `LFM_*` env settings (pydantic-settings; reads `./.env`).
- `model.py` — downloads + loads ONNX session + tokenizer **once** at startup.
- `generate.py` — the dual-state decode loop + sampling + streaming (the core).
- `api.py` — `/health`, `/chat`, `/chat/stream`, OpenAI `/v1/*`; generation serialized with a lock.
- `chat_cli.py` — interactive client (`lfm-chat`).

## Conventions
- All config is env vars prefixed `LFM_`. Secrets/quant live in gitignored `.env`.
  `LFM_QUANT` default `q4`; on-disk cache in `models/`. (Deployed default: `q4f16`.)
- Model loads once and stays resident; generation is serialized (single worker).
- GPU is opt-in only; CPU is the default build. `/health` reports active ORT providers.

## Gotchas (learned the hard way)
- **ONNX Runtime ≥1.30 rejects the HF symlink blob cache** ("external data path
  escapes model directory"). `model.py` uses `snapshot_download(local_dir=...)`
  to materialize real co-located files — keep it that way.
- **Never commit `models/` (weights, GBs) or `.env` (secret).** Already gitignored.
- **Rebuild the Docker image after source changes** — the image does not auto-update.
- On this host, port **8000** is taken (portainer) — the app uses **8383**.
- **Don't `pkill -f onnx_lfm_api`** — the pattern matches the running shell and kills it. Target the PID/container.

## Git
Branch `main`; remote `SonapSav/onnx-lfm-api`. Repo-local identity:
Panos Vasilopoulos <sonap.sav@gmail.com> (global config is intentionally different).
