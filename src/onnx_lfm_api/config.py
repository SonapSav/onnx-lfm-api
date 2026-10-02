from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration. Every field is overridable via an ``LFM_`` env var
    (e.g. ``LFM_QUANT=q8``), which keeps the container fully config-driven."""

    model_config = SettingsConfigDict(
        env_prefix="LFM_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),  # we use "model_*" field names deliberately
    )

    # --- Model selection ---
    model_repo: str = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"
    # "" / "fp32" -> model.onnx ; otherwise model_<quant>.onnx
    # options: "", "fp16", "q4", "q4f16", "q8", "quantized"
    quant: str = "q4"
    # If set, load from this local dir instead of downloading from the Hub.
    local_model_dir: str | None = None
    # Directory into which Hub files are materialized as real, co-located files.
    # (ONNX Runtime requires the .onnx and its .onnx_data shards to live in the
    # same real directory — the HF symlinked blob cache does not satisfy this.)
    cache_dir: str = "models"

    # --- ONNX Runtime ---
    intra_op_threads: int = 0  # 0 = let ORT pick (all cores)
    providers: list[str] = ["CPUExecutionProvider"]

    # --- Generation defaults (Liquid's recommended settings) ---
    max_tokens: int = 256
    temperature: float = 0.1
    top_k: int = 50
    repetition_penalty: float = 1.05

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8383
    # If set, /chat endpoints require a matching "X-API-Key" header. Leave empty
    # to disable auth (fine for a fully trusted LAN). /health is always open.
    api_key: str | None = None


settings = Settings()
