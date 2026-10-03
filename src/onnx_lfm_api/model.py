from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .config import settings
from .prefix_cache import PrefixCache

# Special chat-end token for the LFM2 ChatML-like template. We stop on this in
# addition to the tokenizer's EOS so generation ends cleanly at turn boundaries.
IM_END = "<|im_end|>"


def _model_basename(quant: str) -> str:
    return "model" if quant in ("", "fp32") else f"model_{quant}"


def _resolve_model_dir() -> str:
    """Return a local directory holding the model as real, co-located files.

    The ``.onnx`` graph references its external-data shards (``*.onnx_data*``)
    by relative path, and newer ONNX Runtime rejects symlinks that resolve
    outside the graph's directory — which is exactly how the HF blob cache
    stores them. ``snapshot_download(local_dir=...)`` materializes real files
    side by side, which ORT accepts. Tokenizer/config/template files are pulled
    in too so everything loads from one directory (offline-friendly).
    """
    if settings.local_model_dir:
        return settings.local_model_dir

    base = _model_basename(settings.quant)
    return snapshot_download(
        settings.model_repo,
        allow_patterns=[
            f"onnx/{base}.onnx",
            f"onnx/{base}.onnx_data*",
            "*.json",
            "*.jinja",
        ],
        local_dir=settings.cache_dir,
    )


@dataclass
class ModelBundle:
    session: ort.InferenceSession
    tokenizer: Any
    input_names: set[str]
    stop_ids: set[int]
    providers: list[str]
    io_binding: bool = False  # decode keeps the cache on the GPU (see generate.py)
    boundary_id: int | None = None  # <|im_end|>: message boundary for prefix snapshots
    prefix_cache: PrefixCache | None = None


def use_io_binding(session: ort.InferenceSession, mode: str) -> bool:
    """Whether to keep the cache on the GPU between steps (LFM_IO_BINDING).

    Only pays off when attention runs on the GPU too: ORT's CUDA
    GroupQueryAttention is fp16/bf16-only, so with an fp32 cache (q4, fp32) it
    falls back to CPU and a GPU-resident cache adds a copy each way per layer.
    GTX 1660, decode at 64/700/2000 tokens of context: q4f16 x1.08/1.24/1.34,
    fp16 x1.03/1.12/1.20, q4 x0.89/0.68/0.61.
    """
    if mode == "off" or "CUDAExecutionProvider" not in session.get_providers():
        return False
    if mode == "on":
        return True
    return any(i.type in ("tensor(float16)", "tensor(bfloat16)")
               for i in session.get_inputs() if i.name.startswith("past_key_values"))


def load_model() -> ModelBundle:
    """Load the ONNX session + tokenizer once. Called at app startup."""
    model_dir = _resolve_model_dir()
    base = _model_basename(settings.quant)
    model_path = str(Path(model_dir) / "onnx" / f"{base}.onnx")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ONNX file not found: {model_path}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = settings.intra_op_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path, sess_options=so, providers=settings.providers
    )
    io_binding = use_io_binding(session, settings.io_binding)
    # Report which EPs actually engaged. ORT silently falls back to CPU if a
    # requested provider (e.g. CUDA) can't load, so this makes GPU use verifiable.
    print(
        f"[onnx-lfm-api] loaded {model_path} | providers={session.get_providers()}"
        f" | io_binding={io_binding}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    input_names = {i.name for i in session.get_inputs()}

    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    if isinstance(im_end_id, int) and im_end_id >= 0:
        stop_ids.add(im_end_id)
    else:
        im_end_id = None

    return ModelBundle(session=session, tokenizer=tokenizer,
                       input_names=input_names, stop_ids=stop_ids,
                       providers=session.get_providers(), io_binding=io_binding,
                       boundary_id=im_end_id,
                       prefix_cache=PrefixCache(settings.prefix_cache_size)
                       if settings.prefix_cache_size > 0 else None)
