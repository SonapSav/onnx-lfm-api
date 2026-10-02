"""Pre-fetch the ONNX model (+ external data) and tokenizer into the HF cache.

Useful for warming a Docker volume before first request so the initial
/chat call isn't blocked on a multi-hundred-MB download.

Run:  python scripts/download_model.py
"""

from transformers import AutoTokenizer

from onnx_lfm_api.config import settings
from onnx_lfm_api.model import _resolve_model_dir


def main() -> None:
    print(f"Repo:  {settings.model_repo}")
    print(f"Quant: {settings.quant or 'fp32'}")
    model_dir = _resolve_model_dir()
    AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    print(f"Model ready in: {model_dir}")


if __name__ == "__main__":
    main()
