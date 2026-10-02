"""Correctness smoke test.

Marked `integration` because it downloads the model and runs real inference.
Run with:  pytest -m integration
Uses greedy decoding (temperature=0) so the result is deterministic.
"""

import pytest

from onnx_lfm_api.generate import GenParams, generate_text
from onnx_lfm_api.model import load_model


@pytest.fixture(scope="module")
def bundle():
    return load_model()


@pytest.mark.integration
def test_capital_of_france(bundle):
    messages = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    out = generate_text(bundle, messages, GenParams(max_tokens=16, temperature=0.0))
    assert out.strip(), "expected non-empty output"
    assert "paris" in out.lower(), f"unexpected answer: {out!r}"
