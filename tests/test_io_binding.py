"""When decode keeps the cache on the GPU (LFM_IO_BINDING). No model or GPU
needed: the session is a stand-in exposing what use_io_binding() reads."""

from types import SimpleNamespace as NS

import pytest

from onnx_lfm_api.model import use_io_binding

CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU = ["CPUExecutionProvider"]


def session(providers, kv_type):
    inputs = [NS(name="input_ids", type="tensor(int64)"),
              NS(name="past_conv.0", type="tensor(float)"),
              NS(name="past_key_values.2.key", type=kv_type)]
    return NS(get_providers=lambda: providers, get_inputs=lambda: inputs)


@pytest.mark.parametrize("providers, kv_type, mode, expected", [
    (CUDA, "tensor(float16)", "auto", True),     # fp16 / q4f16 on GPU
    (CUDA, "tensor(bfloat16)", "auto", True),
    (CUDA, "tensor(float)", "auto", False),      # q4 / fp32: attention falls back to CPU
    (CUDA, "tensor(float)", "on", True),
    (CUDA, "tensor(float16)", "off", False),
    (CPU, "tensor(float16)", "auto", False),     # no GPU, nothing to bind to
    (CPU, "tensor(float16)", "on", False),
])
def test_use_io_binding(providers, kv_type, mode, expected):
    assert use_io_binding(session(providers, kv_type), mode) is expected
