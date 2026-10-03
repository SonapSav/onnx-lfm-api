"""Tool-calling correctness (integration — downloads model, real inference).

Run with:  pytest -m integration
"""

import pytest

from onnx_lfm_api.generate import GenParams, generate_turn
from onnx_lfm_api.model import load_model

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


@pytest.fixture(scope="module")
def bundle():
    return load_model()


@pytest.mark.integration
def test_emits_tool_call(bundle):
    messages = [{"role": "user", "content": "What's the weather in Paris right now?"}]
    r = generate_turn(bundle, messages, GenParams(max_tokens=48, temperature=0.0), tools=TOOLS)
    assert r["finish_reason"] == "tool_calls"
    assert r["tool_calls"], "expected a tool call"
    call = r["tool_calls"][0]
    assert call["name"] == "get_weather"
    assert call["arguments"].get("city", "").lower() == "paris"


@pytest.mark.integration
def test_uses_tool_result(bundle):
    # History already in the template's string form (what the server produces).
    messages = [
        {"role": "user", "content": "What's the weather in Paris right now?"},
        {"role": "assistant", "content": '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>'},
        {"role": "tool", "content": '{"temp_c": 18, "conditions": "sunny"}'},
    ]
    r = generate_turn(bundle, messages, GenParams(max_tokens=48, temperature=0.0), tools=TOOLS)
    assert r["tool_calls"] is None
    assert "18" in (r["content"] or ""), f"expected the tool result in the answer: {r['content']!r}"
