from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .config import settings
from .model import ModelBundle
from .prefix_cache import prefill_cuts

# Map ORT input dtype strings to numpy dtypes for building the empty cache.
_DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
}

_CORE_INPUTS = {"input_ids", "attention_mask", "position_ids"}

# LFM2 emits tool calls as  <|tool_call_start|>[fn(arg=val), ...]<|tool_call_end|>
TOOL_CALL_START_ID = 10
TOOL_CALL_END_ID = 11


@dataclass
class GenParams:
    max_tokens: int = settings.max_tokens
    temperature: float = settings.temperature
    top_k: int = settings.top_k
    repetition_penalty: float = settings.repetition_penalty


def _init_cache(session) -> dict[str, np.ndarray]:
    """Build the initial (empty) past-state tensors.

    This adapts to whatever cache the graph declares — for LFM2 that's both the
    GQA KV cache (``past_key_values.*``) *and* the conv-block state
    (``past_conv*``). The sequence dimension starts at length 0.
    """
    cache: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        if inp.name in _CORE_INPUTS:
            continue
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        for i, d in enumerate(inp.shape):
            if isinstance(d, str) and "sequence" in d.lower():
                shape[i] = 0
        dtype = _DTYPE_MAP.get(inp.type, np.float32)
        cache[inp.name] = np.zeros(shape, dtype=dtype)
    return cache


def _sample(logits: np.ndarray, generated: list[int], p: GenParams) -> int:
    """Pick the next token id. temperature<=0 is exact greedy (deterministic)."""
    logits = logits.astype(np.float32).copy()

    # Repetition penalty (HF-style: divide positive logits, multiply negative).
    if p.repetition_penalty and p.repetition_penalty != 1.0 and generated:
        for t in set(generated):
            logits[t] = (logits[t] / p.repetition_penalty
                         if logits[t] > 0 else logits[t] * p.repetition_penalty)

    if not p.temperature or p.temperature <= 0:
        return int(np.argmax(logits))

    logits /= p.temperature
    k = min(p.top_k if p.top_k and p.top_k > 0 else logits.shape[-1], logits.shape[-1])
    top_idx = np.argpartition(logits, -k)[-k:]
    top_logits = logits[top_idx]
    top_logits -= top_logits.max()
    probs = np.exp(top_logits)
    probs /= probs.sum()
    return int(np.random.choice(top_idx, p=probs))


def _run_bound(session, binding, out_names: list[str], core: dict, cache: dict):
    """One step with the cache kept on the GPU (ORT IO binding).

    The past* inputs are the previous step's present* outputs, still on the
    device, so the cache never round-trips through host memory; only the logits
    come back. Returns (logits ndarray, present* OrtValues in output order).
    """
    binding.clear_binding_inputs()
    binding.clear_binding_outputs()
    for name, value in core.items():
        binding.bind_cpu_input(name, value)
    for name, value in cache.items():
        if isinstance(value, np.ndarray):  # step 0: the empty initial cache
            binding.bind_cpu_input(name, value)
        else:
            binding.bind_ortvalue_input(name, value)
    binding.bind_output(out_names[0], "cpu")
    for name in out_names[1:]:
        binding.bind_output(name, "cuda", 0)
    session.run_with_iobinding(binding)
    outputs = binding.get_outputs()
    return outputs[0].numpy(), outputs[1:]


def build_prompt(bundle: ModelBundle, messages: list[dict], tools: list | None = None) -> str:
    return bundle.tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )


def generate_ids(
    bundle: ModelBundle,
    messages: list[dict],
    p: GenParams,
    tools: list | None = None,
    extra_stop_ids: set[int] | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time (stop tokens are not yielded).

    This is the heart of the service — Liquid's reference decode loop, which
    threads both the attention KV cache and the convolution state back in on
    every step via the ``present* -> past*`` remap.

    ``tools`` are rendered into the prompt by the chat template; ``extra_stop_ids``
    lets callers stop early (e.g. at ``<|tool_call_end|>`` once a call is emitted).
    """
    session = bundle.session
    stop_ids = bundle.stop_ids | (extra_stop_ids or set())
    out_names = [o.name for o in session.get_outputs()]
    binding = session.io_binding() if bundle.io_binding else None

    def forward(ids: list[int], start: int, cache: dict) -> tuple[np.ndarray, dict]:
        """Run `ids` at positions start.. on top of `cache`.
        Returns (logits of the last position, the next cache)."""
        total = start + len(ids)
        core = {"input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, total), dtype=np.int64)}
        if "position_ids" in bundle.input_names:
            core["position_ids"] = np.arange(start, total, dtype=np.int64).reshape(1, -1)
        if binding is None:
            outputs = session.run(None, {**core, **cache})
            logits, states = outputs[0], outputs[1:]
        else:
            logits, states = _run_bound(session, binding, out_names, core, cache)
        # Feed each present* state back as the matching past* input next step.
        cache = dict(cache)
        for out_name, state in zip(out_names[1:], states):
            name = out_name.replace("present_conv", "past_conv")
            name = name.replace("present.", "past_key_values.")
            if name in cache:
                cache[name] = state
        return logits[0, -1], cache

    prompt = build_prompt(bundle, messages, tools)
    prompt_ids = bundle.tokenizer.encode(prompt, add_special_tokens=False)

    # Prefill, resuming from the longest cached prefix and snapshotting at
    # message boundaries for later prompts (see prefix_cache.py).
    pc = bundle.prefix_cache
    start, cache = pc.lookup(prompt_ids) if pc else (0, None)
    cache = cache or _init_cache(session)
    cuts = prefill_cuts(prompt_ids, start, bundle.boundary_id) if pc else [len(prompt_ids)]
    for cut in cuts:
        logits, cache = forward(prompt_ids[start:cut], start, cache)
        if pc and cut < len(prompt_ids):
            pc.store(prompt_ids[:cut], cache)
        start = cut

    generated: list[int] = []
    for step in range(p.max_tokens):
        if step:
            logits, cache = forward([generated[-1]], len(prompt_ids) + step - 1, cache)
        next_token = _sample(logits, generated, p)
        generated.append(next_token)
        if next_token in stop_ids:
            break
        yield next_token


def generate_text(bundle: ModelBundle, messages: list[dict], p: GenParams) -> str:
    ids = list(generate_ids(bundle, messages, p))
    return bundle.tokenizer.decode(ids, skip_special_tokens=True)


def generate_collect(bundle: ModelBundle, messages: list[dict], p: GenParams) -> tuple[str, int, int]:
    """Return (text, prompt_tokens, completion_tokens) for OpenAI-style usage."""
    tokenizer = bundle.tokenizer
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    ids = list(generate_ids(bundle, messages, p))
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text, prompt_tokens, len(ids)


def generate_stream(bundle: ModelBundle, messages: list[dict], p: GenParams) -> Iterator[str]:
    """Yield decoded text deltas. Decodes the full id list each step and emits
    the suffix, which keeps multi-token characters intact across chunks."""
    tokenizer = bundle.tokenizer
    produced: list[int] = []
    prev_text = ""
    for tid in generate_ids(bundle, messages, p):
        produced.append(tid)
        text = tokenizer.decode(produced, skip_special_tokens=True)
        if len(text) > len(prev_text):
            yield text[len(prev_text):]
            prev_text = text


def _parse_tool_calls(inner: str) -> list[dict] | None:
    """Parse ``fn(a=1, b="x"), fn2(...)`` (the body between the tool-call tokens)
    into ``[{"name": str, "arguments": dict}, ...]`` using ``ast`` — no eval."""
    inner = inner.strip()
    if not inner:
        return None
    # The body is a Python list literal of calls: [fn(...), ...]. Tolerate a
    # bare call without the brackets too.
    expr = inner if inner.startswith("[") else f"[{inner}]"
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None
    elts = node.elts if isinstance(node, ast.List) else [node]

    calls: list[dict] = []
    for call in elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        args: dict[str, Any] = {}
        for i, a in enumerate(call.args):  # positional (best-effort)
            try:
                args[f"arg{i}"] = ast.literal_eval(a)
            except Exception:
                pass
        for kw in call.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                args[kw.arg] = None
        calls.append({"name": call.func.id, "arguments": args})
    return calls or None


def generate_turn(
    bundle: ModelBundle, messages: list[dict], p: GenParams, tools: list | None = None
) -> dict:
    """Non-streaming generation that also detects a tool call.

    Returns a dict: ``content`` (str|None), ``tool_calls`` (list|None),
    ``finish_reason`` ('tool_calls'|'stop'|'length'), and token counts. When
    ``tools`` are provided, generation stops right after ``<|tool_call_end|>``.
    """
    tokenizer = bundle.tokenizer
    prompt = build_prompt(bundle, messages, tools)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    extra_stop = {TOOL_CALL_END_ID} if tools else None
    ids = list(generate_ids(bundle, messages, p, tools=tools, extra_stop_ids=extra_stop))
    completion_tokens = len(ids)

    if tools and TOOL_CALL_START_ID in ids:
        s = ids.index(TOOL_CALL_START_ID)
        try:
            e = ids.index(TOOL_CALL_END_ID, s + 1)
        except ValueError:
            e = len(ids)
        inner = tokenizer.decode(ids[s + 1:e], skip_special_tokens=True)
        after = tokenizer.decode(ids[e + 1:], skip_special_tokens=True).strip()
        tool_calls = _parse_tool_calls(inner)
        if tool_calls:
            return {
                "content": after or None,
                "tool_calls": tool_calls,
                "finish_reason": "tool_calls",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return {
        "content": text,
        "tool_calls": None,
        "finish_reason": "length" if completion_tokens >= p.max_tokens else "stop",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
