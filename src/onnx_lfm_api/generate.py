from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import settings
from .model import ModelBundle

# Map ORT input dtype strings to numpy dtypes for building the empty cache.
_DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
}

_CORE_INPUTS = {"input_ids", "attention_mask", "position_ids"}


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


def generate_ids(bundle: ModelBundle, messages: list[dict], p: GenParams) -> Iterator[int]:
    """Yield generated token ids one at a time (stop tokens are not yielded).

    This is the heart of the service — Liquid's reference decode loop, which
    threads both the attention KV cache and the convolution state back in on
    every step via the ``present* -> past*`` remap.
    """
    session = bundle.session
    tokenizer = bundle.tokenizer

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = np.array(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64
    )
    seq_len = input_ids.shape[1]

    cache = _init_cache(session)
    generated: list[int] = []

    for step in range(p.max_tokens):
        if step == 0:
            ids = input_ids
            pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        else:
            ids = np.array([[generated[-1]]], dtype=np.int64)
            pos = np.array([[seq_len + len(generated) - 1]], dtype=np.int64)

        attn = np.ones((1, seq_len + len(generated)), dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": attn, **cache}
        if "position_ids" in bundle.input_names:
            feed["position_ids"] = pos

        outputs = session.run(None, feed)
        next_token = _sample(outputs[0][0, -1], generated, p)
        generated.append(next_token)

        # Feed each present* state back as the matching past* input next step.
        for i, out in enumerate(session.get_outputs()[1:], 1):
            name = out.name.replace("present_conv", "past_conv")
            name = name.replace("present.", "past_key_values.")
            if name in cache:
                cache[name] = outputs[i]

        if next_token in bundle.stop_ids:
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
