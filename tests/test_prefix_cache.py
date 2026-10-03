"""Prompt-prefix caching: the LRU and where prefill is split (no model), plus an
integration check that cached and uncached generation agree (pytest -m integration)."""

import dataclasses

import pytest

from onnx_lfm_api.generate import GenParams, generate_text
from onnx_lfm_api.prefix_cache import PrefixCache, prefill_cuts

END = 7  # stands in for <|im_end|>


def test_lookup_returns_longest_strict_prefix():
    pc = PrefixCache(4)
    pc.store([1, 2], {"s": "short"})
    pc.store([1, 2, 3, 4], {"s": "long"})
    pc.store([9, 9], {"s": "other"})
    assert pc.lookup([1, 2, 3, 4, 5]) == (4, {"s": "long"})
    assert pc.lookup([1, 2, 3]) == (2, {"s": "short"})
    assert pc.lookup([1, 2, 3, 4]) == (2, {"s": "short"})  # strict: something must be left to run
    assert pc.lookup([5]) == (0, None)
    assert pc.stats() == {"size": 4, "entries": 3, "hits": 3, "misses": 1, "reused_tokens": 8}


def test_lru_evicts_least_recently_used():
    pc = PrefixCache(2)
    pc.store([1], {"s": 1})
    pc.store([2], {"s": 2})
    pc.lookup([1, 0])  # touch [1]
    pc.store([3], {"s": 3})  # evicts [2]
    assert pc.lookup([2, 0]) == (0, None)
    assert pc.lookup([1, 0])[0] == 1 and pc.lookup([3, 0])[0] == 1


def test_returned_state_is_a_copy():
    pc = PrefixCache(1)
    pc.store([1], {"s": 1})
    pc.lookup([1, 2])[1]["s"] = "changed"
    assert pc.lookup([1, 2])[1] == {"s": 1}


@pytest.mark.parametrize("ids, start, expected", [
    ([1, END, 2, END, 3, END, 4], 0, [2, 6, 7]),  # first + last boundary, then the end
    ([1, END, 2, END, 3, END, 4], 2, [6, 7]),     # resumed: only the last boundary
    ([1, END, 2], 0, [2, 3]),                      # one boundary
    ([1, 2, 3], 0, [3]),                           # none
    ([1, 2, END], 0, [3]),                         # a boundary at the very end is just the end
])
def test_prefill_cuts(ids, start, expected):
    assert prefill_cuts(ids, start, END) == expected


def test_prefill_cuts_without_boundary_token():
    assert prefill_cuts([1, END, 2], 0, None) == [3]


@pytest.fixture(scope="module")
def bundle():
    from onnx_lfm_api.model import load_model
    return load_model()


@pytest.mark.integration
def test_cached_generation_matches_uncached(bundle):
    system = {"role": "system", "content": "You are a terse assistant. " * 20}
    convs = [[system, {"role": "user", "content": q}] for q in
             ("Name a primary color.", "What is 2 + 2?", "Name a primary color.")]
    greedy = GenParams(max_tokens=24, temperature=0.0)
    plain = dataclasses.replace(bundle, prefix_cache=None)
    cached = dataclasses.replace(bundle, prefix_cache=PrefixCache(4))
    for msgs in convs:
        assert generate_text(cached, msgs, greedy) == generate_text(plain, msgs, greedy)
    stats = cached.prefix_cache.stats()
    assert stats["hits"] == 2 and stats["reused_tokens"] > 100  # the shared system message
