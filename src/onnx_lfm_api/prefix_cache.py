"""Reuse the model state of a prompt prefix seen before (prompt-prefix caching).

Agent calls share a long prefix: the system message with the tool schemas, and
in multi-round runs the whole conversation so far. Re-reading it dominates a
tool-calling round, so generate_ids() snapshots the state at message boundaries
and later prompts that start with the same tokens resume from there.

LFM2's conv-block state can't be cut back to an earlier position the way the
attention KV cache can, so snapshots are only taken at points the prefill
actually stops at, never derived from a longer state.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class PrefixCache:
    """LRU of {prompt-token prefix: past-state feed dict}.

    Entries hold whatever generate_ids() keeps as the cache (numpy arrays, or
    device OrtValues under IO binding). They are never mutated in place: every
    model step returns fresh outputs.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[tuple[int, ...], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0

    def lookup(self, ids: list[int]) -> tuple[int, dict[str, Any] | None]:
        """(length, state) of the longest stored strict prefix of `ids`, else (0, None)."""
        with self._lock:
            best: tuple[int, ...] | None = None
            for key in self._items:
                if (len(key) < len(ids) and (best is None or len(key) > len(best))
                        and tuple(ids[:len(key)]) == key):
                    best = key
            if best is None:
                self.misses += 1
                return 0, None
            self._items.move_to_end(best)
            self.hits += 1
            self.reused_tokens += len(best)
            return len(best), dict(self._items[best])

    def store(self, ids: list[int], state: dict[str, Any]) -> None:
        if self.size <= 0:
            return
        key = tuple(ids)
        with self._lock:
            self._items[key] = dict(state)
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"size": self.size, "entries": len(self._items), "hits": self.hits,
                    "misses": self.misses, "reused_tokens": self.reused_tokens}


def prefill_cuts(ids: list[int], start: int, boundary_id: int | None) -> list[int]:
    """Where to split the prefill of ids[start:]: the last message boundary
    (just after a ``<|im_end|>``) past `start`, then the end. From scratch
    (start 0) also the first boundary.

    The first boundary is usually the system message + tools, shared by every
    call; the last is the conversation so far, which the next round extends.
    After a resume the first boundary is just mid-conversation: snapshotting it
    would only evict the shared system-message snapshot from the LRU.
    """
    bounds = [i + 1 for i, t in enumerate(ids)
              if t == boundary_id and start < i + 1 < len(ids)] if boundary_id is not None else []
    first = bounds[:1] if start == 0 else []
    return sorted({*first, *bounds[-1:], len(ids)})
