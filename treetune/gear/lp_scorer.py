"""Score log pi(y_i | traj(s)) for the GEAR LP matrix.

Strategy: send `traj(s) + y_i` to the vLLM completions endpoint with
`max_tokens=0`, `prompt_logprobs=1`, then sum the per-token logprobs of the
y_i suffix.  We slice by tokenizing the prefix and the suffix separately so we
know how many tokens to take from the tail of the returned list.

The scorer is async and supports batching via gather.  It also caches results
keyed by (segment_id, y_index) to avoid recomputing fast indices when the
slow tail is requested.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Sequence, Tuple


@dataclass
class LPScorer:
    """Compute prefix-conditioned logprob of `y_i` strings.

    `score_fn(prompt, max_tokens=0, prompt_logprobs=1)` must return a list of
    floats: token-level logprobs for the entire prompt (length == #prompt
    tokens).  In SPO this is the OpenAIVLLM completions endpoint with
    `prompt_logprobs=1`.

    `tokenize_fn(text) -> List[int]` is used to find the boundary between the
    `traj(s)` prefix and the `y_i` suffix tokens.
    """

    score_fn: Callable[..., Awaitable[List[float]]]
    tokenize_fn: Callable[[str], List[int]]
    cache: Dict[Tuple[str, int], float] = field(default_factory=dict)

    async def score_one(self, prefix: str, y_text: str) -> float:
        """Return sum of log pi(y_text | prefix) per token."""

        full = prefix + y_text
        prefix_tokens = self.tokenize_fn(prefix)
        full_tokens = self.tokenize_fn(full)
        suffix_len = len(full_tokens) - len(prefix_tokens)
        if suffix_len <= 0:
            return 0.0

        prompt_logprobs = await self.score_fn(prompt=full)
        if not prompt_logprobs:
            return -math.inf

        # vLLM returns one logprob per prompt token; take the tail of length
        # suffix_len. The first token of any prompt has logprob None, but it
        # belongs to the prefix so it does not affect the suffix slice.
        tail = prompt_logprobs[-suffix_len:]
        clean = [lp for lp in tail if lp is not None]
        if not clean:
            return -math.inf
        return float(sum(clean))

    async def score_batch(
        self,
        segment_id: str,
        prefix: str,
        y: Sequence[str],
        indices: Sequence[int],
    ) -> List[float]:
        """Score a list of y indices.  Cached by (segment_id, index)."""

        results: List[float] = [0.0] * len(indices)
        pending: List[Tuple[int, asyncio.Task]] = []
        for slot, idx in enumerate(indices):
            key = (segment_id, idx)
            if key in self.cache:
                results[slot] = self.cache[key]
                continue
            task = asyncio.create_task(self.score_one(prefix, y[idx]))
            pending.append((slot, task))

        if pending:
            done = await asyncio.gather(*(t for _, t in pending))
            for (slot, _), val in zip(pending, done):
                idx = indices[slot]
                self.cache[(segment_id, idx)] = val
                results[slot] = val
        return results
