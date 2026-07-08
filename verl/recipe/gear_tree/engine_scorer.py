"""Offline-engine log-prob scorer for the GEAR share / TV-budget paths.

Mirrors the vendored ``gear_core.gear.lp_scorer.LPScorer.score_one`` (same
prefix/suffix boundary logic, same tail-sum of ``log pi(y|prefix)``) but runs
**synchronously** against verl's offline vLLM ``LLM`` engine using
``SamplingParams(prompt_logprobs=...)`` — the offline equivalent of the
``/completions`` ``echo=True, logprobs=1`` request the HTTP ``VLLMLogprobClient``
uses. This lets the GEAR gate score answer/continuation sets on the same engine
that generated the tree, with no separate HTTP server.

CPU-testable with a fake engine that returns ``prompt_logprobs``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class EngineLPScorer:
    engine: Any  # vllm.LLM (offline) or a fake with .generate
    tokenizer: Any
    cache: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def _prompt_token_logprobs(self, full_ids: Sequence[int]) -> List[Any]:
        from vllm import SamplingParams  # lazy import (GPU env)

        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        out = self.engine.generate(
            [{"prompt_token_ids": list(full_ids)}], sp, use_tqdm=False
        )
        plp = out[0].prompt_logprobs  # list; entry per prompt token, [0] is None
        res: List[Any] = []
        for i, d in enumerate(plp):
            if d is None:
                res.append(None)
            else:
                tid = full_ids[i]
                res.append(d[tid].logprob if tid in d else None)
        return res

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def score_one(self, prefix: str, y_text: str) -> float:
        """Sum of ``log pi(y_text | prefix)`` per token (LPScorer.score_one parity)."""
        key = (prefix, y_text)
        if key in self.cache:
            return self.cache[key]

        full = prefix + y_text
        prefix_tokens = self._encode(prefix)
        full_tokens = self._encode(full)
        suffix_len = len(full_tokens) - len(prefix_tokens)
        if suffix_len <= 0:
            self.cache[key] = 0.0
            return 0.0

        plp = self._prompt_token_logprobs(full_tokens)
        if not plp:
            return -math.inf
        tail = plp[-suffix_len:]
        clean = [lp for lp in tail if lp is not None]
        val = float(sum(clean)) if clean else -math.inf
        self.cache[key] = val
        return val
