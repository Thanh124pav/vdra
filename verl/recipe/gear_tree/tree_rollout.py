"""Native segment-tree rollout for the GEAR/Tree recipe (Step 2).

Ports treetune's ``HybridInferenceStrategy._construct_tree``
(``treetune/inference_strategies/hybrid_inference_strategy.py:359-453``) to a
verl-native, engine-agnostic builder. The tree topology, segmentation rule
(``max_tokens = None if depth == max_depth-1 else M``), leaf-vs-expand decision
(``finish_reason != "length"``), and reward back-prop
(``node.reward = mean(child_rewards)``, ``node.reward_std = std(...)``) are
byte-for-byte identical to treetune.

Design: the tree math is decoupled from the generation engine through the
``segment_fn`` / ``grade_fn`` callables, so it is fully CPU-testable with mock
generators (golden-numerics parity vs treetune). ``VLLMTreeRollout`` binds
``segment_fn`` to verl's synchronous vLLM SPMD engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# A generated segment sample from the engine.
@dataclass
class SegmentSample:
    token_ids: List[int]
    text: str
    finish_reason: str  # "length" => truncated => expandable
    logprobs: Optional[List[float]] = None  # per-token chosen-token logprob
    sum_logprobs: Optional[float] = None
    num_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_tokens is None:
            self.num_tokens = len(self.token_ids)
        if self.sum_logprobs is None and self.logprobs is not None:
            self.sum_logprobs = float(sum(self.logprobs))


# segment_fn(prompt_token_ids, branch_factor, max_tokens) -> list of samples.
SegmentFn = Callable[[Sequence[int], int, Optional[int]], List[SegmentSample]]
# grade_fn(query_text, response_text, data_instance) -> reward float.
GradeFn = Callable[[str, str, Dict[str, Any]], float]

# Optional GEAR gate object. Two hooks around each expansion:
#   * ``branch_factor(parent, depth, default_bf) -> int`` decides how many
#     children to generate (online prune / budget allocation).
#   * ``filter_children(parent, depth, default_bf, children) -> children``
#     annotates / drops children after generation (share / prune).
# The no-op default keeps SPO / TreeRL / TreePO behaviour byte-identical.
class _NoopGate:
    def branch_factor(self, parent, depth, default_bf):  # noqa: ANN001
        return default_bf

    def filter_children(self, parent, depth, default_bf, children):  # noqa: ANN001
        return children


def build_tree(
    root_prompt_text: str,
    root_prompt_token_ids: Sequence[int],
    data_instance: Dict[str, Any],
    *,
    tree_shape: Sequence[int],
    M: int,
    segment_fn: SegmentFn,
    grade_fn: GradeFn,
    max_depth: Optional[int] = None,
    gear_gate: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build one SPO/GEAR segment tree for a single prompt.

    Faithful to ``_construct_tree``:
      * root at depth 0 with ``full_text == prompt``;
      * at each internal depth, expand ``branch_factor = tree_shape[depth]``
        children with ``max_tokens = None if depth == max_depth-1 else M``;
      * a child with ``finish_reason != "length"`` is a graded leaf, otherwise it
        is expanded recursively;
      * ``node.reward = mean(child rewards)``, ``node.reward_std = std(...)``.
    """
    t0 = time.time()
    if max_depth is None:
        max_depth = len(tree_shape)
    if gear_gate is None:
        gear_gate = _NoopGate()

    tree: Dict[str, Any] = {
        "text": root_prompt_text,
        "depth": 0,
        "full_text": root_prompt_text,
        "stop_text": "aaa",  # not used (matches treetune)
        "_request_object": data_instance,
        "leaf": False,
        "full_token_ids": list(root_prompt_token_ids),
    }

    def dfs(node: Dict[str, Any], prefix: str, depth: int) -> None:
        if depth == max_depth:
            # Truncated past the tree depth: grade the raw segment text.
            node["reward"] = float(grade_fn(prefix, node["text"], data_instance))
            node["leaf"] = True
            return

        max_tokens = None if depth == max_depth - 1 else M
        default_bf = tree_shape[depth] if depth < len(tree_shape) else tree_shape[-1]
        # GEAR online prune / budget allocation may shrink the branch factor.
        branch_factor = int(gear_gate.branch_factor(node, depth, default_bf))

        samples = segment_fn(node["full_token_ids"], branch_factor, max_tokens)

        children: List[Dict[str, Any]] = []
        for s in samples:
            child = {
                "text": s.text,
                "depth": depth + 1,
                "full_text": prefix + s.text,
                "stop_text": None,
                "finish_reason": s.finish_reason,
                "response_token_ids": list(s.token_ids),
                "actor_shifted_log_probs": list(s.logprobs) if s.logprobs is not None else None,
                "full_token_ids": list(node["full_token_ids"]) + list(s.token_ids),
            }
            if s.sum_logprobs is not None:
                child["sum_logprobs"] = float(s.sum_logprobs)
                child["num_tokens"] = int(s.num_tokens)
            children.append(child)

        # Optional GEAR online gate (prune/share/budget). No-op for SPO family.
        children = gear_gate.filter_children(node, depth, default_bf, children)
        node["children"] = children

        for child in children:
            if child["finish_reason"] != "length":
                child["reward"] = float(
                    grade_fn(prefix, child["full_text"], data_instance)
                )
                child["leaf"] = True
            else:
                child["leaf"] = False
                dfs(child, child["full_text"], depth + 1)

        child_rewards = [child["reward"] for child in children]
        node["reward"] = float(np.mean(child_rewards))
        node["reward_std"] = float(np.std(child_rewards))

    dfs(tree, root_prompt_text, 0)
    tree["tree_construction_seconds"] = time.time() - t0
    return tree


def strip_internal_fields(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the internal ``full_token_ids`` accumulator (kept out of edges)."""

    def visit(node: Dict[str, Any]) -> None:
        node.pop("full_token_ids", None)
        for child in node.get("children", []) or []:
            visit(child)

    visit(tree)
    return tree


# --------------------------------------------------------------------------- #
# vLLM binding (GPU path). Imported lazily so the module stays CPU-importable.
# --------------------------------------------------------------------------- #
@dataclass
class VLLMTreeRollout:
    """Bind ``segment_fn`` to a verl vLLM SPMD ``inference_engine``.

    ``inference_engine`` is the ``vllm.LLM`` held by
    ``verl.workers.rollout.vllm_rollout.vllm_rollout_spmd.vLLMRollout``. We drive
    it segment-by-segment: each expansion is one ``engine.generate`` call with
    ``n = branch_factor`` and ``max_tokens = M`` (or the free budget at the last
    internal depth), requesting per-token logprobs for the GEAR gate.
    """

    inference_engine: Any
    tokenizer: Any
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    free_max_tokens: int = 1024  # budget used when max_tokens is None
    logprobs: int = 1
    seed: Optional[int] = None

    def _sampling_params(self, n: int, max_tokens: Optional[int]):
        from vllm import SamplingParams  # lazy import (GPU env only)

        return SamplingParams(
            n=n,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.free_max_tokens,
            logprobs=self.logprobs,
            seed=self.seed,
        )

    def segment_fn(
        self, prompt_token_ids: Sequence[int], branch_factor: int, max_tokens: Optional[int]
    ) -> List[SegmentSample]:
        sp = self._sampling_params(branch_factor, max_tokens)
        outputs = self.inference_engine.generate(
            prompts=[{"prompt_token_ids": list(prompt_token_ids)}],
            sampling_params=sp,
            use_tqdm=False,
        )
        samples: List[SegmentSample] = []
        for completion in outputs[0].outputs:
            token_ids = list(completion.token_ids)
            per_tok_logprobs: Optional[List[float]] = None
            if completion.logprobs is not None:
                per_tok_logprobs = [
                    completion.logprobs[i][tid].logprob
                    for i, tid in enumerate(token_ids)
                ]
            samples.append(
                SegmentSample(
                    token_ids=token_ids,
                    text=completion.text,
                    finish_reason=completion.finish_reason or "stop",
                    logprobs=per_tok_logprobs,
                )
            )
        return samples
