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

import asyncio
import inspect
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


async def _filter_children_any(
    gear_gate: Any,
    node: Dict[str, Any],
    depth: int,
    default_bf: int,
    children: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prefer the gate's async filter (awaits async scorers); fall back to sync."""

    async_filter = getattr(gear_gate, "filter_children_async", None)
    if async_filter is not None:
        return await async_filter(node, depth, default_bf, children)
    result = gear_gate.filter_children(node, depth, default_bf, children)
    if inspect.isawaitable(result):
        return await result
    return result


async def async_build_tree(
    root_prompt_text: str,
    root_prompt_token_ids: Sequence[int],
    data_instance: Dict[str, Any],
    *,
    tree_shape: Sequence[int],
    M: int,
    segment_fn,  # async: (prompt_token_ids, branch_factor, max_tokens) -> List[SegmentSample]
    grade_fn: GradeFn,
    max_depth: Optional[int] = None,
    gear_gate: Optional[Any] = None,
    gear_node_expander: Optional[Any] = None,
) -> Dict[str, Any]:
    """Async mirror of :func:`build_tree` for verl's async (agent-loop) rollout.

    Identical topology / reward / segmentation logic; the only difference is that
    ``segment_fn`` is awaited (it wraps ``AsyncLLMServerManager.generate``). Kept
    byte-for-byte in step with the sync version — any change to the tree math must
    be mirrored in both.

    ``gear_node_expander`` enables the VDRA depth-batched budget allocation:
    when the gate reports ``use_batch_allocation`` and an expander is provided,
    the tree is built level-by-level via :func:`async_build_tree_batch_alloc`
    so sibling frontier nodes share one rollout budget (Summary.md §10-§11).
    """
    if (
        gear_gate is not None
        and gear_node_expander is not None
        and getattr(gear_gate, "use_batch_allocation", False)
    ):
        return await async_build_tree_batch_alloc(
            root_prompt_text,
            root_prompt_token_ids,
            data_instance,
            tree_shape=tree_shape,
            M=M,
            segment_fn=segment_fn,
            grade_fn=grade_fn,
            max_depth=max_depth,
            gear_gate=gear_gate,
            gear_node_expander=gear_node_expander,
        )

    t0 = time.time()
    if max_depth is None:
        max_depth = len(tree_shape)
    if gear_gate is None:
        gear_gate = _NoopGate()

    tree: Dict[str, Any] = {
        "text": root_prompt_text, "depth": 0, "full_text": root_prompt_text,
        "stop_text": "aaa", "_request_object": data_instance, "leaf": False,
        "full_token_ids": list(root_prompt_token_ids),
    }

    async def dfs(node: Dict[str, Any], prefix: str, depth: int) -> None:
        if depth == max_depth:
            node["reward"] = float(grade_fn(prefix, node["text"], data_instance))
            node["leaf"] = True
            return
        max_tokens = None if depth == max_depth - 1 else M
        default_bf = tree_shape[depth] if depth < len(tree_shape) else tree_shape[-1]
        branch_factor = gear_gate.branch_factor(node, depth, default_bf)
        if inspect.isawaitable(branch_factor):
            branch_factor = await branch_factor
        branch_factor = int(branch_factor)
        samples = await segment_fn(node["full_token_ids"], branch_factor, max_tokens)

        children: List[Dict[str, Any]] = []
        for s in samples:
            child = {
                "text": s.text, "depth": depth + 1, "full_text": prefix + s.text,
                "stop_text": None, "finish_reason": s.finish_reason,
                "response_token_ids": list(s.token_ids),
                "actor_shifted_log_probs": list(s.logprobs) if s.logprobs is not None else None,
                "full_token_ids": list(node["full_token_ids"]) + list(s.token_ids),
            }
            if s.sum_logprobs is not None:
                child["sum_logprobs"] = float(s.sum_logprobs)
                child["num_tokens"] = int(s.num_tokens)
            children.append(child)

        children = await _filter_children_any(gear_gate, node, depth, default_bf, children)
        node["children"] = children
        for child in children:
            if child["finish_reason"] != "length":
                child["reward"] = float(grade_fn(prefix, child["full_text"], data_instance))
                child["leaf"] = True
            else:
                child["leaf"] = False
                await dfs(child, child["full_text"], depth + 1)

        child_rewards = [child["reward"] for child in children]
        node["reward"] = float(np.mean(child_rewards))
        node["reward_std"] = float(np.std(child_rewards))

    await dfs(tree, root_prompt_text, 0)
    tree["tree_construction_seconds"] = time.time() - t0
    return tree


def _retained_pilot_samples(node: Dict[str, Any], allocated_k: int) -> List[SegmentSample]:
    samples: List[SegmentSample] = []
    for candidate in list(node.get("vdra_pilot_children") or [])[:allocated_k]:
        token_ids = candidate.get("response_token_ids")
        if token_ids is None:
            parent_ids = list(node.get("full_token_ids") or [])
            full_ids = list(candidate.get("full_token_ids") or [])
            token_ids = full_ids[len(parent_ids):] if full_ids[:len(parent_ids)] == parent_ids else []
        if not token_ids:
            continue
        samples.append(
            SegmentSample(
                token_ids=list(token_ids),
                text=str(candidate.get("text", "")),
                finish_reason=str(candidate.get("finish_reason", "length")),
                logprobs=(
                    list(candidate["actor_shifted_log_probs"])
                    if candidate.get("actor_shifted_log_probs") is not None
                    else None
                ),
                sum_logprobs=candidate.get("sum_logprobs"),
                num_tokens=candidate.get("num_tokens"),
            )
        )
    return samples


async def _expand_reusing_pilots(
    node: Dict[str, Any],
    allocated_k: int,
    max_tokens: Optional[int],
    segment_fn,
) -> List[SegmentSample]:
    retained = _retained_pilot_samples(node, allocated_k)
    missing = max(int(allocated_k) - len(retained), 0)
    additional = (
        await segment_fn(node["full_token_ids"], missing, max_tokens)
        if missing
        else []
    )
    generated = len(node.get("vdra_pilot_children") or [])
    node["vdra_pilot_children_generated"] = generated
    node["vdra_pilot_generated_tokens"] = sum(
        len(candidate.get("response_token_ids") or [])
        for candidate in node.get("vdra_pilot_children") or []
    )
    node["vdra_pilot_children_reused"] = len(retained)
    node["vdra_pilot_children_discarded"] = max(generated - len(retained), 0)
    node["vdra_additional_children_generated"] = len(additional)
    node["vdra_main_expansion_generated_tokens"] = sum(
        len(sample.token_ids) for sample in additional
    )
    node["vdra_total_generated_tokens"] = (
        node["vdra_pilot_generated_tokens"]
        + node["vdra_main_expansion_generated_tokens"]
    )
    node["vdra_pilot_reuse_rate"] = len(retained) / generated if generated else 0.0
    return retained + list(additional)

async def async_build_tree_batch_alloc(
    root_prompt_text: str,
    root_prompt_token_ids: Sequence[int],
    data_instance: Dict[str, Any],
    *,
    tree_shape: Sequence[int],
    M: int,
    segment_fn,
    grade_fn: GradeFn,
    max_depth: Optional[int] = None,
    gear_gate: Any,
    gear_node_expander: Any,
) -> Dict[str, Any]:
    """Level-synchronous tree builder with VDRA budget allocation.

    Same node schema, segmentation rule, leaf handling and reward back-prop as
    :func:`async_build_tree`; the only difference is expansion order: each depth
    expands the whole frontier at once so ``gear_gate.allocate_batch_async`` can
    reallocate one shared budget (``sum(default_bf)``) across sibling frontier
    nodes proportionally to their value-dispersion bound (Summary.md §10-§11).
    A node allocated ``k = 0`` branches (possible when ``n_min = 0``) becomes a
    graded leaf instead of expanding.
    """

    t0 = time.time()
    if max_depth is None:
        max_depth = len(tree_shape)

    tree: Dict[str, Any] = {
        "text": root_prompt_text, "depth": 0, "full_text": root_prompt_text,
        "stop_text": "aaa", "_request_object": data_instance, "leaf": False,
        "full_token_ids": list(root_prompt_token_ids),
    }

    frontier: List[Dict[str, Any]] = [tree]
    for depth in range(max_depth):
        if not frontier:
            break
        max_tokens = None if depth == max_depth - 1 else M
        default_bf = tree_shape[depth] if depth < len(tree_shape) else tree_shape[-1]

        # Depth 0 (single root) and the near-leaf level keep the uniform width;
        # other levels reallocate the shared depth budget.
        near_leaf = (
            getattr(gear_gate, "skip_near_leaf_expand", False)
            and depth == max_depth - 1
        )
        if depth > 0 and not near_leaf:
            await gear_gate.allocate_batch_async(
                frontier, depth, default_bf, gear_node_expander
            )

        expand_nodes: List[Dict[str, Any]] = []
        branch_factors: List[int] = []
        for node in frontier:
            bf = int(gear_gate.branch_factor(node, depth, default_bf))
            if bf <= 0:
                # Pruned to zero: grade the node as a truncated leaf (same
                # grading rule as the depth == max_depth case in the DFS).
                node["reward"] = float(
                    grade_fn(node["full_text"], node["text"], data_instance)
                )
                node["leaf"] = True
                continue
            expand_nodes.append(node)
            branch_factors.append(bf)

        sample_batches = await asyncio.gather(
            *[
                _expand_reusing_pilots(node, bf, max_tokens, segment_fn)
                for node, bf in zip(expand_nodes, branch_factors)
            ]
        ) if expand_nodes else []

        next_frontier: List[Dict[str, Any]] = []
        for node, samples in zip(expand_nodes, sample_batches):
            prefix = node["full_text"]
            children: List[Dict[str, Any]] = []
            for s in samples:
                child = {
                    "text": s.text, "depth": depth + 1, "full_text": prefix + s.text,
                    "stop_text": None, "finish_reason": s.finish_reason,
                    "response_token_ids": list(s.token_ids),
                    "actor_shifted_log_probs": list(s.logprobs) if s.logprobs is not None else None,
                    "full_token_ids": list(node["full_token_ids"]) + list(s.token_ids),
                }
                if s.sum_logprobs is not None:
                    child["sum_logprobs"] = float(s.sum_logprobs)
                    child["num_tokens"] = int(s.num_tokens)
                children.append(child)

            children = await _filter_children_any(
                gear_gate, node, depth, default_bf, children
            )
            node["children"] = children
            for child in children:
                if child["finish_reason"] != "length":
                    child["reward"] = float(
                        grade_fn(prefix, child["full_text"], data_instance)
                    )
                    child["leaf"] = True
                else:
                    child["leaf"] = False
                    next_frontier.append(child)
        frontier = next_frontier

    # Nodes still on the frontier reached max_depth: grade the raw segment text
    # (identical to the ``depth == max_depth`` branch of the DFS builder).
    for node in frontier:
        node["reward"] = float(grade_fn(node["full_text"], node["text"], data_instance))
        node["leaf"] = True

    def backprop(node: Dict[str, Any]) -> None:
        children = node.get("children") or []
        if not children:
            return
        for child in children:
            backprop(child)
        child_rewards = [child["reward"] for child in children]
        node["reward"] = float(np.mean(child_rewards))
        node["reward_std"] = float(np.std(child_rewards))

    backprop(tree)
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
