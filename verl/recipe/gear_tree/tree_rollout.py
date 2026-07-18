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
import itertools
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# PLAN.md P0.2: process-local monotonic counter used as the per-tree tiebreaker
# when the rollout does not supply an explicit tree_instance_uuid. Two trees
# built for the same (question_id, policy_snapshot_id, rollout_iteration) get
# distinct IDs because their counter values differ.
_TREE_INSTANCE_COUNTER = itertools.count(0)
_TREE_INSTANCE_LOCK = Lock()


def make_tree_instance_id(
    *,
    policy_snapshot_id: Any,
    rollout_iteration: Any,
    stable_question_id: Any,
    tree_instance_uuid: Optional[str] = None,
) -> str:
    """PLAN.md P0.2: build a globally-unique tree_instance_id.

    Every stochastic tree gets one instance id combining:
      * the policy snapshot the rollout used;
      * the rollout iteration inside that snapshot;
      * a stable question id from the dataset;
      * a per-tree tiebreaker (explicit uuid, or a monotonic counter).

    Two trees for the same (question, snapshot, iteration) tuple are guaranteed
    to hold different ids so their edges coexist in replay without collision.
    """
    if tree_instance_uuid:
        tiebreaker = str(tree_instance_uuid)
    else:
        with _TREE_INSTANCE_LOCK:
            tiebreaker = f"c{next(_TREE_INSTANCE_COUNTER)}-{uuid.uuid4().hex[:8]}"
    parts = [
        str(policy_snapshot_id or "snapshot:unknown"),
        f"iter:{rollout_iteration if rollout_iteration is not None else 0}",
        f"q:{stable_question_id if stable_question_id is not None else 'na'}",
        tiebreaker,
    ]
    return "|".join(parts)

# A generated segment sample from the engine.
@dataclass
class SegmentSample:
    token_ids: List[int]
    text: str
    finish_reason: str  # "length" => truncated => expandable
    logprobs: Optional[List[float]] = None  # per-token chosen-token logprob
    sum_logprobs: Optional[float] = None
    num_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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

    # PLAN.md P0.2: stamp a globally-unique tree_instance_id at construction
    # time so every downstream stage (edges, replay, tensorization, manifest)
    # can safely key by tree_id without merging repeated rollouts.
    tree_instance_id = make_tree_instance_id(
        policy_snapshot_id=data_instance.get("policy_snapshot_id")
        or data_instance.get("current_rollout_snapshot_id"),
        rollout_iteration=data_instance.get("rollout_iteration"),
        stable_question_id=data_instance.get("_treetune__idx")
        or data_instance.get("uid"),
        tree_instance_uuid=data_instance.get("tree_instance_uuid"),
    )

    tree: Dict[str, Any] = {
        "text": root_prompt_text,
        "depth": 0,
        "full_text": root_prompt_text,
        "stop_text": "aaa",  # not used (matches treetune)
        "_request_object": data_instance,
        "leaf": False,
        "full_token_ids": list(root_prompt_token_ids),
        "tree_id": tree_instance_id,
        "tree_instance_id": tree_instance_id,
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

        _aggregate_child_rewards(node, children)

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
    free_max_tokens: int = 1024,
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
        and getattr(gear_gate, "use_online_allocation", False)
    ):
        return await async_build_tree_online_alloc(
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
            free_max_tokens=free_max_tokens,
        )

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

    # PLAN.md P0.2: same globally-unique tree id as the sync builder.
    tree_instance_id = make_tree_instance_id(
        policy_snapshot_id=data_instance.get("policy_snapshot_id")
        or data_instance.get("current_rollout_snapshot_id"),
        rollout_iteration=data_instance.get("rollout_iteration"),
        stable_question_id=data_instance.get("_treetune__idx")
        or data_instance.get("uid"),
        tree_instance_uuid=data_instance.get("tree_instance_uuid"),
    )

    tree: Dict[str, Any] = {
        "text": root_prompt_text, "depth": 0, "full_text": root_prompt_text,
        "stop_text": "aaa", "_request_object": data_instance, "leaf": False,
        "full_token_ids": list(root_prompt_token_ids),
        "tree_id": tree_instance_id,
        "tree_instance_id": tree_instance_id,
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

        _aggregate_child_rewards(node, children)

    await dfs(tree, root_prompt_text, 0)
    tree["tree_construction_seconds"] = time.time() - t0
    return tree


def _uniform_generated_token_cap(
    tree_shape: Sequence[int], M: int, max_depth: int, free_max_tokens: int
) -> int:
    """Expected generated tokens of the uniform SPO tree with the same shape.

    This is the ``uniform_full_tree_token_cap`` budget cap: pilot, support and
    main expansion generation must all fit under the maximum-style token count
    a full uniform tree would have spent on main expansion alone.
    """

    total = 0
    width = 1
    for depth in range(max_depth):
        width *= int(tree_shape[depth] if depth < len(tree_shape) else tree_shape[-1])
        per_node = int(free_max_tokens) if depth == max_depth - 1 else int(M)
        total += width * per_node
    return total


def _aggregate_child_rewards(node: Dict[str, Any], children: List[Dict[str, Any]]) -> None:
    child_rewards = [float(child["reward"]) for child in children]
    if not child_rewards:
        node["reward"] = 0.0
        node["reward_std"] = 0.0
        return
    weights = []
    has_weights = False
    for child in children:
        maybe = child.get("edge_weight", child.get("vdra_representative_weight"))
        if maybe is None:
            weights.append(1.0)
            continue
        weight = float(maybe)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"invalid weighted-reuse child weight: {maybe!r}")
        weights.append(weight)
        has_weights = True
    if not has_weights:
        node["reward"] = float(np.mean(child_rewards))
        node["reward_std"] = float(np.std(child_rewards))
        return
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        raise ValueError("weighted-reuse child weights sum to zero")
    reward = sum(w * r for w, r in zip(weights, child_rewards)) / total_weight
    variance = sum(w * (r - reward) ** 2 for w, r in zip(weights, child_rewards)) / total_weight
    node["reward"] = float(reward)
    node["reward_std"] = float(np.sqrt(max(variance, 0.0)))
    node["vdra_weighted_parent_value"] = float(reward)
    node["vdra_child_weight_sum"] = total_weight


async def _segment_fn_call(
    segment_fn,
    prompt_ids: Sequence[int],
    branch_factor: int,
    max_tokens: Optional[int],
    *,
    sibling_sticky_keys: Optional[Sequence[str]] = None,
) -> List[SegmentSample]:
    """Call ``segment_fn`` and forward ``sibling_sticky_keys`` when accepted.

    P0.1: production segment functions (AsyncServerSegmentGenerator.segment_fn)
    forward the sticky key to the server manager for prefix-cache-aware
    routing. Test doubles that only accept positional args continue to work
    via the ``TypeError`` fallback.
    """

    if sibling_sticky_keys is not None:
        try:
            return await segment_fn(
                prompt_ids, int(branch_factor), max_tokens,
                sibling_sticky_keys=list(sibling_sticky_keys),
            )
        except TypeError:
            pass
    return await segment_fn(prompt_ids, int(branch_factor), max_tokens)


def _candidate_to_sample(node: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[SegmentSample]:
    token_ids = candidate.get("response_token_ids")
    if token_ids is None:
        parent_ids = list(node.get("full_token_ids") or [])
        full_ids = list(candidate.get("full_token_ids") or [])
        token_ids = full_ids[len(parent_ids):] if full_ids[:len(parent_ids)] == parent_ids else []
    if not token_ids:
        return None
    metadata = dict(candidate.get("vdra_sample_metadata") or {})
    # P0.W1: cluster metadata is written on the candidate node by the TV
    # estimator (see gear_core/gear/tv_estimators.py). Copy the known VDRA
    # fields into the sample metadata so downstream code (_expand_reusing_pilots,
    # _annotate_weighted_reuse_samples, _sample_child) can read them without
    # walking back to the original candidate dict.
    for key in (
        "vdra_cluster_id",
        "vdra_cluster_multiplicity",
        "vdra_representative_weight",
        "vdra_original_pilot_indices",
    ):
        if candidate.get(key) is not None and key not in metadata:
            metadata[key] = candidate[key]
    # P0.1: propagate the pilot's routing key so its continuation lands on
    # the same server replica for prefix cache reuse.
    if candidate.get("vdra_sticky_key") is not None and "vdra_sticky_key" not in metadata:
        metadata["vdra_sticky_key"] = candidate["vdra_sticky_key"]
    return SegmentSample(
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
        metadata=metadata,
    )


def _shortcut_pilot_samples(node: Dict[str, Any]) -> List[SegmentSample]:
    """Terminal phase-1 pilots: complete answers attached as graded leaves."""

    samples: List[SegmentSample] = []
    for candidate in list(node.get("vdra_shortcut_children") or []):
        sample = _candidate_to_sample(node, candidate)
        if sample is not None:
            samples.append(sample)
    return samples


def _retained_pilot_samples(node: Dict[str, Any], count: int) -> List[SegmentSample]:
    from recipe.gear_tree.gear_core.gear.tv_estimators import select_reuse_candidates

    pool = list(node.get("vdra_reusable_pilot_children") or node.get("vdra_pilot_children") or [])
    seed = f"vdra-reuse:{node.get('gear_segment_id', node.get('vdra_node_id', 'root'))}"
    samples: List[SegmentSample] = []
    for candidate in select_reuse_candidates(pool, count, seed=seed):
        sample = _candidate_to_sample(node, candidate)
        if sample is not None:
            samples.append(sample)
    return samples


def _pilot_overhead(node: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
    all_pilots = list(node.get("vdra_all_pilot_children") or node.get("vdra_pilot_children") or [])
    pilot_tokens = sum(len(candidate.get("response_token_ids") or []) for candidate in all_pilots)
    return all_pilots, pilot_tokens


def _refresh_generation_totals(node: Dict[str, Any]) -> None:
    node["vdra_total_generated_tokens"] = (
        int(node.get("vdra_pilot_generated_tokens", 0) or 0)
        + int(node.get("vdra_pilot_support_generated_tokens", 0) or 0)
        + int(node.get("vdra_main_expansion_generated_tokens", 0) or 0)
    )


async def _expand_fresh_iid(
    node: Dict[str, Any],
    allocated_k: int,
    max_tokens: Optional[int],
    segment_fn,
    token_budget: Optional[Dict[str, Any]] = None,
) -> List[SegmentSample]:
    requested = max(int(allocated_k), 0)
    final_count = requested
    if token_budget is not None and final_count:
        per_branch = int(max_tokens) if max_tokens is not None else int(token_budget.get("free_max_tokens", 1024))
        remaining = max(int(token_budget["cap"]) - int(token_budget["used"]), 0)
        allowed = min(final_count, remaining // per_branch) if per_branch > 0 else final_count
        if allowed < final_count:
            node["vdra_token_cap_hit"] = True
            node["vdra_budget_capped"] = True
            token_budget["cap_hit_count"] = int(token_budget.get("cap_hit_count", 0)) + 1
        final_count = max(allowed, 0)

    samples = await segment_fn(node["full_token_ids"], final_count, max_tokens) if final_count else []
    if token_budget is not None and samples:
        token_budget["used"] = int(token_budget["used"]) + sum(len(sample.token_ids) for sample in samples)

    all_pilots, pilot_tokens = _pilot_overhead(node)
    node["vdra_pilot_execution_mode"] = "fresh_iid"
    node["vdra_pilot_children_generated"] = len(all_pilots)
    node["vdra_pilot_generated_tokens"] = pilot_tokens
    node["vdra_pilot_completion_generated_tokens"] = 0
    node["vdra_pilot_children_shortcut"] = 0
    node["vdra_shortcut_overage"] = len(node.get("vdra_shortcut_children") or [])
    node["vdra_pilot_children_reused"] = 0
    node["vdra_pilot_children_discarded"] = len(all_pilots)
    node["vdra_additional_children_generated"] = len(samples)
    node["vdra_generation_request_count"] = node.get("vdra_generation_request_count", 0) + len(samples)
    node["vdra_main_expansion_generated_tokens"] = sum(len(sample.token_ids) for sample in samples)
    node["vdra_pilot_reuse_rate"] = 0.0
    _refresh_generation_totals(node)
    if not node.get("vdra_budget_capped") and len(samples) != requested:
        raise RuntimeError("fresh_iid expansion did not produce allocated_k final children")
    return list(samples)


def _sample_metadata(sample: Any) -> Dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if metadata is None:
        metadata = {}
        try:
            setattr(sample, "metadata", metadata)
        except Exception:
            return {}
    return metadata


def _annotate_weighted_reuse_samples(
    samples: List[SegmentSample],
    *,
    denominator: int,
    strict: bool = False,
) -> None:
    """Stamp cluster metadata on each weighted-reuse sample.

    P0.W1 / P0.W3: representative samples carry the cluster multiplicity
    written by the TV estimator (via _candidate_to_sample); fresh additional
    samples carry multiplicity 1. The runtime *must never* reconstruct a
    missing known multiplicity as 1 without an explicit error, so strict mode
    raises when a sample is missing the cluster fields but was supposed to be
    a representative. The per-sample ``edge_weight`` is set to the raw
    cluster multiplicity (representatives) or 1 (fresh), matching the
    "duplicate empirical samples" actor objective; representative weights are
    also stored (as multiplicity / total_weight) so the parent-value reducer
    can weight representative contributions correctly, but the actor uses
    ``edge_weight`` directly per PLAN.md P0.W3.
    """
    multiplicities: List[int] = []
    is_representative: List[bool] = []
    for sample in samples:
        metadata = _sample_metadata(sample)
        raw = metadata.get("vdra_cluster_multiplicity")
        if raw is None:
            multiplicities.append(1)
            is_representative.append(False)
        else:
            try:
                mult = max(int(raw), 1)
            except (TypeError, ValueError):
                if strict:
                    raise RuntimeError(
                        "strict weighted_reuse: sample carries a non-integer "
                        f"vdra_cluster_multiplicity {raw!r}"
                    )
                mult = 1
            multiplicities.append(mult)
            is_representative.append(True)
    total_weight = float(sum(multiplicities)) if multiplicities else 1.0
    for idx, (sample, multiplicity) in enumerate(zip(samples, multiplicities)):
        metadata = _sample_metadata(sample)
        metadata.setdefault("vdra_cluster_id", idx)
        metadata["vdra_cluster_multiplicity"] = multiplicity
        # P0.W3: representative_weight is kept for parent-value aggregation
        # (multiplicity / total_weight is the empirical proportion of the
        # parent distribution). The actor's global weight is the raw
        # multiplicity via edge_weight below.
        metadata["vdra_representative_weight"] = multiplicity / total_weight
        metadata["edge_weight"] = float(multiplicity)
        metadata["vdra_weight_objective"] = "duplicate_empirical_samples"
        metadata.setdefault("vdra_original_pilot_indices", [idx])


async def _expand_reusing_pilots(
    node: Dict[str, Any],
    allocated_k: int,
    max_tokens: Optional[int],
    segment_fn,
    token_budget: Optional[Dict[str, Any]] = None,
) -> List[SegmentSample]:
    all_shortcuts = _shortcut_pilot_samples(node)
    shortcut_budget = max(int(allocated_k), 0)
    # Terminal shortcut pilots also consume final branch slots; keep a
    # deterministic generation-order prefix so reward never influences which
    # over-budget shortcuts survive.
    shortcut = list(all_shortcuts[:shortcut_budget])
    node["vdra_shortcut_overage"] = max(len(all_shortcuts) - len(shortcut), 0)
    reuse_budget = max(int(allocated_k) - len(shortcut), 0)
    retained = _retained_pilot_samples(node, reuse_budget)
    completed: List[SegmentSample] = []
    completion_tokens = 0
    completion_requests = 0
    for sample in retained:
        sample_metadata = dict(_sample_metadata(sample))
        if sample.finish_reason == "length":
            # Free-budget depths (max_tokens None) must also finish the pilot,
            # otherwise a truncated 60-token prefix would be graded as final.
            remaining = (
                None
                if max_tokens is None
                else int(max_tokens) - int(sample.num_tokens or len(sample.token_ids))
            )
            if remaining is None or remaining > 0:
                prompt_ids = list(node["full_token_ids"]) + list(sample.token_ids)
                # P0.1: reuse the pilot's sticky_key for its continuation so
                # both requests hit the same server replica (prefix cache).
                cont_sticky = sample_metadata.get("vdra_sticky_key")
                continuation = await _segment_fn_call(
                    segment_fn, prompt_ids, 1, remaining,
                    sibling_sticky_keys=[cont_sticky] if cont_sticky else None,
                )
                completion_requests += 1
                if continuation:
                    cont = continuation[0]
                    sample = SegmentSample(
                        token_ids=list(sample.token_ids) + list(cont.token_ids),
                        text=sample.text + cont.text,
                        finish_reason=cont.finish_reason,
                        logprobs=(
                            (list(sample.logprobs) if sample.logprobs is not None else [])
                            + (list(cont.logprobs) if cont.logprobs is not None else [])
                        ) or None,
                        sum_logprobs=(
                            float(sample.sum_logprobs or 0.0)
                            + float(cont.sum_logprobs or 0.0)
                        ),
                        num_tokens=int(sample.num_tokens or 0) + int(cont.num_tokens or 0),
                        metadata=sample_metadata,
                    )
                    completion_tokens += len(cont.token_ids)
        completed.append(sample)

    if token_budget is not None:
        token_budget["used"] = int(token_budget.get("used", 0)) + completion_tokens

    missing = max(reuse_budget - len(completed), 0)
    if token_budget is not None and missing:
        # uniform_full_tree_token_cap: fresh branches only while the shared cap has
        # room for a full segment each (completions may overshoot by at most
        # one segment).
        per_branch = (
            int(max_tokens)
            if max_tokens is not None
            else int(token_budget.get("free_max_tokens", 1024))
        )
        remaining = max(int(token_budget["cap"]) - int(token_budget["used"]), 0)
        allowed = min(missing, remaining // per_branch) if per_branch > 0 else missing
        if allowed < missing:
            node["vdra_token_cap_hit"] = True
            token_budget["cap_hit_count"] = int(token_budget.get("cap_hit_count", 0)) + 1
        missing = max(allowed, 0)
    additional = (
        await segment_fn(node["full_token_ids"], missing, max_tokens)
        if missing
        else []
    )
    if token_budget is not None and additional:
        token_budget["used"] = int(token_budget["used"]) + sum(
            len(sample.token_ids) for sample in additional
        )
    all_pilots, pilot_tokens = _pilot_overhead(node)
    generated = len(all_pilots)
    reused = len(completed) + len(shortcut)
    node["vdra_pilot_execution_mode"] = "weighted_reuse"
    node["vdra_pilot_children_generated"] = generated
    node["vdra_pilot_generated_tokens"] = pilot_tokens
    node["vdra_pilot_completion_generated_tokens"] = completion_tokens
    node["vdra_pilot_children_shortcut"] = len(shortcut)
    node["vdra_pilot_children_reused"] = reused
    node["vdra_pilot_children_discarded"] = max(generated - reused, 0)
    node["vdra_additional_children_generated"] = len(additional)
    node["vdra_generation_request_count"] = (
        node.get("vdra_generation_request_count", 0) + completion_requests + len(additional)
    )
    node["vdra_main_expansion_generated_tokens"] = completion_tokens + sum(
        len(sample.token_ids) for sample in additional
    )
    node["vdra_total_generated_tokens"] = (
        node["vdra_pilot_generated_tokens"]
        + int(node.get("vdra_pilot_support_generated_tokens", 0) or 0)
        + node["vdra_main_expansion_generated_tokens"]
    )
    node["vdra_pilot_reuse_rate"] = reused / generated if generated else 0.0
    samples = shortcut + completed + list(additional)
    _annotate_weighted_reuse_samples(samples, denominator=generated + len(additional))
    return samples

async def async_build_tree_online_alloc(
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
    free_max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Online VDRA tree builder using one long-lived queue manager per tree."""

    from recipe.gear_tree.gear_core.gear.online_budget import OnlineQueueItem
    from vdra_core.logging_schema import (
        COMPUTE_PROXY_DEFINITION,
        allocation_node_records,
        budget_claim_for_mode,
        persist_vdra_artifacts,
        summarize_vdra_tree,
        validate_node_accounting,
        write_node_accounting,
    )

    t0 = time.time()
    if max_depth is None:
        max_depth = len(tree_shape)
    gear_gate.validate_main_config(
        max_default_branch_factor=max(int(x) for x in tree_shape),
        segment_length=M,
    )
    policy_snapshot_id = str(
        data_instance.get("policy_snapshot_id")
        or data_instance.get("current_rollout_snapshot_id")
        or "rollout_step:unknown"
    )
    run_id = str(data_instance.get("run_id", policy_snapshot_id))
    # PLAN.md P0.2: even in the online-alloc path, prefer the caller-supplied
    # tree_instance_id when present; otherwise mint a globally unique one so
    # replay/tensorization/manifest never see two independent trees collapse
    # to the same tree_id.
    tree_id = str(
        data_instance.get("tree_instance_id")
        or data_instance.get("tree_id")
        or make_tree_instance_id(
            policy_snapshot_id=policy_snapshot_id,
            rollout_iteration=data_instance.get("rollout_iteration"),
            stable_question_id=data_instance.get("_treetune__idx")
            or data_instance.get("uid"),
            tree_instance_uuid=data_instance.get("tree_instance_uuid"),
        )
    )
    budget_mode = str(getattr(gear_gate, "budget_mode", "fixed_main"))
    token_budget: Optional[Dict[str, Any]] = None
    if budget_mode in {"fixed_total_generated", "uniform_full_tree_token_cap"}:
        token_budget = {
            "cap": _uniform_generated_token_cap(tree_shape, M, max_depth, free_max_tokens),
            "used": 0,
            "cap_hit_count": 0,
            "free_max_tokens": int(free_max_tokens),
        }

    allocation_proxy = str(getattr(gear_gate, "allocation_proxy", "vdra"))
    proxy_rollout_fn = None
    if allocation_proxy in {"empirical_variance", "oracle"}:
        async def proxy_rollout_fn(node: Dict[str, Any], rollouts: int) -> List[float]:
            """Grade ``rollouts`` free continuations of ``node`` (proxy input).

            Unfinished continuations score 0.0. Token cost is charged to
            ``vdra_proxy_rollout_tokens`` (and the shared cap when active),
            never to pilot or main-expansion counters.
            """

            samples = await segment_fn(node["full_token_ids"], int(rollouts), None)
            generated = sum(len(sample.token_ids) for sample in samples)
            node["vdra_proxy_rollout_tokens"] = (
                int(node.get("vdra_proxy_rollout_tokens", 0) or 0) + generated
            )
            node["vdra_generation_request_count"] = (
                node.get("vdra_generation_request_count", 0) + len(samples)
            )
            if token_budget is not None:
                token_budget["used"] = int(token_budget["used"]) + generated
            return [
                float(grade_fn(node["full_text"], node["full_text"] + sample.text, data_instance))
                if sample.finish_reason != "length"
                else 0.0
                for sample in samples
            ]

    manager = gear_gate.make_queue_manager(policy_snapshot_id=policy_snapshot_id)
    poll_interval = max(min(float(manager.timeout_seconds) / 4.0, 0.1), 0.01)
    stop_timeout_worker = asyncio.Event()
    pending_futures: List[asyncio.Future] = []
    queue_flush_records: List[Dict[str, Any]] = []
    worker_error: Optional[BaseException] = None

    tree: Dict[str, Any] = {
        "text": root_prompt_text,
        "depth": 0,
        "full_text": root_prompt_text,
        "stop_text": "aaa",
        "_request_object": data_instance,
        "leaf": False,
        "full_token_ids": list(root_prompt_token_ids),
        "gear_segment_id": "root",
        "policy_snapshot_id": policy_snapshot_id,
        "vdra_policy_snapshot_id": policy_snapshot_id,
        "tree_id": tree_id,
        "tree_instance_id": tree_id,
    }

    def _default_bf(depth: int) -> int:
        return int(tree_shape[depth] if depth < len(tree_shape) else tree_shape[-1])

    def _max_tokens(depth: int) -> Optional[int]:
        return None if depth == max_depth - 1 else M

    def _sample_child(parent: Dict[str, Any], depth: int, idx: int, sample: SegmentSample) -> Dict[str, Any]:
        prefix = parent["full_text"]
        child = {
            "text": sample.text,
            "depth": depth + 1,
            "full_text": prefix + sample.text,
            "stop_text": None,
            "finish_reason": sample.finish_reason,
            "response_token_ids": list(sample.token_ids),
            "actor_shifted_log_probs": list(sample.logprobs) if sample.logprobs is not None else None,
            "full_token_ids": list(parent["full_token_ids"]) + list(sample.token_ids),
            "gear_segment_id": f"{parent.get('gear_segment_id', 'root')}/{depth}/{idx}",
            "gear_parent_segment_id": parent.get("gear_segment_id", "root"),
        }
        metadata = _sample_metadata(sample)
        child.update(dict(metadata or {}))
        # P0.W3: the actor's global weight is the raw cluster multiplicity
        # (equivalent to duplicating the represented pilots), NOT the per-
        # parent normalized probability. _annotate_weighted_reuse_samples
        # writes edge_weight=multiplicity onto metadata; use it directly so
        # the actor loss matches the "duplicate empirical samples" objective.
        raw_edge_weight = metadata.get("edge_weight")
        if raw_edge_weight is not None:
            child["edge_weight"] = float(raw_edge_weight)
        elif metadata.get("vdra_representative_weight") is not None:
            # Legacy fallback (fresh_iid + noop annotator paths).
            child["edge_weight"] = float(metadata["vdra_representative_weight"])
        if sample.sum_logprobs is not None:
            child["sum_logprobs"] = float(sample.sum_logprobs)
            child["num_tokens"] = int(sample.num_tokens)
        return child

    def _raise_worker_error() -> None:
        if worker_error is not None:
            raise RuntimeError("VDRA queue timeout worker failed") from worker_error

    expansion_tasks: set = set()

    async def _expand_flushed_item(result, item) -> None:
        node = item.node
        node_key = str(node.get("gear_segment_id"))
        try:
            allocated = int(result.summary.allocations.get(node_key, node.get("vdra_allocated_k", 0)))
            node["vdra_allocation_seconds"] = result.allocation_seconds
            await _expand_parent(
                node,
                depth=item.depth,
                allocated_branch_factor=allocated,
                default_branch_factor=item.default_branch_factor,
            )
            if "reward" not in node:
                raise RuntimeError(f"VDRA queued node {node_key} returned without reward")
            if item.completion_future is not None and not item.completion_future.done():
                item.completion_future.set_result(node)
        except Exception as exc:
            if item.completion_future is not None and not item.completion_future.done():
                # The waiter re-raises; swallowing here avoids a duplicate
                # "exception never retrieved" from the background task.
                item.completion_future.set_exception(exc)
            else:
                raise

    async def _handle_flush(result) -> None:
        # Never expand inline: the flush caller may be the timeout worker, and
        # blocking it on a whole-subtree expansion would serialize the tree
        # (and deadlock nodes the subtree enqueues). Expansion runs as tasks.
        queue_flush_records.append(result.to_record())
        for item in result.items:
            task = asyncio.create_task(_expand_flushed_item(result, item))
            expansion_tasks.add(task)
            task.add_done_callback(expansion_tasks.discard)

    async def _flush_ready() -> None:
        _raise_worker_error()
        for result in await manager.flush_ready():
            await _handle_flush(result)
        _raise_worker_error()

    async def _queue_timeout_worker() -> None:
        nonlocal worker_error
        try:
            while not stop_timeout_worker.is_set():
                await asyncio.sleep(poll_interval)
                for result in await manager.flush_ready():
                    await _handle_flush(result)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # propagate to all waiters and main task
            worker_error = exc
            for future in pending_futures:
                if not future.done():
                    future.set_exception(exc)

    timeout_worker = asyncio.create_task(_queue_timeout_worker())

    async def _drain_final() -> None:
        while True:
            results = await manager.drain()
            if not results:
                break
            for result in results:
                await _handle_flush(result)

    async def _process_expandable(node: Dict[str, Any], depth: int) -> None:
        _raise_worker_error()
        if depth >= max_depth:
            node["vdra_expansion_skipped_terminal"] = True
            node["reward"] = float(grade_fn(node["full_text"], node["text"], data_instance))
            node["leaf"] = True
            return
        if token_budget is not None and token_budget["used"] >= token_budget["cap"]:
            # uniform_full_tree_token_cap: the shared cap is exhausted — grade the
            # truncated node instead of spending more pilot/main tokens.
            token_budget["cap_hit_count"] = int(token_budget["cap_hit_count"]) + 1
            node["vdra_token_cap_hit"] = True
            node["vdra_expansion_skipped_token_cap"] = True
            node["reward"] = float(grade_fn(node["full_text"], node["text"], data_instance))
            node["leaf"] = True
            return
        default_bf = _default_bf(depth)
        near_leaf = bool(getattr(gear_gate, "skip_near_leaf_expand", False) and depth == max_depth - 1)
        if near_leaf or (depth == 0 and not getattr(gear_gate, "root_allocation", False)):
            await _expand_parent(
                node,
                depth=depth,
                allocated_branch_factor=default_bf,
                default_branch_factor=default_bf,
            )
            if "reward" not in node:
                raise RuntimeError(f"VDRA direct node {node.get('gear_segment_id')} returned without reward")
            return

        try:
            await gear_gate.estimate_node_async(
                node,
                depth=depth,
                default_bf=default_bf,
                node_expander=gear_node_expander,
                proxy_rollout_fn=proxy_rollout_fn,
            )
        except Exception as exc:
            gear_gate.allocation_error_count += 1
            raise RuntimeError(
                f"VDRA pilot/scoring failed at depth {depth}; no fallback is allowed"
            ) from exc

        if token_budget is not None:
            token_budget["used"] = (
                int(token_budget["used"])
                + sum(
                    len(candidate.get("response_token_ids") or [])
                    for candidate in node.get("vdra_all_pilot_children") or []
                )
                + int(node.get("vdra_pilot_support_generated_tokens", 0) or 0)
            )

        # Queue every estimated node. The unified bounded integer solver decides
        # pruning, expansion, or no-op jointly across the flush frontier.
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        pending_futures.append(future)
        manager.enqueue(
            OnlineQueueItem(
                node=node,
                default_branch_factor=default_bf,
                depth=depth,
                policy_snapshot_id=policy_snapshot_id,
                completion_future=future,
            )
        )
        # Flush immediately when this enqueue filled a queue to capacity;
        # otherwise the background worker fires the timeout flush while
        # sibling coroutines keep filling the queue concurrently.
        await _flush_ready()
        await future
        if "reward" not in node:
            raise RuntimeError(f"VDRA queued node {node.get('gear_segment_id')} returned without reward")

    async def _expand_parent(
        node: Dict[str, Any], *, depth: int, allocated_branch_factor: int, default_branch_factor: int
    ) -> None:
        allocated_branch_factor = max(int(allocated_branch_factor), 0)
        if "vdra_default_k" not in node:
            write_node_accounting(
                node,
                default_k=default_branch_factor,
                predicted_k=default_branch_factor,
                allocated_k=allocated_branch_factor,
                k_min=getattr(gear_gate, "n_min", 1),
                dispersion_C=0.0,
            )
        validate_node_accounting(node, k_min=getattr(gear_gate, "n_min", 1))
        node["vdra_generation_request_count"] = node.get("vdra_generation_request_count", 0)
        if allocated_branch_factor <= 0:
            node["reward"] = float(grade_fn(node["full_text"], node["text"], data_instance))
            node["leaf"] = True
            return
        pilot_mode = str(getattr(gear_gate, "pilot_execution_mode", "fresh_iid"))
        if pilot_mode == "fresh_iid":
            samples = await _expand_fresh_iid(
                node, allocated_branch_factor, _max_tokens(depth), segment_fn,
                token_budget=token_budget,
            )
        elif pilot_mode == "weighted_reuse":
            # P0.6: weighted_reuse's parent value estimator sums over every
            # cluster's representative weight; if allocated_k drops even one
            # required cluster the remaining weights cannot recover the
            # missing probability mass. Detect the coverage shortfall and
            # execute the configured fallback here rather than trusting the
            # config field to have effect.
            # P0.W2: required coverage counts continuable representative
            # clusters PLUS terminal singleton clusters. The estimator writes
            # vdra_required_cluster_count = len(representative_index_per_cluster)
            # which already sums both; the fallback below reconstructs the
            # same total from the reusable pilots and shortcut children in
            # case the field was not set upstream.
            required_clusters = int(
                node.get("vdra_required_cluster_count")
                or (
                    len(node.get("vdra_reusable_pilot_children") or node.get("vdra_pilot_children") or [])
                    + len(node.get("vdra_shortcut_children") or [])
                )
            )
            node["vdra_required_cluster_count"] = required_clusters
            node["vdra_allocated_k"] = allocated_branch_factor
            if allocated_branch_factor >= required_clusters or required_clusters == 0:
                samples = await _expand_reusing_pilots(
                    node, allocated_branch_factor, _max_tokens(depth), segment_fn,
                    token_budget=token_budget,
                )
                node["vdra_weighted_reuse_fallback_triggered"] = False
                node["vdra_weighted_reuse_fallback_reason"] = ""
            else:
                fallback = str(getattr(gear_gate, "weighted_reuse_fallback", "fresh_iid"))
                reason = (
                    f"allocated_k={allocated_branch_factor} < "
                    f"required_cluster_count={required_clusters}"
                )
                if fallback == "error":
                    raise RuntimeError(
                        "weighted_reuse coverage impossible and "
                        f"weighted_reuse_fallback='error': {reason}"
                    )
                if fallback != "fresh_iid":
                    raise ValueError(
                        f"Unsupported weighted_reuse_fallback: {fallback!r}"
                    )
                samples = await _expand_fresh_iid(
                    node, allocated_branch_factor, _max_tokens(depth), segment_fn,
                    token_budget=token_budget,
                )
                node["vdra_weighted_reuse_fallback_triggered"] = True
                node["vdra_weighted_reuse_fallback_reason"] = reason
                # fresh_iid children must not carry representative weights, so
                # strip any cluster metadata the caller may have pre-populated.
                for sample in samples:
                    metadata = _sample_metadata(sample)
                    for key in (
                        "vdra_cluster_id",
                        "vdra_cluster_multiplicity",
                        "vdra_representative_weight",
                        "vdra_original_pilot_indices",
                    ):
                        metadata.pop(key, None)
        else:
            raise ValueError(f"Unsupported pilot_execution_mode: {pilot_mode!r}")
        children = [
            _sample_child(node, depth, idx, sample)
            for idx, sample in enumerate(samples)
        ]
        children = await _filter_children_any(gear_gate, node, depth, default_branch_factor, children)
        node["children"] = children
        expandable: List[Dict[str, Any]] = []
        for child in children:
            child_depth = depth + 1
            if child["finish_reason"] != "length" or child_depth >= max_depth:
                child["vdra_expansion_skipped_terminal"] = True
                child["reward"] = float(grade_fn(node["full_text"], child["full_text"], data_instance))
                child["leaf"] = True
            else:
                child["leaf"] = False
                expandable.append(child)
        if expandable:
            # Siblings expand concurrently so they can co-occupy allocation
            # queues — the batchwise sqrt(C_s) solve needs more than one node
            # per flush to have any effect.
            await asyncio.gather(
                *(_process_expandable(child, depth + 1) for child in expandable)
            )
        _aggregate_child_rewards(node, children)

    try:
        await _process_expandable(tree, 0)
        # Drain while the worker is still alive: a drained subtree may enqueue
        # deeper nodes that then need timeout flushes of their own.
        await _drain_final()
        while expansion_tasks:
            expansion_tasks.difference_update(task for task in list(expansion_tasks) if task.done())
            if not expansion_tasks:
                break
            await asyncio.gather(*list(expansion_tasks))
            expansion_tasks.difference_update(task for task in list(expansion_tasks) if task.done())
            await _drain_final()
    finally:
        stop_timeout_worker.set()
        await timeout_worker

    if any(not future.done() for future in pending_futures):
        raise RuntimeError("VDRA final drain left pending queue futures")

    def backprop(node: Dict[str, Any]) -> None:
        children = node.get("children") or []
        for child in children:
            backprop(child)
        if children:
            _aggregate_child_rewards(node, children)

    backprop(tree)

    if getattr(gear_gate, "strict_vdra", True):
        if manager.reserve_pool.value != 0:
            raise RuntimeError("VDRA reserve invariant failed")
        if any(queue.items for queue in manager.queues):
            raise RuntimeError("VDRA queue invariant failed: queues are not empty")
        for record in allocation_node_records(tree, run_id=run_id, tree_id=tree_id):
            node = next(n for n in _iter_dict_nodes(tree) if str(n.get("gear_segment_id", n.get("vdra_node_id", ""))) == record["node_id"] or str(n.get("vdra_node_id", "")) == record["node_id"])
            validate_node_accounting(node, k_min=getattr(gear_gate, "n_min", 1))
            generated = int(node.get("vdra_pilot_children_generated", 0) or 0)
            reused = int(node.get("vdra_pilot_children_reused", 0) or 0)
            shortcut = int(node.get("vdra_pilot_children_shortcut", 0) or 0)
            discarded = int(node.get("vdra_pilot_children_discarded", 0) or 0)
            mode = str(node.get("vdra_pilot_execution_mode", tree.get("vdra_pilot_execution_mode", "fresh_iid")))
            if mode == "fresh_iid":
                if reused != 0 or shortcut != 0 or discarded != generated:
                    raise RuntimeError("VDRA fresh_iid pilot accounting invariant failed")
                allocated = int(node.get("vdra_allocated_k", 0) or 0)
                final_children = len(node.get("children") or [])
                if not node.get("vdra_budget_capped") and final_children != allocated:
                    raise RuntimeError("VDRA fresh_iid final child count invariant failed")
            elif reused > generated or shortcut > reused or discarded != generated - reused:
                raise RuntimeError("VDRA pilot accounting invariant failed")
            if int(node.get("vdra_total_generated_tokens", 0) or 0) != (
                int(node.get("vdra_pilot_generated_tokens", 0) or 0)
                + int(node.get("vdra_pilot_support_generated_tokens", 0) or 0)
                + int(node.get("vdra_main_expansion_generated_tokens", 0) or 0)
            ):
                raise RuntimeError("VDRA generated-token accounting invariant failed")
            if int(node.get("vdra_total_scored_tokens", 0) or 0) != int(node.get("vdra_likelihood_scored_prompt_tokens", 0) or 0) + int(node.get("vdra_likelihood_scored_continuation_tokens", 0) or 0):
                raise RuntimeError("VDRA scored-token accounting invariant failed")
        totals = summarize_vdra_tree(tree)
        if int(totals["vdra_total_redistributed_branches"]) != int(manager.reserve_consumed):
            raise RuntimeError("VDRA redistribution accounting invariant failed")

    tree["gear_queue_flush_count"] = int(manager.flush_count)
    tree["gear_queue_timeout_flush_count"] = int(manager.timeout_flush_count)
    tree["vdra_queue_capacity_flush_count"] = int(manager.capacity_flush_count)
    tree["vdra_queue_final_drain_count"] = int(manager.final_drain_count)
    tree["gear_reserve_contributed"] = int(sum(record.get("total_saved_budget", 0) for record in queue_flush_records))
    tree["gear_reserve_consumed"] = int(manager.reserve_consumed)
    tree["gear_reserve_remaining"] = 0
    tree["vdra_queue_flush_records"] = queue_flush_records
    # P1.1: runtime allocation is solved separately per queue flush, not for
    # the whole tree or the whole depth frontier.
    tree["vdra_allocation_scope"] = "per_queue_flush_within_tree"
    tree["vdra_budget_mode"] = budget_mode
    tree["vdra_allocation_proxy"] = allocation_proxy
    tree["vdra_pilot_execution_mode"] = str(getattr(gear_gate, "pilot_execution_mode", "fresh_iid"))
    if token_budget is not None:
        tree["vdra_token_cap"] = int(token_budget["cap"])
        tree["vdra_generated_tokens_under_cap"] = int(token_budget["used"])
        tree["vdra_token_cap_hit_count"] = int(token_budget["cap_hit_count"])
    tree["tree_construction_seconds"] = time.time() - t0

    artifact_dir = getattr(gear_gate, "artifact_dir", None) or data_instance.get("vdra_artifact_dir")
    if artifact_dir:
        persist_vdra_artifacts(
            artifact_dir,
            tree,
            run_id=run_id,
            tree_id=tree_id,
            queue_flushes=queue_flush_records,
            run_manifest={
                "algorithm_requested": "VDRA",
                "algorithm_executed": "VDRA-online-timeout",
                # Oracle-proxy runs consume graded oracle rollouts and are
                # evaluation-only; they must never be reported as main results.
                "run_valid_for_main_results": allocation_proxy != "oracle",
                "allocation_proxy": allocation_proxy,
                "pilot_execution_mode": tree["vdra_pilot_execution_mode"],
                "oracle_rollouts_per_node": (
                    getattr(gear_gate, "oracle_rollouts_per_node", None)
                    if allocation_proxy == "oracle"
                    else None
                ),
                "token_cap": tree.get("vdra_token_cap"),
                "strict_vdra": bool(getattr(gear_gate, "strict_vdra", True)),
                "tree_shape": list(tree_shape),
                "segment_length": M,
                "pilot_branch_factor": getattr(gear_gate, "pilot_branch_factor", None),
                "likelihood_samples_per_distribution": getattr(gear_gate, "likelihood_samples_per_distribution", None),
                "first_phase_tokens": getattr(gear_gate, "tv_first_phase_tokens", None),
                "second_phase_tokens": getattr(gear_gate, "tv_second_phase_tokens", None),
                "allocation_runtime": getattr(gear_gate, "allocation_runtime", None),
                # P1.3: canonical value everywhere — see PLAN.md.
                "allocation_scope": "per_queue_flush_within_tree",
                "queue_count": getattr(gear_gate, "queue_count", None),
                "queue_capacity": getattr(gear_gate, "queue_capacity", None),
                "queue_timeout_seconds": getattr(gear_gate, "queue_timeout_seconds", None),
                "root_allocation": bool(getattr(gear_gate, "root_allocation", False)),
                "use_residual_budget": bool(getattr(gear_gate, "use_residual_budget", True)),
                "n_min": getattr(gear_gate, "n_min", None),
                "tv_estimator": getattr(gear_gate, "tv_estimator", None),
                "bound_form": getattr(getattr(gear_gate, "cfg", None), "bound_form", None),
                "eps_tail": getattr(getattr(gear_gate, "cfg", None), "eps_tail", None),
                "eps_tail_calibration_path": getattr(gear_gate, "eps_tail_calibration_path", None),
                "eps_tail_calibration_metadata": getattr(gear_gate, "eps_tail_calibration_metadata", None),
                "policy_snapshot_id": policy_snapshot_id,
                "scorer_snapshot_id": getattr(getattr(gear_gate, "scorer", None), "scorer_snapshot_id", None),
                "scorer_model": getattr(getattr(gear_gate, "scorer", None), "scorer_model", None),
                "budget_mode": tree["vdra_budget_mode"],
                "budget_claim": budget_claim_for_mode(tree["vdra_budget_mode"]),
                "compute_proxy_definition": COMPUTE_PROXY_DEFINITION,
            },
        )
    return tree


def _iter_dict_nodes(tree: Dict[str, Any]):
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.get("children") or []))

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

    # PLAN.md P0.2: same globally-unique tree id as the sync/online builders.
    tree_instance_id = make_tree_instance_id(
        policy_snapshot_id=data_instance.get("policy_snapshot_id")
        or data_instance.get("current_rollout_snapshot_id"),
        rollout_iteration=data_instance.get("rollout_iteration"),
        stable_question_id=data_instance.get("_treetune__idx")
        or data_instance.get("uid"),
        tree_instance_uuid=data_instance.get("tree_instance_uuid"),
    )

    tree: Dict[str, Any] = {
        "text": root_prompt_text, "depth": 0, "full_text": root_prompt_text,
        "stop_text": "aaa", "_request_object": data_instance, "leaf": False,
        "full_token_ids": list(root_prompt_token_ids),
        "tree_id": tree_instance_id,
        "tree_instance_id": tree_instance_id,
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

        pilot_mode = str(getattr(gear_gate, "pilot_execution_mode", "fresh_iid"))
        expand_fn = _expand_fresh_iid if pilot_mode == "fresh_iid" else _expand_reusing_pilots
        sample_batches = await asyncio.gather(
            *[expand_fn(node, bf, max_tokens, segment_fn) for node, bf in zip(expand_nodes, branch_factors)]
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
                metadata = _sample_metadata(s)
                child.update(dict(metadata or {}))
                # P0.W3: multiplicity-based edge_weight (see _sample_child).
                raw_edge_weight = metadata.get("edge_weight")
                if raw_edge_weight is not None:
                    child["edge_weight"] = float(raw_edge_weight)
                elif metadata.get("vdra_representative_weight") is not None:
                    child["edge_weight"] = float(metadata["vdra_representative_weight"])
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
        _aggregate_child_rewards(node, children)

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
