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
    for candidate in list(node.get("vdra_reusable_pilot_children") or node.get("vdra_pilot_children") or [])[:allocated_k]:
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
    completed: List[SegmentSample] = []
    completion_tokens = 0
    for sample in retained:
        if (
            max_tokens is not None
            and sample.finish_reason == "length"
            and int(sample.num_tokens or len(sample.token_ids)) < int(max_tokens)
        ):
            remaining = int(max_tokens) - int(sample.num_tokens or len(sample.token_ids))
            prompt_ids = list(node["full_token_ids"]) + list(sample.token_ids)
            continuation = await segment_fn(prompt_ids, 1, remaining)
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
                )
                completion_tokens += len(cont.token_ids)
        completed.append(sample)

    missing = max(int(allocated_k) - len(completed), 0)
    additional = (
        await segment_fn(node["full_token_ids"], missing, max_tokens)
        if missing
        else []
    )
    all_pilots = list(node.get("vdra_all_pilot_children") or node.get("vdra_pilot_children") or [])
    generated = len(all_pilots)
    node["vdra_pilot_children_generated"] = generated
    node["vdra_pilot_generated_tokens"] = sum(
        len(candidate.get("response_token_ids") or [])
        for candidate in all_pilots
    )
    node["vdra_pilot_completion_generated_tokens"] = completion_tokens
    node["vdra_pilot_children_reused"] = len(completed)
    node["vdra_pilot_children_discarded"] = max(generated - len(completed), 0)
    node["vdra_additional_children_generated"] = len(additional)
    node["vdra_generation_request_count"] = node.get("vdra_generation_request_count", 0) + int(bool(completion_tokens)) + len(additional)
    node["vdra_main_expansion_generated_tokens"] = completion_tokens + sum(
        len(sample.token_ids) for sample in additional
    )
    node["vdra_total_generated_tokens"] = (
        node["vdra_pilot_generated_tokens"]
        + node["vdra_main_expansion_generated_tokens"]
    )
    node["vdra_pilot_reuse_rate"] = len(completed) / generated if generated else 0.0
    return completed + list(additional)

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
) -> Dict[str, Any]:
    """Online VDRA tree builder using one long-lived queue manager per tree."""

    from recipe.gear_tree.gear_core.gear.online_budget import OnlineQueueItem
    from vdra_core.logging_schema import (
        allocation_node_records,
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
    policy_snapshot_id = str(data_instance.get("_treetune__idx", "tree"))
    run_id = str(data_instance.get("run_id", policy_snapshot_id))
    tree_id = str(data_instance.get("tree_id", policy_snapshot_id))
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
        if sample.sum_logprobs is not None:
            child["sum_logprobs"] = float(sample.sum_logprobs)
            child["num_tokens"] = int(sample.num_tokens)
        return child

    def _raise_worker_error() -> None:
        if worker_error is not None:
            raise RuntimeError("VDRA queue timeout worker failed") from worker_error

    async def _handle_flush(result) -> None:
        queue_flush_records.append(result.to_record())
        for item in result.items:
            node = item.node
            try:
                node_key = str(node.get("gear_segment_id"))
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
                    item.completion_future.set_exception(exc)
                raise

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
            )
        except Exception as exc:
            gear_gate.allocation_error_count += 1
            raise RuntimeError(
                f"VDRA pilot/scoring failed at depth {depth}; no fallback is allowed"
            ) from exc

        saved_k = int(node.get("vdra_saved_k", 0) or 0)
        if saved_k and getattr(gear_gate, "use_residual_budget", False):
            await manager.reserve_pool.add(saved_k)
        if int(node.get("vdra_unmet_demand", 0) or 0) <= 0:
            await _expand_parent(
                node,
                depth=depth,
                allocated_branch_factor=int(node.get("vdra_base_k", default_bf)),
                default_branch_factor=default_bf,
            )
            if "reward" not in node:
                raise RuntimeError(f"VDRA direct node {node.get('gear_segment_id')} returned without reward")
            return

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
        await _flush_ready()
        if not future.done():
            # If this coroutine itself is inside a flush handler, the background
            # worker cannot make progress until we yield past the timeout.
            await asyncio.sleep(float(manager.timeout_seconds) + poll_interval)
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
        samples = await _expand_reusing_pilots(
            node, allocated_branch_factor, _max_tokens(depth), segment_fn
        )
        children = [
            _sample_child(node, depth, idx, sample)
            for idx, sample in enumerate(samples[:allocated_branch_factor])
        ]
        children = await _filter_children_any(gear_gate, node, depth, default_branch_factor, children)
        node["children"] = children
        for child in children:
            child_depth = depth + 1
            if child["finish_reason"] != "length" or child_depth >= max_depth:
                child["vdra_expansion_skipped_terminal"] = True
                child["reward"] = float(grade_fn(node["full_text"], child["full_text"], data_instance))
                child["leaf"] = True
            else:
                child["leaf"] = False
                await _process_expandable(child, child_depth)
        child_rewards = [child["reward"] for child in children]
        node["reward"] = float(np.mean(child_rewards)) if child_rewards else 0.0
        node["reward_std"] = float(np.std(child_rewards)) if child_rewards else 0.0

    try:
        await _process_expandable(tree, 0)
    finally:
        stop_timeout_worker.set()
        await timeout_worker

    await _drain_final()
    if any(not future.done() for future in pending_futures):
        raise RuntimeError("VDRA final drain left pending queue futures")

    def backprop(node: Dict[str, Any]) -> None:
        children = node.get("children") or []
        for child in children:
            backprop(child)
        if children:
            child_rewards = [child["reward"] for child in children]
            node["reward"] = float(np.mean(child_rewards))
            node["reward_std"] = float(np.std(child_rewards))

    backprop(tree)

    if getattr(gear_gate, "strict_vdra", True):
        if manager.reserve_pool.contributed != manager.reserve_pool.consumed + manager.reserve_pool.value:
            raise RuntimeError("VDRA reserve invariant failed")
        if any(queue.items for queue in manager.queues):
            raise RuntimeError("VDRA queue invariant failed: queues are not empty")
        for record in allocation_node_records(tree, run_id=run_id, tree_id=tree_id):
            node = next(n for n in _iter_dict_nodes(tree) if str(n.get("gear_segment_id", n.get("vdra_node_id", ""))) == record["node_id"] or str(n.get("vdra_node_id", "")) == record["node_id"])
            validate_node_accounting(node, k_min=getattr(gear_gate, "n_min", 1))
            generated = int(node.get("vdra_pilot_children_generated", 0) or 0)
            reused = int(node.get("vdra_pilot_children_reused", 0) or 0)
            discarded = int(node.get("vdra_pilot_children_discarded", 0) or 0)
            if reused > generated or discarded != generated - reused:
                raise RuntimeError("VDRA pilot accounting invariant failed")
            if int(node.get("vdra_total_generated_tokens", 0) or 0) != int(node.get("vdra_pilot_generated_tokens", 0) or 0) + int(node.get("vdra_main_expansion_generated_tokens", 0) or 0):
                raise RuntimeError("VDRA generated-token accounting invariant failed")
            if int(node.get("vdra_total_scored_tokens", 0) or 0) != int(node.get("vdra_likelihood_scored_prompt_tokens", 0) or 0) + int(node.get("vdra_likelihood_scored_continuation_tokens", 0) or 0):
                raise RuntimeError("VDRA scored-token accounting invariant failed")
        totals = summarize_vdra_tree(tree)
        if int(totals["vdra_total_redistributed_branches"]) != int(manager.reserve_pool.consumed):
            raise RuntimeError("VDRA redistribution accounting invariant failed")

    tree["gear_queue_flush_count"] = int(manager.flush_count)
    tree["gear_queue_timeout_flush_count"] = int(manager.timeout_flush_count)
    tree["vdra_queue_capacity_flush_count"] = int(manager.capacity_flush_count)
    tree["vdra_queue_final_drain_count"] = int(manager.final_drain_count)
    tree["gear_reserve_contributed"] = int(manager.reserve_pool.contributed)
    tree["gear_reserve_consumed"] = int(manager.reserve_pool.consumed)
    tree["gear_reserve_remaining"] = int(manager.reserve_pool.value)
    tree["vdra_queue_flush_records"] = queue_flush_records
    tree["vdra_allocation_scope"] = "one_tree"
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
                "run_valid_for_main_results": True,
                "strict_vdra": bool(getattr(gear_gate, "strict_vdra", True)),
                "tree_shape": list(tree_shape),
                "segment_length": M,
                "pilot_branch_factor": getattr(gear_gate, "pilot_branch_factor", None),
                "likelihood_samples_per_distribution": getattr(gear_gate, "likelihood_samples_per_distribution", None),
                "first_phase_tokens": getattr(gear_gate, "tv_first_phase_tokens", None),
                "second_phase_tokens": getattr(gear_gate, "tv_second_phase_tokens", None),
                "allocation_runtime": getattr(gear_gate, "allocation_runtime", None),
                "allocation_scope": "one_tree",
                "queue_count": getattr(gear_gate, "queue_count", None),
                "queue_capacity": getattr(gear_gate, "queue_capacity", None),
                "queue_timeout_seconds": getattr(gear_gate, "queue_timeout_seconds", None),
                "root_allocation": bool(getattr(gear_gate, "root_allocation", False)),
                "use_residual_budget": bool(getattr(gear_gate, "use_residual_budget", True)),
                "n_min": getattr(gear_gate, "n_min", None),
                "tv_estimator": getattr(gear_gate, "tv_estimator", None),
                "bound_form": getattr(getattr(gear_gate, "cfg", None), "bound_form", None),
                "eps_tail": getattr(getattr(gear_gate, "cfg", None), "eps_tail", None),
                "eps_tail_calibration_path": None,
                "eps_tail_calibration_metadata": None,
                "budget_claim": "fixed main expansion budget; pilot and scoring overhead reported separately",
                "compute_proxy_definition": "pilot decode tokens + main-expansion decode tokens + scored prompt tokens + scored continuation tokens",
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
