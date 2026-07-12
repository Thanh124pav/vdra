import asyncio
import importlib.util
from pathlib import Path

import pytest

from treetune.gear import budget_allocation as treetune_alloc
from vdra_core.online_budget import (
    OnlineQueueItem,
    RootQueueManager,
    SharedReservePool,
)


def _node(name, default_k, predicted_k, dispersion_C):
    return {
        "vdra_node_id": name,
        "vdra_default_k": default_k,
        "vdra_predicted_k": predicted_k,
        "vdra_dispersion_C": dispersion_C,
    }


def test_treetune_and_verl_use_the_same_allocator():
    from verl.recipe.gear_tree.gear_core.gear import budget_allocation as verl_alloc

    nodes = [_node("a", 6, 2, 0.1), _node("b", 6, 10, 1.0)]
    left = treetune_alloc.allocate_branch_factors(nodes, total_budget=12)
    right = verl_alloc.allocate_branch_factors(nodes, total_budget=12)
    assert left == right


def test_queue_flushes_on_capacity_and_preserves_snapshot():
    async def run():
        pool = SharedReservePool(queue_count=1)
        manager = RootQueueManager(
            queue_count=1,
            queue_capacity=2,
            timeout_seconds=99,
            reserve_pool=pool,
            policy_snapshot_id="p0",
        )
        for idx in range(2):
            node = _node(str(idx), 2, 3, idx + 1.0)
            manager.enqueue(
                OnlineQueueItem(node, 2, 0, policy_snapshot_id="p0"), now=0.0
            )
        flushed = await manager.flush_ready(now=0.1)
        return manager, flushed

    manager, flushed = asyncio.run(run())
    assert len(flushed) == 1
    assert flushed[0].flush_reason == "capacity"
    assert manager.capacity_flush_count == 1


def test_queue_flushes_by_timeout_without_later_frontier_node():
    async def run():
        manager = RootQueueManager(
            queue_count=1,
            queue_capacity=8,
            timeout_seconds=0.5,
            reserve_pool=SharedReservePool(queue_count=1),
            policy_snapshot_id="p0",
        )
        node = _node("a", 1, 2, 1.0)
        manager.enqueue(OnlineQueueItem(node, 1, 0, policy_snapshot_id="p0"), now=1.0)
        return await manager.flush_ready(now=1.6)

    flushed = asyncio.run(run())
    assert flushed[0].flush_reason == "timeout"
    assert flushed[0].queue_wait_seconds == pytest.approx(0.6)


def test_calibration_grades_full_pilot_plus_continuation():
    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_tail_divergence.py"
    spec = importlib.util.spec_from_file_location("vdra_calibration_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pilot = "reasoning\n# Answer\n42"
    continuation = ""
    assert module.simple_math_grade(continuation, "42", "# Answer\n") == 0.0
    assert module.simple_math_grade(pilot + continuation, "42", "# Answer\n") == 1.0
    assert "simple_math_grade(pilot + text" in script.read_text(encoding="utf-8")


def test_no_public_threshold_lambda_in_production_sources():
    root = Path(__file__).resolve().parents[1]
    checked = [
        root / "vdra_core",
        root / "treetune" / "gear",
        root / "verl" / "recipe" / "gear_tree",
    ]
    offenders = []
    for base in checked:
        for path in base.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".jsonnet", ".sh"}:
                continue
            if "budget_lambda" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []
