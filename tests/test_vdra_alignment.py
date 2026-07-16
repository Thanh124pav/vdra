import asyncio
import json
import importlib.util
import time
from pathlib import Path

import pytest

from treetune.gear import budget_allocation as treetune_alloc
from vdra_core import allocate_branch_factors, node_allocated_k, summarize_vdra_tree, validate_node_accounting
from vdra_core.calibration import load_tail_calibration
from vdra_core.logging_schema import persist_vdra_artifacts
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
    from recipe.gear_tree.gear_core.gear import budget_allocation as verl_alloc

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



def test_canonical_vdra_field_is_visible_to_logging_helpers():
    from treetune.gear.logging_helpers import aggregate_tree_stats

    tree = {
        "gear_algorithm_mode": "budget_allocation",
        "gear_segment_id": "root",
        "vdra_default_k": 6,
        "vdra_predicted_k": 3,
        "vdra_allocated_k": 3,
        "children": [{"gear_segment_id": "c"} for _ in range(3)],
    }
    out = aggregate_tree_stats(tree)
    assert node_allocated_k(tree) == 3
    assert out["gear/budget/allocated_node_budget"] == 3.0


def test_pruning_trace_identities():
    node = _node("a", 6, 3, 0.5)
    allocate_branch_factors([node], total_budget=6, n_min=1)
    validate_node_accounting(node, k_min=1)
    assert node["vdra_cap_k"] == 3
    assert node["vdra_base_k"] == 3
    assert node["vdra_saved_k"] == 3
    assert node["vdra_unmet_demand"] == 0
    assert node["vdra_additional_k"] == 0
    assert node["vdra_allocated_k"] == 3
    assert node["vdra_reserve_contribution"] == 3


def test_redistribution_trace_transfers_saved_budget_to_unmet_demand():
    nodes = [_node("A", 6, 2, 0.1), _node("B", 6, 10, 1.0)]
    summary = allocate_branch_factors(nodes, total_budget=12, n_min=1)
    assert nodes[0]["vdra_saved_k"] == 4
    assert nodes[1]["vdra_unmet_demand"] == 4
    assert nodes[1]["vdra_additional_k"] == 4
    assert summary.allocations["B"] <= nodes[1]["vdra_predicted_k"]
    assert sum(n["vdra_saved_k"] for n in nodes) == 4
    assert sum(n["vdra_additional_k"] for n in nodes) == 4


def test_capped_allocation_cannot_exceed_unmet_demand():
    node = _node("hot", 6, 7, 1e9)
    allocate_branch_factors([node], total_budget=100, n_min=1)
    validate_node_accounting(node, k_min=1)
    assert node["vdra_unmet_demand"] == 1
    assert node["vdra_additional_k"] == 1
    assert node["vdra_allocated_k"] == 7


def test_minimum_branch_factor_caps_zero_prediction():
    node = _node("cold", 6, 0, 0.0)
    allocate_branch_factors([node], total_budget=6, n_min=1)
    validate_node_accounting(node, k_min=1)
    assert node["vdra_cap_k"] == 1
    assert node["vdra_base_k"] == 1
    assert node["vdra_allocated_k"] >= 1


def test_queue_zero_timeout_does_not_emit_timeout_flush():
    async def run():
        manager = RootQueueManager(
            queue_count=1,
            queue_capacity=8,
            timeout_seconds=0.0,
            reserve_pool=SharedReservePool(queue_count=1),
            policy_snapshot_id="p0",
        )
        manager.enqueue(OnlineQueueItem(_node("a", 1, 2, 1.0), 1, 0, policy_snapshot_id="p0"), now=1.0)
        return manager, await manager.flush_ready(now=999.0)

    manager, flushed = asyncio.run(run())
    assert flushed == []
    assert manager.timeout_flush_count == 0


def test_queue_final_drain_reason():
    async def run():
        manager = RootQueueManager(
            queue_count=1,
            queue_capacity=8,
            timeout_seconds=99.0,
            reserve_pool=SharedReservePool(queue_count=1),
            policy_snapshot_id="p0",
        )
        manager.enqueue(OnlineQueueItem(_node("a", 1, 2, 1.0), 1, 0, policy_snapshot_id="p0"), now=1.0)
        return manager, await manager.drain(now=1.2)

    manager, flushed = asyncio.run(run())
    assert flushed[0].flush_reason == "final_drain"
    assert manager.final_drain_count == 1


def test_allocation_timing_wraps_allocator_call(monkeypatch):
    import vdra_core.online_budget as online_budget

    original = online_budget.allocate_branch_factors

    def slow_allocator(*args, **kwargs):
        time.sleep(0.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(online_budget, "allocate_branch_factors", slow_allocator)

    async def run():
        manager = RootQueueManager(
            queue_count=1,
            queue_capacity=1,
            timeout_seconds=99.0,
            reserve_pool=SharedReservePool(queue_count=1),
            policy_snapshot_id="p0",
        )
        manager.enqueue(OnlineQueueItem(_node("a", 1, 2, 1.0), 1, 0, policy_snapshot_id="p0"), now=1.0)
        return await manager.flush_ready(now=1.0)

    flushed = asyncio.run(run())
    assert flushed[0].allocation_seconds >= 0.02
    assert flushed[0].to_record()["allocation_seconds"] >= 0.02


def test_default_verl_config_passes_strict_startup_invariants():
    import yaml

    cfg_path = Path(__file__).resolve().parents[1] / "verl" / "recipe" / "gear_tree" / "config" / "gear_tree_trainer.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    gear_tree = cfg["gear_tree"]
    gear = gear_tree["gear"]
    tree_shape = gear_tree["tree_shape"]
    assert gear["tv_first_phase_tokens"] <= gear_tree["segment_length"]
    assert gear["pilot_branch_factor"] > max(tree_shape)
    assert gear["queue_timeout_seconds"] > 0
    assert gear["root_allocation"] is False
    assert gear["allocation_scope"] == "one_tree"


def test_calibration_artifact_round_trips_through_strict_loader(tmp_path):
    from argparse import Namespace

    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_tail_divergence.py"
    spec = importlib.util.spec_from_file_location("vdra_calibration_roundtrip", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = Namespace(
        model="m", checkpoint="ckpt", dataset="math", k0=8, r=2,
        first_phase_tokens=60, short_horizon=60, full_tokens=512,
        quantile=0.99, seed=0, horizons=[60],
    )
    metadata = module.build_metadata(args, module.selected_runtime_horizon(args))
    artifact = {
        "metadata": metadata,
        "args": vars(args),
        "summary": {
            "per_horizon": {
                "60": {
                    "eps_tail_quantiles": {"0.99": 0.12},
                    "eps_tail_by_depth": {"0": {"0.99": 0.1}},
                }
            }
        },
        "records": [],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = load_tail_calibration(
        str(path), model="m", checkpoint="ckpt", dataset="math",
        pilot_branch_factor=8, likelihood_samples_per_distribution=2,
        short_horizon=60, quantile=0.99, strict_metadata=True,
    )
    assert loaded["eps_tail"] == pytest.approx(0.12)
    for kwargs in [
        {"pilot_branch_factor": 7},
        {"likelihood_samples_per_distribution": 3},
        {"short_horizon": 32},
    ]:
        base = dict(
            model="m", checkpoint="ckpt", dataset="math",
            pilot_branch_factor=8, likelihood_samples_per_distribution=2,
            short_horizon=60, quantile=0.99, strict_metadata=True,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            load_tail_calibration(str(path), **base)


def test_compute_accounting_consistency_summary():
    tree = {
        "gear_algorithm_mode": "budget_allocation",
        "gear_segment_id": "root",
        "vdra_default_k": 4,
        "vdra_allocated_k": 4,
        "vdra_pilot_generated_tokens": 3,
        "vdra_main_expansion_generated_tokens": 7,
        "vdra_generation_request_count": 2,
        "vdra_likelihood_scored_prompt_tokens": 5,
        "vdra_likelihood_scored_continuation_tokens": 2,
        "children": [],
    }
    summary = summarize_vdra_tree(tree)
    assert summary["vdra_total_generated_tokens"] == 10
    assert summary["vdra_generation_decode_tokens"] == 10
    assert summary["vdra_total_scored_tokens"] == 7
    assert summary["vdra_generation_request_count"] == 2
    assert summary["vdra_token_equivalent_compute_proxy"] == 17
    assert "vdra_generation_forward_calls" not in summary
    assert "vdra_total_model_forward_calls" not in summary


def test_persist_vdra_artifacts_writes_canonical_files(tmp_path):
    node = _node("root", 6, 8, 1.0)
    allocate_branch_factors([node], total_budget=6, n_min=1)
    node.update({
        "depth": 0,
        "vdra_pilot_children_generated": 8,
        "vdra_pilot_children_reused": 2,
        "vdra_pilot_children_discarded": 6,
        "vdra_pilot_generated_tokens": 24,
        "vdra_main_expansion_generated_tokens": 4,
        "vdra_total_generated_tokens": 28,
        "vdra_likelihood_scored_prompt_tokens": 3,
        "vdra_likelihood_scored_continuation_tokens": 5,
        "vdra_total_scored_tokens": 8,
        "children": [{} for _ in range(node["vdra_allocated_k"])],
    })
    persist_vdra_artifacts(
        tmp_path, node, run_id="r", tree_id="t",
        queue_flushes=[{"queue_id": 0, "flush_reason": "timeout"}],
        run_manifest={"algorithm_executed": "test"},
    )
    assert (tmp_path / "nodes.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads((tmp_path / "queue_flushes.jsonl").read_text(encoding="utf-8"))["flush_reason"] == "timeout"
    assert json.loads((tmp_path / "compute_summary.json").read_text(encoding="utf-8"))["token_equivalent_compute_proxy"] == 36
    assert json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))["algorithm_executed"] == "test"



def test_reserve_draw_is_capped_by_total_unmet_demand():
    async def run():
        pool = SharedReservePool(queue_count=1)
        await pool.add(10)
        manager = RootQueueManager(
            queue_count=1,
            queue_capacity=1,
            timeout_seconds=99.0,
            reserve_pool=pool,
            policy_snapshot_id="p0",
        )
        node = _node("hot", 1, 4, 1.0)
        manager.enqueue(OnlineQueueItem(node, 1, 0, policy_snapshot_id="p0"), now=0.0)
        return pool, await manager.flush_ready(now=0.0)

    pool, flushed = asyncio.run(run())
    assert flushed[0].reserve_draw == 3
    assert flushed[0].reserve_consumed == 3
    assert pool.contributed == 10
    assert pool.consumed == 3
    assert pool.value == 7


def test_strict_calibration_requires_metadata_and_matching_k0(tmp_path):
    from vdra_core.calibration import load_tail_calibration

    missing_meta = tmp_path / "missing.json"
    missing_meta.write_text(json.dumps({"summary": {"per_horizon": {"60": {"eps_tail_quantiles": {"0.99": 0.2}}}}}))
    with pytest.raises(ValueError, match="metadata"):
        load_tail_calibration(str(missing_meta), pilot_branch_factor=8, likelihood_samples_per_distribution=2, short_horizon=60)

    artifact = tmp_path / "cal.json"
    artifact.write_text(json.dumps({
        "metadata": {
            "model": "m",
            "checkpoint": "c",
            "dataset": "d",
            "pilot_branch_factor": 4,
            "likelihood_samples_per_distribution": 2,
            "short_horizon": 60,
            "quantile": 0.99,
        },
        "summary": {"per_horizon": {"60": {"eps_tail_quantiles": {"0.99": 0.23}}}},
        "records": [],
    }))
    with pytest.raises(ValueError, match="pilot_branch_factor"):
        load_tail_calibration(str(artifact), pilot_branch_factor=8, likelihood_samples_per_distribution=2, short_horizon=60)
    loaded = load_tail_calibration(str(artifact), pilot_branch_factor=4, likelihood_samples_per_distribution=2, short_horizon=60)
    assert loaded["eps_tail"] == 0.23
