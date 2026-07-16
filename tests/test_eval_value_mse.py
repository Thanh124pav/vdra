"""CPU tests for the RQ5/RQ6 aggregation (no server / no HF model needed)."""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rq5 = _load("_test_rq5", "eval_value_mse.py")
rq6 = _load("_test_rq6", "eval_gradient_quality.py")


def _pool(rng, p, n=64):
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


def _records():
    rng = random.Random(7)
    # Low-dispersion nodes (near-deterministic pools) and one hard node
    # (p=0.5, maximal Bernoulli variance). c_s mirrors the true dispersion.
    return [
        {"node_id": "easy1", "rewards": _pool(rng, 0.98), "c_s": 0.005},
        {"node_id": "easy2", "rewards": _pool(rng, 0.02), "c_s": 0.005},
        {"node_id": "hard", "rewards": _pool(rng, 0.5), "c_s": 0.5},
    ]


def test_allocation_budget_is_preserved_with_floor():
    allocations = rq5.allocate_by_weights(
        {"a": 0.0, "b": 1.0, "c": 3.0}, budget=12, n_min=1
    )
    assert sum(allocations.values()) == 12
    assert min(allocations.values()) >= 1
    assert allocations["c"] > allocations["b"] > allocations["a"]


def test_method_weights():
    rec = {"node_id": "n", "rewards": [0.0, 1.0, 0.0, 1.0], "c_s": 0.09}
    assert rq5.method_weight("uniform", rec, k0=4) == 1.0
    assert rq5.method_weight("vdra", rec, k0=4) == pytest.approx(0.3)
    assert rq5.method_weight("oracle", rec, k0=4) == pytest.approx(0.5)
    assert rq5.method_weight("empirical_variance", rec, k0=4) == pytest.approx(0.5)
    r1 = rq5.method_weight("random", rec, k0=4)
    assert 0.0 < r1 <= 1.0
    assert r1 == rq5.method_weight("random", rec, k0=4)  # seeded per node


def test_variance_aware_allocations_beat_uniform_on_mse():
    summary = rq5.evaluate_value_mse(
        _records(), default_bf=6, n_min=1, seeds=200, k0=8
    )
    per = summary["per_method"]
    assert summary["budget"] == 18
    for method in rq5.METHODS:
        assert per[method]["mse_v"] is not None
    # Oracle knows the true pool dispersion; VDRA's c_s mirrors it here.
    assert per["oracle"]["mse_v"] < per["uniform"]["mse_v"]
    assert per["vdra"]["mse_v"] < per["uniform"]["mse_v"]
    assert per["uniform"]["mse_ratio_vs_uniform"] == pytest.approx(1.0)
    # The hard node received the extra branches under the oracle allocation.
    assert per["oracle"]["allocations"]["hard"] > per["oracle"]["allocations"]["easy1"]


def test_rq6_pure_helpers():
    assert rq6.flat_cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert rq6.flat_cosine([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert rq6.flat_cosine([0.0], [1.0]) is None
    assert rq6.l2_sq([1.0, 2.0], [0.0, 0.0]) == pytest.approx(5.0)

    pool = [0.0, 1.0, 1.0, 0.0]
    rng_a = random.Random("s")
    rng_b = random.Random("s")
    assert rq6.subsample_value(pool, 2, rng_a) == rq6.subsample_value(pool, 2, rng_b)
    assert rq6.subsample_value(pool, 99, random.Random(0)) == pytest.approx(
        sum(pool) / len(pool), abs=0.5
    )
    # tanh TV identity check (§9).
    import math

    assert rq6.tanh_tv([math.log(0.2)], [math.log(0.1)]) == pytest.approx(
        (0.2 - 0.1) / (0.2 + 0.1)
    )
