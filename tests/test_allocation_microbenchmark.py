import importlib.util
import math
from pathlib import Path


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_allocation_solver.py"
    spec = importlib.util.spec_from_file_location("benchmark_allocation_solver", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_allocation_microbenchmark_contract():
    metrics = _load_benchmark_module().run_benchmark(rounds=5)
    assert metrics["allocation/queue_size"] == 32.0
    assert metrics["allocation/target_budget"] == 512.0
    assert metrics["allocation/increment_steps"] == 480.0
    assert metrics["allocation/solver_time_ms_median"] >= 0.0
    assert math.isfinite(metrics["allocation/solver_time_ms_p99"])
