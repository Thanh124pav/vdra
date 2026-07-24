"""Regression tests for ExternalZeroMQDistributedExecutor's collective_rpc signature.

vLLM's engine core calls `execute_model(scheduler_output, non_block=True)` and then
`future.result()`. An executor whose `collective_rpc` rejects `non_block` (or returns a
plain list for it) kills EngineCore on the first scheduled batch.

The executor class lives in a module that imports vllm at load time, which needs a CUDA
runtime, so the methods under test are extracted from the source instead of imported.
"""

import ast
import logging
import pickle
import textwrap
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
METHODS = ("collective_rpc", "execute_model", "sample_tokens")


def _load_executor_methods():
    src = SOURCE.read_text()
    tree = ast.parse(src)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ExternalZeroMQDistributedExecutor"
    )
    body = "\n".join(
        ast.get_source_segment(src, node) for node in cls.body if getattr(node, "name", None) in METHODS
    )
    namespace = {
        "Future": Future,
        "pickle": pickle,
        "logger": logging.getLogger(__name__),
        "Any": Any,
        "Callable": Callable,
        "Optional": Optional,
    }
    exec("class Executor:\n" + textwrap.indent(body, "    "), namespace)  # noqa: S102
    return namespace["Executor"]


@pytest.fixture
def executor():
    cls = _load_executor_methods()
    obj = object.__new__(cls)
    obj._blocking_rpc = lambda method, args=(), kwargs=None: [(method, args)]
    return obj


def test_blocking_rpc_returns_one_output_per_worker(executor):
    assert executor.collective_rpc("load_model") == [("load_model", ())]


def test_non_block_returns_resolved_future(executor):
    future = executor.collective_rpc("load_model", non_block=True)
    assert isinstance(future, Future)
    assert future.result() == [("load_model", ())]


def test_execute_model_unwraps_driver_output(executor):
    assert executor.execute_model("scheduler-output") == ("execute_model", ("scheduler-output",))
    future = executor.execute_model("scheduler-output", non_block=True)
    assert future.result() == ("execute_model", ("scheduler-output",))


def test_sample_tokens_unwraps_driver_output(executor):
    future = executor.sample_tokens(None, non_block=True)
    assert future.result() == ("sample_tokens", (None,))


def test_unique_reply_rank_selects_single_worker(executor):
    executor._blocking_rpc = lambda method, args=(), kwargs=None: ["rank0", "rank1"]
    assert executor.collective_rpc("foo", unique_reply_rank=1) == "rank1"


def test_unknown_kwargs_are_ignored(executor):
    assert executor.collective_rpc("foo", kwargs_added_by_future_vllm=True) == [("foo", ())]


def test_worker_failure_propagates_through_future(executor):
    def boom(*args, **kwargs):
        raise RuntimeError("worker died")

    executor._blocking_rpc = boom
    future = executor.execute_model("scheduler-output", non_block=True)
    with pytest.raises(RuntimeError, match="worker died"):
        future.result()
