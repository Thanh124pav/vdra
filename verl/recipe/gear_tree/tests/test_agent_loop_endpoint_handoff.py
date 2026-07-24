"""The rollout endpoint must reach TreeAgentLoop without probing a busy server.

The manager resolves the endpoint on the driver (where the addresses are already known
from launch) and stamps it on the batch; the worker mirrors it into per-row kwargs;
TreeAgentLoop consumes it. Without that hand-off, construction probes the rollout actor
while it is saturated with generation requests and the run dies with
"no rollout vLLM server endpoint could be resolved from Ray server handles".
"""

from types import SimpleNamespace

import numpy as np
import pytest

from recipe.gear_tree import async_tree_rollout as atr
from verl.experimental.agent_loop import agent_loop as al


class _Batch:
    """Minimal DataProto stand-in: meta_info + non_tensor_batch + len()."""

    def __init__(self, rows=2, meta=None):
        self._rows = rows
        self.meta_info = dict(meta or {})
        self.non_tensor_batch = {}

    def __len__(self):
        return self._rows


class _Manager:
    def __init__(self, addresses, model_id="served-model", fail_model_id=False):
        self.server_addresses = list(addresses)
        self._fail = fail_model_id
        self._model_id = model_id
        self.server_handles = [SimpleNamespace(get_model_id=SimpleNamespace(remote=lambda: "ref"))]


@pytest.fixture
def fake_ray(monkeypatch):
    calls = {"get": 0}

    def _get(ref, timeout=None):
        calls["get"] += 1
        if isinstance(ref, Exception):
            raise ref
        return "served-model"

    monkeypatch.setattr(al.ray, "get", _get)
    return calls


def test_manager_stamps_endpoint_on_batch(fake_ray):
    batch = _Batch()
    manager = _Manager(["10.0.0.5:33109"])

    al._ensure_gear_tree_rollout_endpoint(batch, manager)

    endpoint = batch.meta_info[al.GEAR_TREE_ENDPOINT_META_KEY]
    assert endpoint == {"api_base": "http://10.0.0.5:33109/v1", "model_id": "served-model"}


def test_manager_resolves_endpoint_once_per_process(fake_ray):
    manager = _Manager(["10.0.0.5:33109"])

    al._ensure_gear_tree_rollout_endpoint(_Batch(), manager)
    al._ensure_gear_tree_rollout_endpoint(_Batch(), manager)

    assert fake_ray["get"] == 1


def test_missing_model_id_still_yields_an_endpoint(monkeypatch):
    def _boom(ref, timeout=None):
        raise TimeoutError("actor event loop busy")

    monkeypatch.setattr(al.ray, "get", _boom)
    batch = _Batch()

    al._ensure_gear_tree_rollout_endpoint(batch, _Manager(["10.0.0.5:33109"]))

    endpoint = batch.meta_info[al.GEAR_TREE_ENDPOINT_META_KEY]
    assert endpoint["api_base"] == "http://10.0.0.5:33109/v1"
    assert endpoint["model_id"] is None


def test_worker_mirrors_endpoint_into_per_row_kwargs(fake_ray):
    batch = _Batch(rows=3)
    al._ensure_gear_tree_rollout_endpoint(batch, _Manager(["10.0.0.5:33109"]))

    al._ensure_gear_tree_agent_kwargs(batch, SimpleNamespace())

    assert list(batch.non_tensor_batch["rollout_endpoint_api_base"]) == [
        "http://10.0.0.5:33109/v1"
    ] * 3
    assert list(batch.non_tensor_batch["rollout_endpoint_model_id"]) == ["served-model"] * 3


def test_agent_loop_uses_request_endpoint_without_probing_the_server():
    loop = object.__new__(atr.TreeAgentLoop)
    loop._scorer_endpoint_resolved = False
    loop._gt = {"gear": {"enabled": True, "scorer_uses_rollout_server": True}}
    loop.tokenizer = None
    loop.server_manager = SimpleNamespace()  # no server_handles: a probe would fail
    built = {}

    import recipe.gear_tree.async_tree_rollout as module

    original = module._build_gate
    module._build_gate = lambda gt, tokenizer=None: built.setdefault("gear", gt["gear"])
    try:
        loop._ensure_scorer_endpoint(
            {
                "rollout_endpoint_api_base": "http://10.0.0.5:33109/v1",
                "rollout_endpoint_model_id": "served-model",
            }
        )
    finally:
        module._build_gate = original

    assert loop._scorer_endpoint_resolved
    assert built["gear"]["scorer_api_base"] == "http://10.0.0.5:33109/v1"
    assert built["gear"]["scorer_model"] == "served-model"


def test_agent_loop_still_fails_when_no_endpoint_is_available_anywhere():
    loop = object.__new__(atr.TreeAgentLoop)
    loop._scorer_endpoint_resolved = False
    loop._gt = {"gear": {"enabled": True, "scorer_uses_rollout_server": True}}
    loop.tokenizer = None
    loop.server_manager = SimpleNamespace(server_handles=[])

    with pytest.raises(RuntimeError, match="no rollout vLLM server endpoint"):
        loop._ensure_scorer_endpoint({})


def test_explicit_scorer_model_is_not_overridden_by_the_request():
    loop = object.__new__(atr.TreeAgentLoop)
    loop._scorer_endpoint_resolved = False
    loop._gt = {
        "gear": {
            "enabled": True,
            "scorer_uses_rollout_server": True,
            "scorer_model": "pinned-model",
        }
    }
    loop.tokenizer = None
    loop.server_manager = SimpleNamespace()

    import recipe.gear_tree.async_tree_rollout as module

    original = module._build_gate
    module._build_gate = lambda gt, tokenizer=None: None
    try:
        loop._ensure_scorer_endpoint(
            {
                "rollout_endpoint_api_base": "http://10.0.0.5:33109/v1",
                "rollout_endpoint_model_id": "served-model",
            }
        )
    finally:
        module._build_gate = original

    assert loop._gt["gear"]["scorer_model"] == "pinned-model"


def test_endpoint_columns_are_absent_when_manager_has_no_addresses():
    batch = _Batch()
    al._ensure_gear_tree_rollout_endpoint(batch, _Manager([]))
    assert al.GEAR_TREE_ENDPOINT_META_KEY not in batch.meta_info

    al._ensure_gear_tree_agent_kwargs(batch, SimpleNamespace())
    assert "rollout_endpoint_api_base" not in batch.non_tensor_batch


def test_object_column_shape_matches_rows():
    column = al._object_column("x", 4)
    assert isinstance(column, np.ndarray)
    assert list(column) == ["x"] * 4
