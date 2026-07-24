"""Rollout-endpoint/model-id resolution must survive a briefly unresponsive server.

The vLLM http server is a Ray async actor whose event loop is blocked while it sleeps,
wakes and syncs weights. A single short-timeout probe can therefore fail on a healthy
server; when the served model id is lost that way, the scorer falls back to the
``/models`` HTTP probe against the same blocked actor and the run dies with
"Could not resolve scorer_model from the vLLM /models endpoint".
"""

import sys
from types import SimpleNamespace

import pytest

from recipe.gear_tree import async_tree_rollout as atr


class _Remote:
    def __init__(self, ref):
        self._ref = ref

    def remote(self):
        return self._ref


class _ServerHandle:
    get_server_address = _Remote("address-ref")
    get_model_id = _Remote("model-ref")


class _Manager:
    server_handles = [_ServerHandle()]


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    atr._ROLLOUT_MODEL_ID_CACHE.clear()
    monkeypatch.setattr(atr.time, "sleep", lambda _: None)
    yield
    atr._ROLLOUT_MODEL_ID_CACHE.clear()


def _install_ray(monkeypatch, get_fn):
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=get_fn))


def _capture_warnings(monkeypatch):
    """Collect warning-level log lines (the module logger bypasses caplog's handler)."""
    records: list[str] = []
    monkeypatch.setattr(
        atr.logger, "warning", lambda msg, *args: records.append(msg % args if args else msg)
    )
    return records


def test_model_id_probe_retries_transient_timeout(monkeypatch):
    attempts = {"model": 0}

    def _get(ref, timeout=None):
        if ref == "address-ref":
            return "10.0.0.5", 12345
        attempts["model"] += 1
        if attempts["model"] < 3:
            raise TimeoutError("actor event loop busy")
        return "served-model"

    _install_ray(monkeypatch, _get)

    api_base, model_id = atr._resolve_rollout_server_info(_Manager())

    assert api_base == "http://10.0.0.5:12345/v1"
    assert model_id == "served-model"
    assert attempts["model"] == 3


def test_resolved_model_id_is_cached_across_calls(monkeypatch):
    calls = {"model": 0}

    def _get(ref, timeout=None):
        if ref == "address-ref":
            return "10.0.0.5", 12345
        calls["model"] += 1
        return "served-model"

    _install_ray(monkeypatch, _get)

    assert atr._resolve_rollout_server_info(_Manager())[1] == "served-model"
    assert atr._resolve_rollout_server_info(_Manager())[1] == "served-model"
    assert calls["model"] == 1


def test_attach_uses_cached_model_id_when_probe_is_down(monkeypatch):
    """A later step must not lose scorer_model just because the actor is busy then."""
    state = {"model_up": True}

    def _get(ref, timeout=None):
        if ref == "address-ref":
            return "10.0.0.5", 12345
        if not state["model_up"]:
            raise TimeoutError("actor event loop busy")
        return "served-model"

    _install_ray(monkeypatch, _get)
    base_cfg = {"enabled": True, "scorer_uses_rollout_server": True}

    first = atr._attach_rollout_scorer_endpoint(dict(base_cfg), _Manager())
    assert first["scorer_model"] == "served-model"

    state["model_up"] = False
    second = atr._attach_rollout_scorer_endpoint(dict(base_cfg), _Manager())
    assert second["scorer_api_base"] == "http://10.0.0.5:12345/v1"
    assert second["scorer_model"] == "served-model"


def test_persistent_probe_failure_leaves_model_unset(monkeypatch):
    def _get(ref, timeout=None):
        if ref == "address-ref":
            return "10.0.0.5", 12345
        raise TimeoutError("actor event loop busy")

    _install_ray(monkeypatch, _get)
    records = _capture_warnings(monkeypatch)

    api_base, model_id = atr._resolve_rollout_server_info(_Manager())

    assert api_base == "http://10.0.0.5:12345/v1"
    assert model_id is None
    # The degradation must be visible in the log instead of silently falling back.
    logged = "\n".join(records)
    assert "get_model_id" in logged
    assert "actor event loop busy" in logged


def test_scorer_model_error_names_the_underlying_cause():
    from recipe.gear_tree.gear_core.gear import vllm_scorer

    with pytest.raises(RuntimeError, match="ConnectError|ConnectTimeout|Connect"):
        vllm_scorer.resolve_vllm_model_id("http://127.0.0.1:1/v1", None, timeout=0.05)
