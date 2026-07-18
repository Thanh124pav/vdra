"""PLAN.md P0.5: two explicit rollout/scorer endpoint modes."""

from __future__ import annotations

import pytest

from recipe.gear_tree.scorer_verification import (
    fetch_rollout_weight_version,
    resolve_endpoints,
)


def _fetcher(versions):
    def fake_fetch(api_base, *, api_key="EMPTY", timeout=5.0):
        return versions.get(str(api_base))

    return fake_fetch


def test_same_server_explicit_mode_passes():
    versions = {"http://server:9000/v1": "abc123"}
    gear_cfg = {
        "scorer_uses_rollout_server": True,
        "rollout_api_base": None,
        "scorer_api_base": "http://server:9000/v1",
        "strict_vdra": True,
    }
    assert fetch_rollout_weight_version(gear_cfg, fetch_fn=_fetcher(versions)) == "abc123"


def test_same_server_forbids_distinct_rollout_api_base():
    gear_cfg = {
        "scorer_uses_rollout_server": True,
        "rollout_api_base": "http://server-other:9000/v1",
        "scorer_api_base": "http://server:9000/v1",
        "strict_vdra": True,
    }
    with pytest.raises(ValueError, match="scorer_uses_rollout_server"):
        resolve_endpoints(gear_cfg)


def test_two_server_matching_versions_pass():
    versions = {
        "http://rollout:9000/v1": "abc123",
        "http://scorer:9001/v1": "abc123",
    }
    gear_cfg = {
        "scorer_uses_rollout_server": False,
        "rollout_api_base": "http://rollout:9000/v1",
        "scorer_api_base": "http://scorer:9001/v1",
        "strict_vdra": True,
    }
    assert fetch_rollout_weight_version(gear_cfg, fetch_fn=_fetcher(versions)) == "abc123"


def test_two_server_missing_scorer_endpoint_fails_strict():
    gear_cfg = {
        "scorer_uses_rollout_server": False,
        "rollout_api_base": "http://rollout:9000/v1",
        "scorer_api_base": None,
        "strict_vdra": True,
    }
    with pytest.raises(ValueError, match="scorer_api_base"):
        resolve_endpoints(gear_cfg)


def test_two_server_missing_rollout_endpoint_fails_strict():
    gear_cfg = {
        "scorer_uses_rollout_server": False,
        "rollout_api_base": None,
        "scorer_api_base": "http://scorer:9001/v1",
        "strict_vdra": True,
    }
    with pytest.raises(ValueError, match="rollout_api_base"):
        resolve_endpoints(gear_cfg)


def test_two_server_missing_server_fingerprint_fails_strict():
    versions = {}  # server returns no fingerprint
    gear_cfg = {
        "scorer_uses_rollout_server": False,
        "rollout_api_base": "http://rollout:9000/v1",
        "scorer_api_base": "http://scorer:9001/v1",
        "strict_vdra": True,
    }
    with pytest.raises(RuntimeError, match="server-reported weight version"):
        fetch_rollout_weight_version(gear_cfg, fetch_fn=_fetcher(versions))


def test_non_strict_returns_none_when_endpoint_missing():
    gear_cfg = {
        "scorer_uses_rollout_server": False,
        "rollout_api_base": None,
        "scorer_api_base": None,
        "strict_vdra": False,
    }
    assert (
        fetch_rollout_weight_version(gear_cfg, fetch_fn=_fetcher({})) is None
    )
