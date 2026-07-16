import pytest

from vdra_core.proxies import select_dispersion_proxy


def test_vdra_uniform_and_direct_tv_proxies():
    pair_tvs = {(0, 1): 0.2, (0, 2): 0.4, (1, 2): 0.1}
    assert select_dispersion_proxy(
        "vdra", vdra_dispersion_C=0.7, pair_tvs=pair_tvs, pilot_count=3, node={}
    ) == 0.7
    assert select_dispersion_proxy(
        "uniform", vdra_dispersion_C=0.7, pair_tvs=pair_tvs, pilot_count=3, node={}
    ) == 1.0
    assert select_dispersion_proxy(
        "direct_tv", vdra_dispersion_C=0.7, pair_tvs=pair_tvs, pilot_count=3, node={}
    ) == pytest.approx((0.2**2 + 0.4**2 + 0.1**2) / 9)


def test_random_proxy_is_seeded_per_node_and_positive():
    kwargs = dict(vdra_dispersion_C=0.7, pair_tvs={}, pilot_count=3)
    node_a = {"vdra_node_id": "a"}
    node_b = {"vdra_node_id": "b"}
    first = select_dispersion_proxy("random", node=node_a, **kwargs)
    again = select_dispersion_proxy("random", node=node_a, **kwargs)
    other = select_dispersion_proxy("random", node=node_b, **kwargs)
    assert first == again  # deterministic per node id
    assert 0.0 < first <= 1.0
    assert 0.0 < other <= 1.0
    assert first != other  # non-uniform across nodes


@pytest.mark.parametrize(
    ("method", "field"),
    [
        ("empirical_variance", "vdra_empirical_reward_variance"),
        ("external_score", "vdra_external_dispersion_C"),
        ("oracle", "vdra_oracle_value_dispersion"),
    ],
)
def test_evaluation_proxies_require_explicit_node_fields(method, field):
    with pytest.raises(ValueError, match="requires node field"):
        select_dispersion_proxy(
            method, vdra_dispersion_C=0.0, pair_tvs={}, pilot_count=2, node={}
        )
    assert select_dispersion_proxy(
        method,
        vdra_dispersion_C=0.0,
        pair_tvs={},
        pilot_count=2,
        node={field: 0.25},
    ) == 0.25
