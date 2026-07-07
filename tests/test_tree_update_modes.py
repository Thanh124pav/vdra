import pytest

from treetune.episode_generators.tree_update_modes import compute_tree_update_values


def test_spo_update_mode_matches_existing_local_advantage():
    out = compute_tree_update_values(
        child_reward=0.75,
        parent_reward=0.25,
        root_reward=0.1,
        mode='spo',
    )
    assert out['advantage'] == pytest.approx(0.5)
    assert out['value'] == pytest.approx(0.75)
    assert out['tree_update_local_advantage'] == pytest.approx(0.5)


def test_treepo_original_mixes_local_and_global_advantage():
    out = compute_tree_update_values(
        child_reward=0.9,
        parent_reward=0.4,
        root_reward=0.2,
        mode='treepo_original',
        treepo_global_weight=0.25,
    )
    assert out['advantage'] == pytest.approx(0.75 * 0.5 + 0.25 * 0.7)
    assert out['tree_update_global_advantage'] == pytest.approx(0.7)


def test_treerl_original_uses_dense_td_style_target():
    out = compute_tree_update_values(
        child_reward=0.8,
        parent_reward=0.3,
        root_reward=0.0,
        mode='treerl_original',
        treerl_gamma=0.5,
    )
    assert out['value'] == pytest.approx((0.8 - 0.3) + 0.5 * 0.8)
    assert out['advantage'] == pytest.approx(out['value'] - 0.3)


def test_unknown_tree_update_mode_fails_loudly():
    with pytest.raises(ValueError):
        compute_tree_update_values(
            child_reward=1.0,
            parent_reward=0.0,
            root_reward=0.0,
            mode='unknown',
        )
