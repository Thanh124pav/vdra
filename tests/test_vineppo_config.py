from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vineppo_overlay_matches_upstream_empty_delimiter_configuration():
    overlay = (ROOT / "configs/algorithms/vineppo.libsonnet").read_text()

    assert "type: 'vineppo_episode_generator'" in overlay
    assert "reasoning_step_delimiter: ''" in overlay


def test_deepseek_r1_qwen_vineppo_math_reuses_tuned_spo_chain_base():
    config = (ROOT / "configs/polIter_deepseekR1Qwen_vineppo_MATH.jsonnet").read_text()

    assert "import 'polIter_deepseekR1Qwen_spo_chain_MATH.jsonnet'" in config
    assert "import 'algorithms/vineppo.libsonnet'" in config
