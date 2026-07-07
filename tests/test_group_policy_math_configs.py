import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"


def test_deepseek_grpo_math_uses_128_sibling_rollouts():
    config = (CONFIGS / "polIter_deepseekR1Qwen_grpo_MATH.jsonnet").read_text()

    assert re.search(r"local\s+num_rollouts_per_sample\s*=\s*128\s*;", config)
    assert "samples: num_rollouts_per_sample" in config
    assert "dataset_num_samples_per_iteration: num_dataset_samples_per_iteration" in config
    assert "adv_method: 'grpo'" in config


def test_deepseek_rloo_math_reuses_the_128_rollout_grpo_config():
    config = (CONFIGS / "polIter_deepseekR1Qwen_rloo_MATH.jsonnet").read_text()

    assert "import 'polIter_deepseekR1Qwen_grpo_MATH.jsonnet'" in config
    assert "import 'algorithms/rloo.libsonnet'" in config


def test_math_launchers_default_to_deepseek_r1_qwen():
    for algorithm in ("grpo", "rloo"):
        script = (SCRIPTS / f"train_{algorithm}_MATH.sh").read_text()

        assert 'MODEL="${MODEL:-deepseekR1Qwen}"' in script
        assert f'resolve_math_config {algorithm} "${{MODEL}}"' in script
