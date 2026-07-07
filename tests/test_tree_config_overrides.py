import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON_SCRIPT = ROOT / "scripts" / "_common.sh"


def _resolve_tree_config(tmp_path: Path, shape: str, tree_m: str):
    env = os.environ.copy()
    env["APP_DIRECTORY"] = str(tmp_path)
    env["TREE_M"] = tree_m
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; ensure_tree_config "$2"',
            "bash",
            str(COMMON_SCRIPT),
            shape,
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return result


def test_tree_m_overrides_checked_in_tree_config(tmp_path):
    result = _resolve_tree_config(tmp_path, "666", "777")

    assert result.returncode == 0, result.stderr
    tree_config, m_override = result.stdout.strip().split(",")
    assert tree_config.endswith("branch_factor_666.jsonnet")
    assert "M: 777" in Path(m_override).read_text()


def test_tree_m_is_separate_from_generated_shape_cache(tmp_path):
    result = _resolve_tree_config(tmp_path, "3927", "321")

    assert result.returncode == 0, result.stderr
    tree_config, m_override = result.stdout.strip().split(",")
    assert tree_config.endswith("branch_factor_3927.jsonnet")
    assert "M: 321" in Path(m_override).read_text()


def test_tree_m_rejects_invalid_values(tmp_path):
    result = _resolve_tree_config(tmp_path, "666", "auto")

    assert result.returncode != 0
    assert "must be a positive integer" in result.stderr
