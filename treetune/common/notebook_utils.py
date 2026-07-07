from pathlib import Path


def get_repo_dir() -> Path:
    # treetune/common/notebook_utils.py -> treetune/common -> treetune -> repo root
    return Path(__file__).parent.parent.parent