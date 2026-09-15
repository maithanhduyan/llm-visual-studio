"""
paths.py — Mọi đường dẫn nằm ở một chỗ.

Dự án này đứng cạnh các cấp trước và dùng lại model của Cấp 3, nên có
đường dẫn sang đó.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

LEVEL3_ROOT = REPO_ROOT / "deepseek_lite"

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"


def ensure_run_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR
