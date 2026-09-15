"""paths.py — Mọi đường dẫn nằm ở một chỗ."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

LEVEL3_ROOT = REPO_ROOT / "deepseek_lite"

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
FLIGHTS_DIR = PROJECT_ROOT / "flights"
VIEWER_DIR = PROJECT_ROOT / "viewer"


def ensure_dirs() -> None:
    for folder in (DATA_DIR, RUNS_DIR, FLIGHTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
