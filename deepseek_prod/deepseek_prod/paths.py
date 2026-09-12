"""
paths.py — Mọi đường dẫn của dự án nằm ở đúng một chỗ.

Cấp 4 đứng CẠNH Cấp 3 và dùng lại model của Cấp 3, nên ở đây có thêm
đường dẫn sang thư mục đó.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

# Model của Cấp 3 — Cấp 4 không viết lại model, chỉ dùng lại.
LEVEL3_ROOT = REPO_ROOT / "deepseek_lite"
LEVEL3_CHECKPOINT = LEVEL3_ROOT / "runs" / "model.pt"

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_DATA_FILE = DATA_DIR / "data.txt"
DEFAULT_CHECKPOINT = RUNS_DIR / "model.pt"


def ensure_run_dir() -> Path:
    """Tạo thư mục đầu ra nếu chưa có."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR
