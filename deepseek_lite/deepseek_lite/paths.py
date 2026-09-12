"""
paths.py — Mọi đường dẫn của dự án nằm ở đúng một chỗ.

Nhờ file này, các file khác không phải tự đoán xem mình đang ở đâu.
Và nhờ nó, chạy từ thư mục nào cũng đúng.

    deepseek_lite/            <- PROJECT_ROOT
    ├── data/data.txt         <- DEFAULT_DATA_FILE
    ├── runs/model.pt         <- DEFAULT_CHECKPOINT
    └── deepseek_lite/        <- PACKAGE_DIR
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_DATA_FILE = DATA_DIR / "data.txt"
DEFAULT_CHECKPOINT = RUNS_DIR / "model.pt"


def ensure_run_dir() -> Path:
    """Tạo thư mục đầu ra nếu chưa có, rồi trả về đường dẫn."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR
