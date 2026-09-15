"""
paths.py — Mọi đường dẫn của dự án nằm ở đây, không rải rác trong code.

Cả dự án là một thư mục riêng, cố ý: dữ liệu huấn luyện là ảnh 2MP, vài nghìn
tấm là vài GB. Nó phải nằm tách khỏi các dự án khác trong repo.
"""

from __future__ import annotations

from pathlib import Path

# .../llm-visual-studio/vision_qc
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Ảnh thô thu từ camera, chia theo lớp: data/raw/OK/, data/raw/NG/
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

# Model, log huấn luyện, log vận hành
RUNS_DIR = PROJECT_DIR / "runs"
MODELS_DIR = RUNS_DIR / "models"
LOGS_DIR = RUNS_DIR / "logs"

CLASSES = ("OK", "NG")


def ensure_dirs() -> None:
    """Tạo sẵn cây thư mục. Gọi trước khi ghi bất cứ thứ gì."""
    for path in (DATA_DIR, RAW_DIR, RUNS_DIR, MODELS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for name in CLASSES:
        (RAW_DIR / name).mkdir(parents=True, exist_ok=True)


def class_dir(name: str) -> Path:
    """Thư mục của một lớp. Viết hoa cho khớp tên lớp, giữ nguyên nếu lạ."""
    upper = name.upper()
    return RAW_DIR / (upper if upper in CLASSES else name)
