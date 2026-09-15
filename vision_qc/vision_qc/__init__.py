"""
vision_qc — Kiểm tra lỗi sản phẩm bằng camera 2MP, CNN huấn luyện từ đầu,
điều khiển PLC Mitsubishi loại sản phẩm lỗi ra khỏi dây chuyền.

Bốn câu hỏi, bốn nhóm file:

    Ảnh ở đâu ra?        camera.py · capture.py
    Dạy model thế nào?   dataset.py · augment.py · model.py · train.py
    Quyết định ra sao?   evaluate.py
    Nói với PLC thế nào? mc.py · plc.py · station.py
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import CameraConfig, Config, LineConfig, ModelConfig, PlcConfig
from .paths import CLASSES, RAW_DIR, RUNS_DIR, ensure_dirs

__all__ = [
    "__version__",
    "Config",
    "CameraConfig",
    "LineConfig",
    "PlcConfig",
    "ModelConfig",
    "CLASSES",
    "RAW_DIR",
    "RUNS_DIR",
    "ensure_dirs",
]
