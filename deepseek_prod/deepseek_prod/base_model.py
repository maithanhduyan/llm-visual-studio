"""
base_model.py — Lấy model của Cấp 3 ra dùng.

Cấp 4 cố tình KHÔNG viết lại model. Viết lại thì vừa thừa vừa sai:
Cấp 4 không phải là "model khác", mà là "cách chạy model đó ở quy mô lớn".

Nên file này chỉ làm một việc: tìm và nạp package `deepseek_lite`.
Nếu nó chưa được cài (`pip install -e .`), ta tự thêm thư mục bên cạnh
vào đường dẫn tìm kiếm.
"""

from __future__ import annotations

import sys

from .paths import LEVEL3_ROOT


def ensure_level3_importable() -> None:
    """Bảo đảm `import deepseek_lite` chạy được."""
    try:
        import deepseek_lite  # noqa: F401, PLC0415

        return
    except ModuleNotFoundError:
        pass

    if not LEVEL3_ROOT.exists():
        raise ModuleNotFoundError(
            f"Không tìm thấy Cấp 3 ở {LEVEL3_ROOT}.\n"
            "Cấp 4 dùng lại model của Cấp 3, nên cần thư mục đó nằm cạnh.\n"
            "Hoặc cài đặt:  pip install -e ../deepseek_lite"
        )

    path = str(LEVEL3_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


ensure_level3_importable()

from deepseek_lite import Config, DeepSeekLite, KVCache, Tokenizer  # noqa: E402

__all__ = ["Config", "DeepSeekLite", "KVCache", "Tokenizer", "ensure_level3_importable"]
