"""
base.py — Lấy kiến trúc model của Cấp 3 ra dùng.

Bộ ra quyết định cấp cao cũng chỉ là một model sinh chữ. Không cần kiến trúc
mới, chỉ cần thu nhỏ lại: ở đây nó chỉ phải đọc bốn con số và chọn một trong
bốn lệnh, dễ hơn làm toán nhiều.
"""

from __future__ import annotations

import sys

from .paths import LEVEL3_ROOT


def ensure_level3_importable() -> None:
    try:
        import deepseek_lite  # noqa: F401, PLC0415

        return
    except ModuleNotFoundError:
        pass

    if not LEVEL3_ROOT.exists():
        raise ModuleNotFoundError(
            f"Không tìm thấy Cấp 3 ở {LEVEL3_ROOT}.\n"
            "Dự án này dùng lại kiến trúc của Cấp 3."
        )

    path = str(LEVEL3_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


ensure_level3_importable()

from deepseek_lite import Config, DeepSeekLite, Tokenizer  # noqa: E402

__all__ = ["Config", "DeepSeekLite", "Tokenizer"]
