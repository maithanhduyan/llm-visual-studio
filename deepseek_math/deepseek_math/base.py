"""
base.py — Lấy model của Cấp 3 ra dùng.

Dự án này KHÔNG viết model mới. Nó dùng lại kiến trúc của Cấp 3, chỉ đổi
bộ từ vựng (14 ký tự số thay vì 145 ký tự tiếng Việt) và thu nhỏ lại.

Việc học ở đây không nằm ở kiến trúc, mà ở DỮ LIỆU và TRÌNH TỰ DẠY.
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
            "Dự án này dùng lại kiến trúc của Cấp 3.\n"
            "Hoặc cài đặt:  pip install -e ../deepseek_lite"
        )

    path = str(LEVEL3_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


ensure_level3_importable()

from deepseek_lite import Config, DeepSeekLite, Tokenizer  # noqa: E402

__all__ = ["Config", "DeepSeekLite", "Tokenizer", "ensure_level3_importable"]
