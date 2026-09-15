"""Cho pytest tìm thấy package `vision_qc` mà không cần cài đặt gì."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
