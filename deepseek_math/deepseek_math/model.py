"""
model.py — Model dùng cho việc học toán.

Vẫn là kiến trúc của Cấp 3 (GQA, MoE, hyper-connections), chỉ thu nhỏ lại
và đổi bộ từ vựng từ 145 ký tự tiếng Việt xuống 14 ký tự số.

    d_model      96      (Cấp 3: 192)
    n_layers      3
    chuyên gia    4      (Cấp 3: 8)
    max_seq_len  64      (bài dài nhất 41 ký tự)

Vì sao vẫn giữ MoE: để xem các chuyên gia có tự chia nhau việc không. Một
chuyên gia lo hàng đơn vị, một chuyên gia lo hàng chục chẳng hạn.
"""

from __future__ import annotations

from .base import Config, DeepSeekLite

MATH_CONFIG = dict(
    d_model=96,
    n_heads=4,
    n_kv_heads=2,
    n_layers=3,
    max_seq_len=64,
    window=0,  # bài ngắn, không cần cửa sổ trượt
    n_experts=4,
    top_k=2,
    n_shared=1,
    expert_hidden=64,
    n_streams=2,
)


def build_model(vocab_size: int, **overrides) -> DeepSeekLite:
    settings = {**MATH_CONFIG, **overrides}
    return DeepSeekLite(Config(vocab_size=vocab_size, **settings))
