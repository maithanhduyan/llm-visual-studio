"""
conftest.py — Mấy thứ dùng chung cho mọi bài test.

Model trong test phải THẬT NHỎ, để chạy vài giây là xong.
Nhưng phải giữ đủ đặc điểm của Cấp 3: GQA, cửa sổ trượt, chuyên gia, nhiều dòng.
"""

import pytest

from deepseek_lite import Config


@pytest.fixture
def small_config() -> Config:
    """Một model tí hon nhưng đủ mọi bộ phận."""
    return Config(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,  # GQA: 2 nhóm
        n_layers=2,
        max_seq_len=64,
        window=0,  # không cửa sổ, để test riêng phần cửa sổ
        n_experts=4,
        top_k=2,
        n_shared=1,
        expert_hidden=16,
        n_streams=2,
    )
