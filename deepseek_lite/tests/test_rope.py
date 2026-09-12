"""Test RoPE: xoay thì không được làm vector dài ra, và phải nhất quán theo vị trí."""

import pytest
import torch

from deepseek_lite.rope import RoPE


def test_preserves_length():
    rope = RoPE(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 3, 5, 8)

    out = rope(x)

    # Xoay thì đổi hướng, không đổi độ dài.
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)


def test_different_positions_get_different_rotation():
    rope = RoPE(head_dim=8, max_seq_len=16)

    # Cùng một vector, đặt ở hai vị trí khác nhau -> phải ra hai kết quả khác nhau.
    x = torch.randn(1, 1, 1, 8).expand(1, 1, 4, 8).contiguous()
    out = rope(x)

    assert not torch.allclose(out[0, 0, 0], out[0, 0, 1])


def test_start_pos_matches_full_sequence():
    """Xoay token ở vị trí 5 phải giống hệt xoay cả câu rồi lấy ra token thứ 5.

    Đây là điều kiện để KV cache hoạt động: sinh từng chữ cho ra kết quả
    giống hệt chạy cả câu một lần.
    """
    rope = RoPE(head_dim=8, max_seq_len=32)
    x = torch.randn(1, 2, 10, 8)

    full = rope(x)  # vị trí 0..9
    part = rope(x[:, :, 5:], start_pos=5)  # vị trí 5..9

    assert torch.allclose(full[:, :, 5:], part, atol=1e-5)


def test_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="chẵn"):
        RoPE(head_dim=7, max_seq_len=16)


def test_rejects_too_long_sequence():
    rope = RoPE(head_dim=8, max_seq_len=4)

    with pytest.raises(ValueError, match="max_seq_len"):
        rope(torch.randn(1, 1, 5, 8))
