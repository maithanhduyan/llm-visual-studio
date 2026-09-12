"""
rope.py — Làm sao AI biết token nào đứng trước, token nào đứng sau?

Nếu chỉ đưa các token vào mà không nói thứ tự, thì
"con chó cắn con mèo" và "con mèo cắn con chó" giống hệt nhau.

RoPE (Rotary Position Embedding) giải quyết bằng cách
XOAY các vector theo một góc phụ thuộc vào vị trí.

Ý tưởng cho bé:
    Vị trí 0 xoay 0 độ.
    Vị trí 1 xoay một chút.
    Vị trí 2 xoay nhiều hơn.
    ... nhờ vậy hai token càng xa nhau thì càng "khác pha".

Điều hay nhất: RoPE chỉ xoay q và k, không xoay v.
"""

import torch


def rope(x):
    """
    x có hình dạng [B, H, T, D]:
        B = số câu (batch)
        H = số đầu (head)
        T = số token trong câu
        D = độ rộng mỗi đầu (head_dim)

    Trả về tensor cùng hình dạng, nhưng đã được xoay theo vị trí.
    """
    B, H, T, D = x.shape
    half = D // 2

    # position = 0, 1, 2, ... T-1
    position = torch.arange(T, device=x.device, dtype=torch.float32)

    # Tần số: cặp đầu tiên xoay nhanh, cặp cuối xoay rất chậm.
    freq = torch.arange(half, device=x.device, dtype=torch.float32)
    freq = 1.0 / (10000 ** (freq / half))

    # Góc xoay của từng token ở từng cặp chiều: [T, half]
    angle = position[:, None] * freq[None, :]

    cos = angle.cos()[None, None, :, :]  # [1, 1, T, half]
    sin = angle.sin()[None, None, :, :]

    # Cắt đôi vector rồi xoay như xoay một mũi tên trên giấy.
    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
