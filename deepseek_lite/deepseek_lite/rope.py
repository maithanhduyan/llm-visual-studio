"""
rope.py — Làm sao AI biết token nào đứng trước, token nào đứng sau?

RoPE xoay vector của q và k theo một góc phụ thuộc vào vị trí.
Token càng xa nhau thì góc càng lệch, nên model phân biệt được thứ tự.

Cấp 3 khác Cấp 2 ở hai chỗ, cả hai đều để chạy nhanh hơn:

    1. Bảng cos/sin được tính SẴN một lần, không tính lại mỗi lần forward.
    2. Có tham số start_pos, để sinh chữ thứ 100 không phải tính lại
       vị trí của 99 chữ trước (xem cache.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) phải là số chẵn để xoay được theo cặp.")

        half = head_dim // 2

        # Tần số: cặp chiều đầu tiên xoay nhanh, cặp cuối xoay rất chậm.
        freq = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))

        # Góc của từng vị trí: [max_seq_len, half]
        angle = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), freq)

        # Tính sẵn một lần rồi để đó dùng dần.
        # persistent=False: không lưu vào file model, vì tính lại được.
        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        x: [B, H, T, D]

        start_pos: token đầu tiên trong x đang ở vị trí thứ mấy.
            Lúc học thì luôn là 0.
            Lúc sinh chữ thì tăng dần: 0, 1, 2, ...

            Cũng có thể là một tensor [B] — mỗi câu trong lô đang ở một vị
            trí khác nhau. Chuyện này xảy ra khi phục vụ nhiều người cùng
            lúc: câu của người này đã viết được 50 chữ, câu của người kia
            mới được 3 chữ. Cấp 4 (deepseek_prod) cần điều này.
        """
        T = x.shape[-2]

        if torch.is_tensor(start_pos):
            # Mỗi câu một vị trí: dựng bảng vị trí [B, T] rồi tra thẳng.
            positions = start_pos[:, None] + torch.arange(T, device=x.device)[None, :]

            if positions.max().item() >= self.cos.shape[0]:
                raise ValueError(
                    f"Vị trí {int(positions.max()) + 1} vượt quá max_seq_len "
                    f"({self.cos.shape[0]})."
                )

            cos = self.cos[positions][:, None, :, :]
            sin = self.sin[positions][:, None, :, :]
        else:
            if start_pos + T > self.cos.shape[0]:
                raise ValueError(
                    f"Vị trí {start_pos + T} vượt quá max_seq_len ({self.cos.shape[0]}). "
                    "Tăng max_seq_len trong config.py nếu cần câu dài hơn."
                )

            cos = self.cos[start_pos : start_pos + T][None, None, :, :]
            sin = self.sin[start_pos : start_pos + T][None, None, :, :]

        # Cắt đôi vector rồi xoay như xoay một mũi tên trên giấy.
        x1, x2 = x.chunk(2, dim=-1)

        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
