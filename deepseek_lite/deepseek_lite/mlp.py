"""
mlp.py — AI suy nghĩ bên trong thế nào?

SwiGLU, giống Cấp 2. Khác một chỗ: độ rộng bên trong được truyền vào,
vì Cấp 3 dùng nhiều chuyên gia NHỎ (fine-grained) thay vì ít chuyên gia to.

    Input
      |
      +--> Gate  --> Silu -->  \\        (cửa mở bao nhiêu?)
      |                        x  --> Out
      +--> Value ----------->  /        (nội dung là gì?)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()

        self.gate = nn.Linear(d_model, hidden)  # cửa
        self.value = nn.Linear(d_model, hidden)  # nội dung
        self.out = nn.Linear(hidden, d_model)  # thu lại cho gọn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate(x))
        value = self.value(x)

        # Nhân hai thứ với nhau: cửa mở thì nội dung mới đi qua.
        return self.out(gate * value)
