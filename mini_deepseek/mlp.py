"""
mlp.py — AI "suy nghĩ" bên trong thế nào?

Attention chỉ lo việc các token trao đổi thông tin với nhau.
Còn MLP là chỗ model thật sự "tính toán" một mình.

Bản này dùng SwiGLU — đúng loại mà DeepSeek dùng:

    Input
      |
      +--> Gate  --> Silu -->  \\        (cửa mở bao nhiêu?)
      |                        ˣ  --> Out
      +--> Value ----------->  /        (nội dung là gì?)

Cửa (gate) quyết định cho bao nhiêu thông tin đi qua.
Đó là lý do nó giỏi hơn MLP thường một chút.
"""

import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # Bên trong rộng hơn bên ngoài 4 lần: có chỗ để "suy nghĩ".
        hidden = d_model * 4

        self.gate = nn.Linear(d_model, hidden)   # cửa
        self.value = nn.Linear(d_model, hidden)  # nội dung
        self.out = nn.Linear(hidden, d_model)    # thu lại cho gọn

    def forward(self, x):
        gate = F.silu(self.gate(x))  # Silu: số âm -> gần 0, số dương -> giữ
        value = self.value(x)

        # Nhân hai thứ với nhau: cửa mở thì nội dung mới đi qua.
        return self.out(gate * value)
