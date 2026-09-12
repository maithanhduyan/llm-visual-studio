"""
attention.py — Trái tim của Transformer.

Cách dễ nhất để hiểu Attention:

    Q = Question  (tôi đang cần tìm thông tin gì?)
    K = Key       (tôi có thông tin gì?)
    V = Value     (thông tin của tôi là gì?)

Mỗi token tự hỏi: "Trong các token đứng trước, token nào giống
với câu hỏi của mình nhất?" Rồi nó lấy thông tin của token đó.

Ví dụ: "Hôm nay trời rất ___"
    Token "trời" sẽ hỏi và tìm thấy "Hôm nay" và "rất".
    Nhờ vậy nó đoán được chữ tiếp theo là "đẹp".

Chú ý: is_causal=True nghĩa là token chỉ được nhìn về PHÍA TRƯỚC.
Nếu nhìn được cả tương lai thì lúc học nó sẽ "chép bài".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import rope


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.d_model = config.d_model

        # Mỗi đầu rộng bao nhiêu ô.
        self.head_dim = config.d_model // config.n_heads

        # Một phép nhân duy nhất sinh ra cả Q, K, V cho nhanh.
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)

        # Sau khi các đầu "bàn xong", trộn lại thành một câu trả lời.
        self.out = nn.Linear(config.d_model, config.d_model)

        # Chỗ để lưu bản đồ attention khi bật config.capture (xem studio/).
        self.capture = config.capture
        self.attention_map = None

    def forward(self, x):
        B, T, C = x.shape  # B câu, T token, C ô

        # 1. Sinh ra Q, K, V rồi tách thành nhiều đầu.
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)  # [B, H, T, head_dim]

        # 2. Xoay q và k theo vị trí, để model biết thứ tự các token.
        q = rope(q)
        k = rope(k)

        # 3. Cho các token "hỏi nhau". PyTorch làm phép tính này rất nhanh.
        if self.capture:
            # Đây chính là công thức gốc: softmax(Q·Kᵀ / √d).
            # Hàm ở dưới cũng làm y hệt, nhưng nhanh hơn và không trả về
            # bản đồ attention. Ta tính tay ở đây chỉ để vẽ hình.
            score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            self.attention_map = score.masked_fill(future, float("-inf")).softmax(-1).detach()

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 4. Ghép các đầu lại và trộn về đúng kích thước ban đầu.
        y = y.transpose(1, 2).reshape(B, T, C)

        return self.out(y)
