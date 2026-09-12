"""
block.py — Một tầng suy nghĩ của AI gồm những gì?

Một tầng chỉ có hai việc, lặp đi lặp lại:

    1. Attention  -> trao đổi thông tin với các token khác
    2. MoE        -> tự mình suy nghĩ nhờ các chuyên gia

Và quan trọng nhất: ĐƯỜNG TẮT (residual).

    x = x + <việc vừa làm>

Đường tắt này là bí quyết giúp model 100 tầng vẫn học được.
Nhờ nó, mỗi tầng chỉ cần học "sửa thêm một chút" chứ không
phải học lại từ đầu. Giống như làm bài: giữ bài cũ, ghi thêm ý mới.
"""

import torch
import torch.nn as nn

from attention import Attention
from moe import MoE


class RMSNorm(nn.Module):
    """
    Giữ cho các con số không quá to cũng không quá nhỏ.

    Giống như chỉnh âm lượng loa trước khi phát nhạc.
    """

    def __init__(self, dim):
        super().__init__()
        # Một con số học được cho mỗi ô, lúc đầu bằng 1.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Chia x cho "độ to trung bình" của chính nó.
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return (x / rms) * self.weight


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = RMSNorm(config.d_model)
        self.attention = Attention(config)

        self.norm2 = RMSNorm(config.d_model)
        self.moe = MoE(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Việc 1: các token nói chuyện với nhau, rồi ghi thêm vào bài cũ.
        x = x + self.dropout(self.attention(self.norm1(x)))

        # Việc 2: suy nghĩ riêng, rồi lại ghi thêm vào bài cũ.
        x = x + self.dropout(self.moe(self.norm2(x)))

        return x
