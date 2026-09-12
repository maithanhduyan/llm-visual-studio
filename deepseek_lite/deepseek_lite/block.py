"""
block.py — Một tầng suy nghĩ gồm những gì?

Một tầng chỉ có hai việc:

    1. Attention  -> trao đổi thông tin với các token khác
    2. MoE        -> tự mình suy nghĩ nhờ các chuyên gia

Khác Cấp 2 ở chỗ: thay vì `x = x + việc`, ở đây dùng hyper-connections
(xem mhc.py) để có nhiều dòng suy nghĩ song song.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import Attention
from .cache import KVCache
from .config import Config
from .mhc import HyperConnections
from .moe import MoE


class RMSNorm(nn.Module):
    """
    Giữ cho các con số không quá to cũng không quá nhỏ.
    Giống như chỉnh âm lượng loa trước khi phát nhạc.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Một con số học được cho mỗi ô, lúc đầu bằng 1.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chia x cho "độ to trung bình" của chính nó.
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return (x / rms) * self.weight


class Block(nn.Module):
    def __init__(self, config: Config, layer_index: int) -> None:
        super().__init__()

        self.norm1 = RMSNorm(config.d_model)
        self.attention = Attention(config, layer_index)

        self.norm2 = RMSNorm(config.d_model)
        self.moe = MoE(config)

        self.dropout = nn.Dropout(config.dropout)

        # Bộ trộn các dòng suy nghĩ.
        self.hyper = HyperConnections(config.n_streams)

    def forward(
        self,
        streams: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        # streams: [B, T, n_streams, C]

        # Việc 1: các token nói chuyện với nhau.
        h = self.hyper.read(streams)
        streams = self.hyper.write(
            streams,
            self.dropout(self.attention(self.norm1(h), cache, start_pos)),
        )

        # Việc 2: suy nghĩ riêng nhờ các chuyên gia.
        h = self.hyper.read(streams)
        streams = self.hyper.write(streams, self.dropout(self.moe(self.norm2(h))))

        return streams
