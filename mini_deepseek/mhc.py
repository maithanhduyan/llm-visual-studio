"""
mhc.py — Nếu AI có 4 "dòng suy nghĩ" song song thì sao?

Trong hình, mHC có "4 parallel residual streams".
Ở block.py, mỗi lúc chỉ có MỘT dòng:

    x ---------------------------------> x + việc

Còn ở đây, ta giữ 4 dòng:

    dòng 1 --\\
    dòng 2 ----> trộn lại thành 1 --> Attention/MoE --> tách ra 4 dòng
    dòng 3 ---->                                              |
    dòng 4 --/                                                v
             ^------------------------------------- cộng lại

Mỗi dòng có "tiếng nói" riêng (read) khi góp ý, và "độ to" riêng
(write) khi nhận kết quả trả về. Model tự học các con số đó.

    LƯU Ý: Đây là bản mini để hiểu ý tưởng, KHÔNG phải bản mHC
    chính thức trong hệ thống thật. Công thức thật phức tạp hơn
    nhiều (có ràng buộc hình học giữa các dòng).
"""

import torch
import torch.nn as nn

from attention import Attention
from block import RMSNorm
from moe import MoE


class MultiStreamBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n = config.n_streams  # số dòng suy nghĩ

        self.norm1 = RMSNorm(config.d_model)
        self.attention = Attention(config)

        self.norm2 = RMSNorm(config.d_model)
        self.moe = MoE(config)

        self.dropout = nn.Dropout(config.dropout)

        # Cho mỗi dòng một chút "cá tính" ban đầu.
        # Nếu tất cả giống hệt nhau, 4 dòng sẽ mãi mãi giống hệt nhau.
        noise = lambda: 0.01 * torch.randn(self.n)
        self.read1 = nn.Parameter(torch.full((self.n,), 1.0 / self.n) + noise())
        self.read2 = nn.Parameter(torch.full((self.n,), 1.0 / self.n) + noise())

        # Độ to khi nhận kết quả trả về, lúc đầu = 1 (giống block thường).
        self.write1 = nn.Parameter(torch.ones(self.n))
        self.write2 = nn.Parameter(torch.ones(self.n))

    def _read(self, streams, weight):
        """Trộn n dòng thành 1 dòng để đưa cho Attention/MoE."""
        return sum(weight[i] * streams[i] for i in range(self.n))

    def _write(self, streams, result, weight):
        """Trả kết quả về lại cho từng dòng, mỗi dòng nhận một lượng khác nhau."""
        for i in range(self.n):
            streams[i] = streams[i] + weight[i] * result
        return streams

    def forward(self, streams):
        # streams là một list gồm n tensor, mỗi tensor [B, T, C]
        x = self._read(streams, self.read1)
        streams = self._write(streams, self.dropout(self.attention(self.norm1(x))), self.write1)

        x = self._read(streams, self.read2)
        streams = self._write(streams, self.dropout(self.moe(self.norm2(x))), self.write2)

        return streams
