"""
model.py — Ghép tất cả lại thành một LLM hoàn chỉnh.

    Chữ
     |  tokenizer
     v
    Số  [12, 45, 91, 7]
     |  embedding: mỗi số -> một vector
     v
    [B, T, d_model]
     |
     |  Block x n_layers      <-- chỗ này lặp lại nhiều lần
     |    RMSNorm -> Attention
     |    RMSNorm -> MoE
     |
     v
    RMSNorm
     |
     v
    LM Head: vector -> xác suất của MỌI chữ trong từ điển
     |
     v
    Chữ tiếp theo

Vậy là hết. Toàn bộ một LLM nằm trong file này.
"""

import torch
import torch.nn as nn

from block import Block, RMSNorm
from mhc import MultiStreamBlock


class DeepSeekMini(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # 1. Bảng tra: mỗi chữ có một vector riêng.
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 2. Các tầng suy nghĩ.
        #    n_streams = 1 -> Block thường (dễ hiểu).
        #    n_streams = 4 -> bản mini của mHC trong mhc.py.
        BlockClass = MultiStreamBlock if config.n_streams > 1 else Block
        self.blocks = nn.ModuleList([BlockClass(config) for _ in range(config.n_layers)])

        # 3. Chuẩn hoá lần cuối trước khi phát biểu.
        self.norm = RMSNorm(config.d_model)

        # 4. LM Head: biến vector thành điểm cho từng chữ.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: dùng chung bảng tra với LM Head.
        # Một bảng học một lần, dùng ở hai chỗ -> đỡ tốn và học nhanh hơn.
        self.lm_head.weight = self.embedding.weight

        # Khởi tạo trọng số thật nhỏ, để lúc đầu model "đoán bừa nhẹ nhàng".
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        # tokens: [B, T] toàn số nguyên
        x = self.embedding(tokens)

        if self.config.n_streams > 1:
            # Nhiều dòng suy nghĩ: bắt đầu bằng n bản giống hệt nhau.
            streams = [x for _ in range(self.config.n_streams)]
            for block in self.blocks:
                streams = block(streams)
            # Cuối cùng trộn n dòng lại thành một câu trả lời.
            x = sum(streams) / self.config.n_streams
        else:
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)
        return self.lm_head(x)

    def aux_loss(self):
        """Điểm phạt nếu model dồn hết việc cho một chuyên gia."""
        return sum(block.moe.aux_loss for block in self.blocks)
