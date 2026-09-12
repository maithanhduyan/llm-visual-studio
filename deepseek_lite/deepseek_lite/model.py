"""
model.py — Ghép tất cả lại thành một LLM hoàn chỉnh (Cấp 3).

    Chữ
     |  tokenizer
     v
    Số  [12, 45, 91, 7]
     |  embedding
     v
    [B, T, C]
     |  nhân thành n dòng suy nghĩ        [B, T, n, C]
     |
     |  Block x n_layers
     |    trộn n dòng -> 1
     |    RMSNorm -> Attention (GQA + cửa sổ trượt + KV cache)
     |    tản kết quả về n dòng
     |    trộn n dòng -> 1
     |    RMSNorm -> MoE (chuyên gia dùng chung + chuyên gia riêng)
     |    tản kết quả về n dòng
     |
     v
    gộp n dòng lại
     |
    RMSNorm
     |
    LM Head: vector -> điểm cho MỌI chữ trong từ điển
     |
     v
    Chữ tiếp theo
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block, RMSNorm
from .cache import KVCache
from .config import Config


class DeepSeekLite(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        # 1. Bảng tra: mỗi chữ có một vector riêng.
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 2. Các tầng suy nghĩ.
        self.blocks = nn.ModuleList(
            Block(config, layer_index=i) for i in range(config.n_layers)
        )

        # 3. Chuẩn hoá lần cuối trước khi phát biểu.
        self.norm = RMSNorm(config.d_model)

        # 4. LM Head: biến vector thành điểm cho từng chữ.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: dùng chung bảng tra với LM Head.
        self.lm_head.weight = self.embedding.weight

        # Khởi tạo trọng số thật nhỏ, để lúc đầu model "đoán bừa nhẹ nhàng".
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ):
        """
        tokens:    [B, T] toàn số nguyên
        targets:   [B, T] đáp án, hoặc None nếu chỉ muốn đoán
        cache:     đệm K/V, hoặc None nếu không cần (lúc học)
        start_pos: token đầu tiên đang ở vị trí thứ mấy (lúc sinh chữ)

        Trả về (logits, loss). loss = None nếu không truyền targets.
        """
        B, T = tokens.shape
        n_streams = self.config.n_streams

        x = self.embedding(tokens)  # [B, T, C]

        # Nhân thành n dòng suy nghĩ giống hệt nhau lúc đầu.
        # (expand không copy dữ liệu, chỉ tạo hình dạng mới.)
        streams = x.unsqueeze(2).expand(B, T, n_streams, x.shape[-1])

        for block in self.blocks:
            streams = block(streams, cache, start_pos)

        # Gộp n dòng lại thành một câu trả lời.
        # n_streams = 1 thì phép tính này không đổi gì cả.
        x = streams.mean(dim=2)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # Đệm đã cất xong K/V cho MỌI tầng -> giờ mới được tăng độ dài.
        if cache is not None:
            cache.advance(T)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Mấy hàm phục vụ việc học
    # ------------------------------------------------------------------

    def update_router_bias(self) -> None:
        """Gọi sau mỗi bước học, để các chuyên gia được chia việc đều hơn."""
        for block in self.blocks:
            block.moe.update_bias()

    @torch.no_grad()
    def router_loads(self) -> list[torch.Tensor]:
        """Tỉ lệ chia việc của từng tầng. Chia đều hoàn hảo = 1/n_experts."""
        return [block.moe.load_share() for block in self.blocks]

    def parameter_counts(self) -> tuple[int, int]:
        """Trả về (tổng thông số, thông số thực sự dùng cho mỗi token).

        Hai số này khác nhau là nhờ MoE: model có 8 chuyên gia nhưng mỗi
        token chỉ hỏi 2, nên phần lớn thông số không được dùng một lúc.
        """
        total = sum(p.numel() for p in self.parameters())

        unused = 0
        for block in self.blocks:
            moe = block.moe
            per_expert = sum(p.numel() for p in moe.experts[0].parameters())
            unused += per_expert * (moe.n_experts - moe.top_k)

        return total, total - unused

    def cache_memory_bytes(self) -> int:
        """Đệm K/V sẽ tốn bao nhiêu byte."""
        return KVCache(self.config).memory_bytes

    def __repr__(self) -> str:
        total, active = self.parameter_counts()
        return (
            f"DeepSeekLite(\n"
            f"  thông số      : {total:,} (mỗi token chỉ dùng {active:,})\n"
            f"  tầng          : {self.config.n_layers}\n"
            f"  đầu Q / K-V   : {self.config.n_heads} / {self.config.n_kv_heads}  (GQA)\n"
            f"  cửa sổ trượt  : {self.config.window or 'không dùng'}\n"
            f"  chuyên gia    : {self.config.n_experts} riêng + {self.config.n_shared} dùng chung,"
            f" chọn {self.config.top_k}\n"
            f"  dòng suy nghĩ : {self.config.n_streams}\n"
            f")"
        )
