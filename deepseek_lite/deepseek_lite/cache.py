"""
cache.py — Bộ nhớ đệm (KV cache).

Vấn đề: khi viết một câu, máy viết từng chữ một.
Viết chữ thứ 100 mà phải tính lại K và V của cả 99 chữ trước thì rất phí.
Mà K và V của những chữ cũ thì không bao giờ thay đổi.

Giải pháp: tính một lần, cất vào đệm, dùng lại mãi.

    Không có đệm:  viết 200 chữ = 200 lần tính cả câu        (chậm dần)
    Có đệm:        viết 200 chữ = 1 lần tính cả câu + 199 lần tính 1 chữ

Đệm chỉ chứa K và V, không chứa Q.
Vì Q là câu hỏi của token MỚI; token cũ hỏi xong rồi thì thôi.

Và vì dùng GQA (n_kv_heads < n_heads), đệm chỉ cần chứa n_kv_heads đầu
thay vì n_heads đầu. Đó là lý do thật người ta dùng GQA.
"""

from __future__ import annotations

import torch

from .config import Config


class KVCache:
    def __init__(self, config: Config, batch_size: int = 1, device=None) -> None:
        self.n_layers = config.n_layers
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.max_len = config.max_seq_len

        shape = (batch_size, self.n_kv_heads, self.max_len, self.head_dim)

        # Mỗi tầng có một cặp K, V riêng.
        self.keys = [torch.zeros(shape, device=device) for _ in range(self.n_layers)]
        self.values = [torch.zeros(shape, device=device) for _ in range(self.n_layers)]

        # Đã cất được bao nhiêu token rồi.
        self.length = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Cất K, V của token mới vào đệm. Trả về TOÀN BỘ K, V từ đầu tới giờ."""
        T = k.shape[-2]
        end = self.length + T

        if end > self.max_len:
            raise ValueError(
                f"Đệm chỉ chứa được {self.max_len} token (đang muốn cất tới {end}). "
                "Tăng max_seq_len trong config.py nếu cần dài hơn."
            )

        self.keys[layer][:, :, self.length : end] = k
        self.values[layer][:, :, self.length : end] = v

        # Chỉ trả về phần đã dùng, không trả về phần còn trống.
        return self.keys[layer][:, :, :end], self.values[layer][:, :, :end]

    def advance(self, T: int) -> None:
        """Sau khi MỌI tầng đã cất xong thì mới tăng độ dài."""
        self.length += T

    def reset(self) -> None:
        """Xoá đệm để bắt đầu một câu mới."""
        self.length = 0

    @property
    def memory_bytes(self) -> int:
        """Đệm tốn bao nhiêu byte. K và V đều là float32 (4 byte)."""
        per_layer = 2 * self.n_kv_heads * self.max_len * self.head_dim * 4
        return per_layer * self.n_layers

    def __repr__(self) -> str:
        mb = self.memory_bytes / 1024 / 1024
        return f"KVCache(length={self.length}/{self.max_len}, {mb:.2f} MB)"
