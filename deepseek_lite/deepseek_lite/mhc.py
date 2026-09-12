"""
mhc.py — Nếu AI có nhiều dòng suy nghĩ song song?

Cấp 2 chỉ có MỘT dòng:

    x -------------------------------> x + việc

Cấp 3 có n dòng chạy song song:

    dòng 1 --\\
    dòng 2 ----> trộn lại thành 1 --> Attention/MoE --> tản ra n lại
    dòng 3 ---->                                            |
    dòng 4 --/                                              v
             ^----------------------------------- cộng vào từng dòng

Mỗi dòng có "tiếng nói" riêng khi góp ý (alpha), và "độ to" riêng khi nhận
kết quả trả về (beta). Model tự học hai con số đó.

Đây là bản đơn giản của hyper-connections. Bản mHC thật trong hình có
thêm ràng buộc hình học giữa các dòng — phức tạp hơn nhiều, để Cấp 4.

Khi n_streams = 1 thì alpha = 1 và beta = 1, nên nó trở về đúng đường tắt
bình thường của Cấp 2. Nhờ vậy bật/tắt không cần sửa code ở đâu khác.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HyperConnections(nn.Module):
    def __init__(self, n_streams: int) -> None:
        super().__init__()

        self.n = n_streams

        # alpha: đọc n dòng thành 1 dòng để đưa cho Attention/MoE.
        # Chia đều lúc đầu: mỗi dòng góp 1/n tiếng nói.
        self.alpha = nn.Parameter(torch.full((n_streams,), 1.0 / n_streams))

        # beta: ghi kết quả trả về cho từng dòng. Lúc đầu = 1 (như Cấp 2).
        self.beta = nn.Parameter(torch.ones(n_streams))

    def read(self, streams: torch.Tensor) -> torch.Tensor:
        """Trộn n dòng [B, T, n, C] thành 1 dòng [B, T, C]."""
        return sum(self.alpha[i] * streams[:, :, i, :] for i in range(self.n))

    def write(self, streams: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """Tản kết quả [B, T, C] về lại n dòng, mỗi dòng nhận một lượng khác nhau."""
        parts = [streams[:, :, i, :] + self.beta[i] * result for i in range(self.n)]
        return torch.stack(parts, dim=2)

    def extra_repr(self) -> str:
        return f"n_streams={self.n}"
