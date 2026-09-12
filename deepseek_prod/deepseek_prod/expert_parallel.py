"""
expert_parallel.py — Mỗi máy giữ vài chuyên gia.

Model này có 8 chuyên gia nhưng mỗi token chỉ hỏi 2. Vậy tại sao máy nào
cũng phải giữ cả 8? Chia ra: máy 0 giữ chuyên gia 0-3, máy 1 giữ 4-7.

    Input (mọi máy đều có, vì dòng residual được nhân bản)
      |
      v
    Router  (chạy trên MỌI máy, cho ra CÙNG một kết quả)
      |
      +--> token chọn chuyên gia 1 -> chỉ máy 0 tính được
      +--> token chọn chuyên gia 6 -> chỉ máy 1 tính được
      |
      v
    máy 0 tính phần của mình        máy 1 tính phần của mình
    (chuyên gia khác thì trả 0)     (chuyên gia khác thì trả 0)
      |                                |
      +----------- all_reduce ---------+
                       |
                       v
        w1 * expert1(x) + w6 * expert6(x)   <- đúng bằng kết quả một máy

Vì sao router phải chạy trên mọi máy và phải cho ra CÙNG kết quả: nếu máy 0
nghĩ token này hỏi chuyên gia 1 còn máy 1 nghĩ nó hỏi chuyên gia 6 thì hai
máy tính hai thứ khác nhau, cộng lại là sai. Nên trọng số router KHÔNG bị
cắt — đó là lý do trong tensor_parallel.py có ghi chú "router để nguyên".

Hệ thật dùng all-to-all để chỉ gửi token tới đúng máy cần, thay vì máy nào
cũng giữ cả bó token rồi trả phần lớn bằng 0. Kết quả y hệt, chỉ khác lượng
dữ liệu phải chuyển. Xem docs/song-song.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .parallel import Parallel
from .tensor_parallel import divide


class ExpertParallelMoE(nn.Module):
    """Bọc một MoE có sẵn, chỉ giữ lại phần chuyên gia của máy mình."""

    def __init__(self, moe: nn.Module, parallel: Parallel) -> None:
        super().__init__()

        self.parallel = parallel
        self.n_experts = moe.n_experts
        self.top_k = moe.top_k
        self.bias_speed = moe.bias_speed

        # Router giữ nguyên trên mọi máy — xem ghi chú đầu file.
        self.router = moe.router
        self.register_buffer("expert_bias", moe.expert_bias)
        self.register_buffer("load_count", moe.load_count)

        # Chỉ giữ phần chuyên gia của mình.
        per_rank = divide(moe.n_experts, parallel.ep_size)
        self.start = parallel.rank_in("ep") * per_rank
        self.end = self.start + per_rank
        self.experts = nn.ModuleList(list(moe.experts)[self.start : self.end])

        # Chuyên gia dùng chung chỉ tính trên máy đầu, nếu không all_reduce
        # sẽ cộng nó lên ep lần.
        self.owns_shared = parallel.rank_in("ep") == 0
        self.shared_experts = moe.shared_experts if self.owns_shared else nn.ModuleList()

        self.capture = getattr(moe, "capture", False)
        self.routing = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Đầu ra của MoE phải GIỐNG HỆT NHAU trên mọi máy, vì nó được cộng
        # vào dòng residual — mà dòng residual thì mọi máy đều giữ một bản.
        # Nếu máy 0 trả "chuyên gia dùng chung + phần của mình" còn máy 1 chỉ
        # trả "phần của mình" thì hai máy sẽ trôi đi hai hướng khác nhau, và
        # mọi tầng sau đó tính trên hai dữ liệu khác nhau. Đây là lỗi đã sập
        # một lần khi làm file này.
        #
        # Nên: gom TẤT CẢ vào một tensor rồi all_reduce, để máy nào cũng ra
        # đúng một kết quả.

        partial = torch.zeros_like(x)

        # 1. Chuyên gia dùng chung: chỉ máy đầu góp, nếu không sẽ bị cộng ep lần.
        if self.owns_shared:
            partial = partial + sum(expert(x) for expert in self.shared_experts)

        # 2. Router: mọi máy chạy, cho ra cùng kết quả.
        scores = self.router(x) + self.expert_bias
        probs = torch.sigmoid(scores)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)

        if self.capture:
            self.routing = (indices.detach(), weights.detach())

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(indices.reshape(-1), minlength=self.n_experts)
                self.load_count += counts.to(self.load_count.dtype)

        # 3. Chỉ tính những chuyên gia MÌNH giữ.
        for k in range(self.top_k):
            expert_id = indices[..., k]
            weight = weights[..., k]

            for local, expert in enumerate(self.experts):
                global_id = self.start + local
                mask = expert_id == global_id

                if mask.any():
                    partial[mask] += expert(x[mask]) * weight[mask].unsqueeze(-1)

        # 4. Cộng phần của mọi máy lại: mỗi token chỉ được đúng những máy
        #    giữ chuyên gia của nó đóng góp, các máy khác góp số 0.
        return self.parallel.all_reduce(partial, dim="ep")

    # Giữ đúng API của MoE gốc, để phần còn lại của chương trình không
    # phải biết model đang chạy thường hay đang chia chuyên gia.
    @torch.no_grad()
    def update_bias(self) -> None:
        total = self.load_count.sum()
        if total == 0:
            return

        load = self.load_count / total
        target = 1.0 / self.n_experts
        self.expert_bias += self.bias_speed * torch.sign(target - load)
        self.load_count.zero_()

    @torch.no_grad()
    def load_share(self) -> torch.Tensor:
        total = self.load_count.sum()
        if total == 0:
            return torch.full_like(self.load_count, 1.0 / self.n_experts)
        return self.load_count / total


def expert_parallelize(model: nn.Module, parallel: Parallel) -> nn.Module:
    """Thay MoE của mọi tầng bằng bản chia chuyên gia ra nhiều máy."""
    if parallel.ep_size == 1:
        return model

    for block in model.blocks:
        block.moe = ExpertParallelMoE(block.moe, parallel)

    return model
