"""
moe.py — AI chọn chuyên gia nào? (bản Cấp 3)

Cấp 2: 4 chuyên gia to, mỗi token hỏi 2, và phải thêm một "hàm phạt" để
bắt model chia việc cho đều.

Cấp 3 thêm hai thứ, đều là chiêu thật của DeepSeek-V3:

1. CHUYÊN GIA DÙNG CHUNG (shared expert)
   Một chuyên gia mà token nào cũng hỏi, không cần router chọn.
   Nó lo những việc cơ bản như giữ cho câu văn trôi chảy.
   Các chuyên gia còn lại chỉ lo việc chuyên môn.

2. CHIA VIỆC KHÔNG CẦN HÀM PHẠT (aux-loss-free load balancing)
   Cấp 2 phạt model nếu chia việc không đều. Nhưng hàm phạt đó bị trộn
   vào loss chính, nên model học kém đi một chút — đúng kiểu "vừa học
   vừa bị nhắc nhở chuyện khác".

   Cấp 3 không đụng vào loss. Thay vào đó mỗi chuyên gia có một
   "điểm thiên vị". Cuối mỗi bước học:
       chuyên gia nào bị chọn QUÁ NHIỀU  -> trừ điểm
       chuyên gia nào bị chọn QUÁ ÍT    -> cộng điểm
   Việc chia đều tự khắc quay lại, mà loss không bị bóp méo.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config
from .mlp import SwiGLU


class MoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.bias_speed = config.router_bias_speed

        # Router: quyết định nên hỏi chuyên gia riêng nào.
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        # Chuyên gia riêng: do router chọn.
        self.experts = nn.ModuleList(
            SwiGLU(config.d_model, config.expert_hidden) for _ in range(config.n_experts)
        )

        # Chuyên gia dùng chung: ai cũng hỏi, không qua router.
        self.shared_experts = nn.ModuleList(
            SwiGLU(config.d_model, config.expert_hidden) for _ in range(config.n_shared)
        )

        # Điểm thiên vị. KHÔNG học bằng gradient — chỉnh tay mỗi bước học.
        self.register_buffer("expert_bias", torch.zeros(config.n_experts))

        # Đếm xem mỗi chuyên gia được chọn bao nhiêu lần trong bước này.
        self.register_buffer("load_count", torch.zeros(config.n_experts))

        self.capture = config.capture
        self.routing = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Chuyên gia dùng chung: ai cũng hỏi, không cần chọn.
        out = sum(expert(x) for expert in self.shared_experts)

        # 2. Router cho điểm từng chuyên gia riêng.
        #    Cộng điểm thiên vị vào trước khi chọn.
        scores = self.router(x) + self.expert_bias  # [B, T, n_experts]
        probs = torch.sigmoid(scores)

        weights, indices = torch.topk(probs, self.top_k, dim=-1)

        # Chuẩn hoá lại để tổng trọng số của mỗi token = 1.
        weights = weights / weights.sum(-1, keepdim=True)

        if self.capture:
            self.routing = (indices.detach(), weights.detach())

        # 3. Đếm lượt chọn, để cuối bước chỉnh lại điểm thiên vị.
        #    Chỉ đếm lúc HỌC. Lúc đánh giá hay sinh chữ thì không đếm,
        #    nếu không bảng thống kê sẽ bị lẫn.
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(indices.reshape(-1), minlength=self.n_experts)
                self.load_count += counts.to(self.load_count.dtype)

        # 4. Hỏi từng chuyên gia được chọn, cộng kết quả theo trọng số.
        routed = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_id = indices[..., k]
            weight = weights[..., k]

            for e, expert in enumerate(self.experts):
                mask = expert_id == e  # token nào chọn chuyên gia e ở vị trí thứ k

                if mask.any():
                    routed[mask] += expert(x[mask]) * weight[mask].unsqueeze(-1)

        return out + routed

    @torch.no_grad()
    def update_bias(self) -> None:
        """Cuối mỗi bước học: chỉnh điểm thiên vị cho công bằng hơn.

        Đây là toàn bộ "thuật toán" chia đều — chỉ có ba dòng.
        """
        total = self.load_count.sum()
        if total == 0:
            return

        load = self.load_count / total  # thực tế: mỗi chuyên gia chiếm bao nhiêu phần
        target = 1.0 / self.n_experts  # mong muốn: chia đều

        # Chuyên gia nào đang nhận nhiều hơn mức đều -> target - load < 0 -> bị trừ điểm.
        self.expert_bias += self.bias_speed * torch.sign(target - load)

        # Xoá bộ đếm để bước sau đếm lại từ đầu.
        self.load_count.zero_()

    @torch.no_grad()
    def load_share(self) -> torch.Tensor:
        """Mỗi chuyên gia đang nhận bao nhiêu phần trăm số lượt chọn.

        Chia đều hoàn hảo thì mọi số đều bằng 1/n_experts.
        """
        total = self.load_count.sum()
        if total == 0:
            return torch.full_like(self.load_count, 1.0 / self.n_experts)
        return self.load_count / total
