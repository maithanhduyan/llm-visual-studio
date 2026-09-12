"""
moe.py — AI chọn chuyên gia nào?

Thay vì bắt MỘT người làm hết mọi việc, ta chia ra nhiều chuyên gia:

    Chuyên gia 0 -> giỏi toán
    Chuyên gia 1 -> giỏi code
    Chuyên gia 2 -> giỏi chữ nghĩa
    Chuyên gia 3 -> giỏi kiến thức chung

Mỗi token chỉ hỏi 2 chuyên gia (top_k = 2) thay vì hỏi cả 4.
Nhờ vậy model có thể rất to, mà mỗi lần chạy vẫn nhanh.

    Input
      |
      v
    Router  --> "token này nên hỏi ai?"
      |
      +--> Chuyên gia được chọn --> cộng kết quả lại

Đây là MoE bản GIÁO DỤC: dễ đọc, chạy được trên máy nhỏ.
Bản thật của DeepSeek chia việc này cho hàng nghìn GPU.
"""

import torch
import torch.nn as nn

from mlp import SwiGLU


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.d_model = config.d_model

        # Router: một lớp nhỏ quyết định nên hỏi chuyên gia nào.
        self.router = nn.Linear(config.d_model, config.n_experts)

        # Danh sách các chuyên gia. Mỗi chuyên gia là một MLP riêng.
        self.experts = nn.ModuleList(
            [SwiGLU(config.d_model) for _ in range(config.n_experts)]
        )

        # Chỗ để ghi lại "điểm phạt chia việc không đều" (xem cuối file).
        self.aux_loss = torch.tensor(0.0)

        # Chỗ để lưu "token nào chọn chuyên gia nào" khi bật config.capture.
        self.capture = config.capture
        self.routing = None

    def forward(self, x):
        # x = [B, T, C]
        scores = self.router(x)  # [B, T, n_experts]

        # 1. Chọn ra top_k chuyên gia có điểm cao nhất cho từng token.
        weights, indices = torch.topk(scores, self.top_k, dim=-1)

        # 2. Điểm đó biến thành phần trăm (tổng = 1).
        weights = weights.softmax(dim=-1)

        if self.capture:
            self.routing = (indices.detach(), weights.detach())

        # 3. Hỏi từng chuyên gia và cộng kết quả lại theo phần trăm.
        output = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_id = indices[..., k]  # token này chọn ai ở vị trí thứ k
            weight = weights[..., k]

            for e, expert in enumerate(self.experts):
                mask = expert_id == e  # những token nào chọn chuyên gia e

                if mask.any():
                    # Chỉ đưa cho chuyên gia đúng những token nó phụ trách.
                    output[mask] += expert(x[mask]) * weight[mask].unsqueeze(-1)

        # 4. Chia việc cho công bằng.
        #    Nếu không phạt, model sẽ dồn hết cho 1 chuyên gia
        #    và các chuyên gia còn lại không bao giờ được học.
        probs = scores.softmax(dim=-1)
        chosen = torch.zeros_like(probs).scatter(-1, indices, 1.0)
        self.aux_loss = self.n_experts * (chosen.mean((0, 1)) * probs.mean((0, 1))).sum()

        return output
