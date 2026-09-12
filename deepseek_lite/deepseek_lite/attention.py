"""
attention.py — AI nhìn vào từ nào? (bản Cấp 3)

Ba thứ mới so với Cấp 2:

1. GQA (Grouped-Query Attention)
      Cấp 2:  6 đầu Q, 6 đầu K, 6 đầu V.
      Cấp 3:  6 đầu Q, nhưng chỉ 2 đầu K và 2 đầu V.
      Ba đầu Q dùng chung một cặp K/V.
   -> Đệm K/V nhỏ đi 3 lần, mà chất lượng gần như không giảm.

2. Cửa sổ trượt (sliding window)
      Token chỉ nhìn lại `window` token gần nhất, không nhìn cả câu.
      Đọc câu dài thì em cũng chỉ cần nhớ đoạn vừa đọc.

3. KV cache
      Token mới chỉ tính Q của chính nó, rồi tra K/V đã cất trong đệm.
      Xem cache.py. Nhờ nó mà viết chữ thứ 100 nhanh gần bằng chữ thứ nhất.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVCache
from .config import Config
from .rope import RoPE


class Attention(nn.Module):
    def __init__(self, config: Config, layer_index: int) -> None:
        super().__init__()

        self.layer_index = layer_index
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.window = config.window

        self.capture = config.capture
        self.attention_map = None

        # GQA: Q rộng, K và V hẹp hơn.
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.rope = RoPE(self.head_dim, config.max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Sinh Q, K, V. Chú ý K và V hẹp hơn Q (GQA).
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Xoay theo vị trí. start_pos để sinh chữ thứ 100 biết nó ở vị trí 100.
        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)

        # 3. Cất K, V mới vào đệm, rồi lấy ra TOÀN BỘ K, V từ đầu tới giờ.
        if cache is not None:
            k, v = cache.append(self.layer_index, k, v)

        # 4. Cửa sổ trượt: bỏ hẳn những K/V quá cũ đi, không chỉ che chúng.
        #
        #    Che bằng mặt nạ thì vẫn tốn công tính (ma trận vẫn T x S).
        #    Cắt hẳn đi thì mới nhanh thật.
        #
        #    Phải cắt TRƯỚC bước 5. Nếu nhân bản GQA rồi mới cắt thì
        #    vẫn phải copy toàn bộ K/V cũ, và cắt còn CHẬM HƠN không cắt.
        #
        #    Chỉ cắt được khi T = 1 (đang sinh từng chữ), vì lúc đó token
        #    duy nhất kia chỉ cần nhìn lại `window` token gần nhất. Còn lúc
        #    học (T lớn) thì token đầu tiên trong đoạn cần nhìn xa hơn.
        k_start = 0
        if cache is not None and self.window > 0 and T == 1:
            available = k.shape[-2]
            if available > self.window:
                k = k[:, :, -self.window :].contiguous()
                v = v[:, :, -self.window :].contiguous()
                k_start = available - self.window

        # 5. GQA: ba đầu Q dùng chung một cặp K/V -> nhân bản K, V lên cho đủ.
        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        S = k.shape[-2]  # số token được nhìn tới

        # Khi phục vụ nhiều câu cùng lúc (Cấp 4), mỗi câu dài một khác nên
        # đệm phải đệm thêm cho bằng nhau. `lengths` cho biết đâu là token
        # thật, đâu là đệm lót — phần đệm phải bị che đi.
        lengths = getattr(cache, "lengths", None) if cache is not None else None

        mask = self._mask(T, S, start_pos, x.device, k_start, lengths)

        # 6. Lưu bản đồ attention nếu đang mở studio/.
        if self.capture:
            score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            score = score.masked_fill(~mask, float("-inf"))
            self.attention_map = score.softmax(-1).detach()

        # 7. Cho các token "hỏi nhau".
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # Ghép các đầu lại. Dùng n_heads * head_dim chứ không dùng C: khi
        # model bị cắt ngang ra nhiều máy (Cấp 4), mỗi máy chỉ giữ một phần
        # số đầu nên đầu ra ở đây hẹp hơn C. Lúc không cắt thì hai số bằng nhau.
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.out_proj(y)

    def _mask(
        self,
        T: int,
        S: int,
        start_pos,
        device,
        k_start: int = 0,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ô nào được nhìn, ô nào không.

        Được nhìn khi thỏa CẢ HAI:
            - đứng trước hoặc bằng mình    (không nhìn tương lai)
            - cách mình không quá `window` (cửa sổ trượt)

        lengths: nếu có, là tensor [B] cho biết mỗi câu dài bao nhiêu token.
                 Các cột từ `lengths[b]` trở đi là đệm lót, phải che đi.
        """
        if lengths is not None:
            if not torch.is_tensor(start_pos):
                # Đệm chia khối cho biết độ dài từng câu, nên vị trí cũng
                # phải tính theo từng câu.
                start_pos = torch.full(
                    (lengths.shape[0],), int(start_pos), dtype=torch.long, device=device
                )

            q_pos = start_pos[:, None] + torch.arange(T, device=device)[None, :]  # [B, T]
            k_pos = (k_start + torch.arange(S, device=device))[None, :]  # [1, S]

            allowed = k_pos[None, :, :] <= q_pos[:, :, None]  # [B, T, S]

            if self.window > 0:
                allowed &= k_pos[None, :, :] > (q_pos[:, :, None] - self.window)

            allowed &= k_pos[None, :, :] < lengths[:, None, None]

            return allowed[:, None, :, :]  # [B, 1, T, S]

        q_pos = torch.arange(start_pos, start_pos + T, device=device)  # [T]
        k_pos = torch.arange(k_start, k_start + S, device=device)  # [S]

        allowed = k_pos[None, :] <= q_pos[:, None]

        if self.window > 0:
            allowed &= k_pos[None, :] > (q_pos[:, None] - self.window)

        return allowed  # [T, S], True = được nhìn
