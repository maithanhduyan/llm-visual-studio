"""
tensor_parallel.py — Cắt NGANG: một phép nhân bị chẻ ra nhiều máy.

Ý tưởng: một ma trận trọng số to thì cắt đôi được, mà cắt kiểu gì thì
kết quả vẫn phải y hệt như không cắt. Có đúng hai kiểu cắt:

    CỘT (ColumnParallel)          HÀNG (RowParallel)
    cắt theo chiều RA             cắt theo chiều VÀO

    W: [out, in]                  W: [out, in]
    máy i giữ W[i]: [out/tp, in]  máy i giữ W[i]: [out, in/tp]

    y_i = x @ W_iᵀ + b_i          y_i = x_i @ W_iᵀ
    (x nguyên vẹn, y bị cắt)      (x bị cắt, y là một phần của tổng)
         |                              |
    không cần nói chuyện          phải CỘNG lại: all_reduce -> y đầy đủ

Ghép hai kiểu lại thành một tầng hoàn chỉnh:

    x (nguyên vẹn)
      -> ColumnParallel  ->  hoạt hoá (bị cắt)  ->  RowParallel  ->  all_reduce
                                                                      |
                                                            x (nguyên vẹn) trở lại

Nhờ vậy chỉ cần nói chuyện với nhau ĐÚNG MỘT LẦN cho mỗi tầng, chứ không
phải sau mỗi phép nhân.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel import Parallel


def divide(numerator: int, denominator: int) -> int:
    """Chia phải hết, không thì báo lỗi ngay chứ đừng âm thầm làm tròn."""
    if numerator % denominator != 0:
        raise ValueError(
            f"{numerator} không chia hết cho {denominator}. "
            "Chọn số máy chia hết cho số đầu (head) và số chuyên gia."
        )
    return numerator // denominator


class ColumnParallelLinear(nn.Module):
    """Cắt theo chiều ra. Đầu ra bị chia cho các máy, đầu vào thì không."""

    def __init__(self, in_features: int, out_features: int, parallel: Parallel, bias: bool = True):
        super().__init__()

        self.parallel = parallel
        self.in_features = in_features
        self.out_features = out_features
        self.out_per_partition = divide(out_features, parallel.tp_size)

        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_per_partition)) if bias else None

        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, parallel: Parallel) -> "ColumnParallelLinear":
        """Cắt một nn.Linear có sẵn ra. Mỗi máy lấy phần của mình."""
        module = cls(
            linear.in_features,
            linear.out_features,
            parallel,
            bias=linear.bias is not None,
        ).to(linear.weight.device, linear.weight.dtype)

        start = parallel.tp_rank * module.out_per_partition
        end = start + module.out_per_partition

        with torch.no_grad():
            module.weight.copy_(linear.weight[start:end])
            if module.bias is not None:
                module.bias.copy_(linear.bias[start:end])

        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Cắt theo chiều vào. Đầu ra đầy đủ, nhưng phải cộng lại giữa các máy."""

    def __init__(self, in_features: int, out_features: int, parallel: Parallel, bias: bool = True):
        super().__init__()

        self.parallel = parallel
        self.in_features = in_features
        self.out_features = out_features
        self.in_per_partition = divide(in_features, parallel.tp_size)

        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition))

        # Chú ý: bias chỉ có MỘT bản, và chỉ được cộng SAU khi all_reduce.
        # Nếu máy nào cũng cộng thì bias bị nhân lên tp lần.
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, parallel: Parallel) -> "RowParallelLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            parallel,
            bias=linear.bias is not None,
        ).to(linear.weight.device, linear.weight.dtype)

        start = parallel.tp_rank * module.in_per_partition
        end = start + module.in_per_partition

        with torch.no_grad():
            module.weight.copy_(linear.weight[:, start:end])
            if module.bias is not None:
                module.bias.copy_(linear.bias)

        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x, self.weight, None)

        # Cộng phần của mọi máy lại thành kết quả đầy đủ.
        out = self.parallel.all_reduce(partial, dim="tp")

        if self.bias is not None:
            out = out + self.bias

        return out


class VocabParallelEmbedding(nn.Module):
    """Bảng tra từ vựng bị cắt: mỗi máy giữ một đoạn các ký tự.

    Token nào không thuộc đoạn của mình thì máy đó trả về 0, rồi all_reduce
    cộng lại — nên chỉ máy giữ ký tự đó đóng góp.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, parallel: Parallel):
        super().__init__()

        self.parallel = parallel
        self.num_embeddings = num_embeddings

        per_partition = math.ceil(num_embeddings / parallel.tp_size)
        self.start = parallel.tp_rank * per_partition
        self.end = min(self.start + per_partition, num_embeddings)
        self.size = self.end - self.start

        self.weight = nn.Parameter(torch.empty(self.size, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding, parallel: Parallel) -> "VocabParallelEmbedding":
        module = cls(embedding.num_embeddings, embedding.embedding_dim, parallel).to(
            embedding.weight.device, embedding.weight.dtype
        )
        with torch.no_grad():
            module.weight.copy_(embedding.weight[module.start : module.end])
        return module

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mine = (tokens >= self.start) & (tokens < self.end)

        # Token không thuộc phần của mình -> tra tạm dòng 0 rồi nhân với 0.
        local = (tokens - self.start).masked_fill(~mine, 0).clamp(0, max(self.size - 1, 0))

        out = F.embedding(local, self.weight) * mine.unsqueeze(-1).to(self.weight.dtype)

        return self.parallel.all_reduce(out, dim="tp")


# ======================================================================
# Cắt cả model
# ======================================================================


def parallelize_swiglu(module: nn.Module, parallel: Parallel) -> None:
    """Cắt một khối SwiGLU: cửa và nội dung theo cột, đầu ra theo hàng."""
    module.gate = ColumnParallelLinear.from_linear(module.gate, parallel)
    module.value = ColumnParallelLinear.from_linear(module.value, parallel)
    module.out = RowParallelLinear.from_linear(module.out, parallel)


def parallelize_attention(module: nn.Module, parallel: Parallel) -> None:
    """Cắt một khối Attention.

    Q, K, V cắt theo cột (mỗi máy giữ vài đầu), đầu ra cắt theo hàng.
    Số đầu trên mỗi máy cũng phải giảm theo, nếu không lúc reshape sẽ sai.
    """
    module.n_heads = divide(module.n_heads, parallel.tp_size)
    module.n_kv_heads = divide(module.n_kv_heads, parallel.tp_size)

    module.q_proj = ColumnParallelLinear.from_linear(module.q_proj, parallel)
    module.k_proj = ColumnParallelLinear.from_linear(module.k_proj, parallel)
    module.v_proj = ColumnParallelLinear.from_linear(module.v_proj, parallel)
    module.out_proj = RowParallelLinear.from_linear(module.out_proj, parallel)


def tensor_parallelize(model: nn.Module, parallel: Parallel) -> nn.Module:
    """Cắt model của Cấp 3 ra `tp` máy.

    Cắt những chỗ TO (attention và các chuyên gia). Bảng embedding và LM Head
    thì để nguyên trên mọi máy — ở model này chúng chỉ có 145 dòng nên cắt
    ra cũng chẳng tiết kiệm được gì, mà lại phải gom logits về mỗi lần tính
    loss. Hệ thật cắt cả bảng đó (xem docs/song-song.md).
    """
    if parallel.tp_size == 1:
        return model

    for block in model.blocks:
        parallelize_attention(block.attention, parallel)

        moe = block.moe

        # Router để nguyên: mọi máy phải chọn CÙNG một chuyên gia, nếu không
        # các máy sẽ tính những thứ khác nhau rồi all_reduce ra kết quả sai.
        for expert in moe.experts:
            parallelize_swiglu(expert, parallel)

        for expert in moe.shared_experts:
            parallelize_swiglu(expert, parallel)

    return model
