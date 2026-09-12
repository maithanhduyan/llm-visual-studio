"""
config.py — Các con số của Cấp 4.

Có hai nhóm, đừng lẫn:

    ModelConfig (của Cấp 3)   model to bao nhiêu, mấy tầng, mấy chuyên gia
    ParallelConfig            chia model đó ra bao nhiêu máy, và chia kiểu gì

Cấp 4 chỉ thêm nhóm thứ hai. Nhóm thứ nhất giữ nguyên của Cấp 3 — đó là
toàn bộ ý tưởng của Cấp 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base_model import Config


@dataclass
class ParallelConfig:
    """Chia model ra bao nhiêu phần, và chia theo kiểu gì.

        tp  tensor parallel   cắt NGANG: một phép nhân bị chẻ ra nhiều máy
        pp  pipeline parallel cắt DỌC:  mỗi máy giữ vài tầng
        dp  data parallel     mỗi máy giữ CẢ model, chia nhau dữ liệu
        ep  expert parallel   mỗi máy giữ vài chuyên gia

    Ở đây world_size = tp * pp * dp. Expert parallel dùng chung nhóm TP
    (hệ thật tách riêng một nhóm EP, xem docs/song-song.md).

        zero  ZeRO giai đoạn 1: chia trạng thái optimizer cho các máy trong
              nhóm DP. 0 = tắt.
    """

    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
    zero: int = 0

    # Số vi lô cho pipeline. Càng nhiều thì máy càng đỡ ngồi chơi,
    # nhưng tốn thêm bộ nhớ.
    micro_batches: int = 4

    def __post_init__(self):
        for name in ("tp", "pp", "dp", "ep"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} phải >= 1, đang là {value}.")
        if self.micro_batches < 1:
            raise ValueError(f"micro_batches phải >= 1, đang là {self.micro_batches}.")
        if self.zero not in (0, 1):
            raise ValueError(f"zero chỉ nhận 0 hoặc 1, đang là {self.zero}.")

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp

    @property
    def is_parallel(self) -> bool:
        return self.world_size > 1

    def describe(self) -> str:
        parts = [f"tp={self.tp}", f"pp={self.pp}", f"dp={self.dp}"]
        if self.ep > 1:
            parts.append(f"ep={self.ep}")
        if self.zero:
            parts.append("zero=1")
        parts.append(f"world={self.world_size}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "zero": self.zero,
            "micro_batches": self.micro_batches,
        }


@dataclass
class ProdConfig:
    """Cấu hình đầy đủ: model của Cấp 3 + cách chia của Cấp 4."""

    model: Config = field(default_factory=Config)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    def describe(self) -> str:
        total, active = 0, 0
        return (
            f"model: {self.model.n_layers} tầng, d_model={self.model.d_model}, "
            f"{self.model.n_experts} chuyên gia\n"
            f"chia : {self.parallel.describe()}"
        )
