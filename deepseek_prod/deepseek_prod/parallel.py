"""
parallel.py — Tôi là ai trong lưới các máy?

Khi model bị cắt ra nhiều máy, mỗi tiến trình phải biết ba thứ:

    1. Mình là ai          -> rank
    2. Mình thuộc nhóm nào -> nhóm TP / PP / DP
    3. Khi nào cần nói chuyện với ai -> gọi all_reduce / all_gather / send

Cách đánh số rank (giống Megatron-LM):

    rank = tp_rank  +  pp_rank * tp  +  dp_rank * tp * pp

Ví dụ tp=2, pp=2, dp=1 thì:

    rank 0 -> (tp 0, pp 0)      rank 2 -> (tp 0, pp 1)
    rank 1 -> (tp 1, pp 0)      rank 3 -> (tp 1, pp 1)

Mỗi tiến trình thuộc BA nhóm khác nhau, và mỗi nhóm dùng cho một việc:

    nhóm TP  những máy cùng giữ MỘT phần của cùng một tầng
             -> all_reduce sau mỗi phép nhân bị chẻ
    nhóm PP  những máy giữ các tầng khác nhau của cùng một model
             -> gửi kết quả trung gian cho nhau
    nhóm DP  những máy giữ BẢN SAO giống hệt của cả model
             -> all_reduce gradient
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist

from .config import ParallelConfig


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


class Parallel:
    def __init__(self, config: ParallelConfig, initialize: bool = True) -> None:
        self.config = config
        self.distributed = is_distributed()

        if self.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        if self.world_size != config.world_size:
            raise ValueError(
                f"Cấu hình nói cần {config.world_size} tiến trình "
                f"({config.describe()}) nhưng thực tế đang có {self.world_size}.\n"
                "Số tiến trình phải bằng tp * pp * dp."
            )

        self.tp_size, self.pp_size, self.dp_size = config.tp, config.pp, config.dp

        # Expert parallel dùng chung nhóm TP: các chuyên gia bị chia cho đúng
        # những máy đang chia nhau một tầng. Hệ thật tách riêng một nhóm EP,
        # nhưng ở quy mô này dùng chung là đủ và ít nhóm hơn thì dễ theo dõi hơn.
        self.ep_size = config.ep if config.ep > 1 else 1
        if self.ep_size not in (1, self.tp_size):
            raise ValueError(
                f"ep ({self.ep_size}) phải bằng 1 hoặc bằng tp ({self.tp_size}). "
                "Ở đây nhóm EP chính là nhóm TP."
            )

        # Tách rank thành ba toạ độ.
        self.tp_rank = self.rank % self.tp_size
        self.pp_rank = (self.rank // self.tp_size) % self.pp_size
        self.dp_rank = self.rank // (self.tp_size * self.pp_size)

        self.tp_group = None
        self.pp_group = None
        self.dp_group = None

        if self.distributed and self.world_size > 1:
            self._make_groups()

    # ------------------------------------------------------------------
    # Dựng nhóm
    # ------------------------------------------------------------------

    def rank_of(self, tp: int, pp: int, dp: int) -> int:
        return tp + pp * self.tp_size + dp * self.tp_size * self.pp_size

    def global_rank_of_pp(self, pp_rank: int) -> int:
        """Rank toàn cục của máy thứ `pp_rank` trong cùng cột tp/dp."""
        return self.rank_of(self.tp_rank, pp_rank, self.dp_rank)

    def global_rank_of_dp(self, dp_rank: int) -> int:
        """Rank toàn cục của máy thứ `dp_rank` trong cùng cột tp/pp."""
        return self.rank_of(self.tp_rank, self.pp_rank, dp_rank)

    def _make_groups(self) -> None:
        """Tạo ba họ nhóm con.

        MỌI tiến trình phải gọi `new_group` theo cùng một thứ tự, nếu không
        sẽ treo. Nên vòng lặp ở đây cố định, không phụ thuộc vào rank.
        """
        tp_groups: dict[tuple[int, int], object] = {}
        pp_groups: dict[tuple[int, int], object] = {}
        dp_groups: dict[tuple[int, int], object] = {}

        for dp in range(self.dp_size):
            for pp in range(self.pp_size):
                ranks = [self.rank_of(tp, pp, dp) for tp in range(self.tp_size)]
                tp_groups[(pp, dp)] = dist.new_group(ranks=ranks)

        for dp in range(self.dp_size):
            for tp in range(self.tp_size):
                ranks = [self.rank_of(tp, pp, dp) for pp in range(self.pp_size)]
                pp_groups[(tp, dp)] = dist.new_group(ranks=ranks)

        for pp in range(self.pp_size):
            for tp in range(self.tp_size):
                ranks = [self.rank_of(tp, pp, dp) for dp in range(self.dp_size)]
                dp_groups[(tp, pp)] = dist.new_group(ranks=ranks)

        self.tp_group = tp_groups[(self.pp_rank, self.dp_rank)]
        self.pp_group = pp_groups[(self.tp_rank, self.dp_rank)]
        self.dp_group = dp_groups[(self.tp_rank, self.pp_rank)]

    # ------------------------------------------------------------------
    # Hỏi cho biết mình đang ở đâu
    # ------------------------------------------------------------------

    def group(self, dim: str):
        return {"tp": self.tp_group, "pp": self.pp_group, "dp": self.dp_group, "ep": self.tp_group}[dim]

    def size(self, dim: str) -> int:
        if dim == "ep":
            return self.ep_size
        return {"tp": self.tp_size, "pp": self.pp_size, "dp": self.dp_size}[dim]

    def rank_in(self, dim: str) -> int:
        if dim == "ep":
            return self.tp_rank
        return {"tp": self.tp_rank, "pp": self.pp_rank, "dp": self.dp_rank}[dim]

    @property
    def is_tp_first(self) -> bool:
        return self.tp_rank == 0

    @property
    def is_pp_first(self) -> bool:
        """Máy giữ tầng đầu tiên (nhận token đầu vào)."""
        return self.pp_rank == 0

    @property
    def is_pp_last(self) -> bool:
        """Máy giữ tầng cuối (tính loss)."""
        return self.pp_rank == self.pp_size - 1

    # ------------------------------------------------------------------
    # Nói chuyện với các máy khác
    # ------------------------------------------------------------------

    def all_reduce(self, tensor: torch.Tensor, dim: str = "tp") -> torch.Tensor:
        """Cộng giá trị của mọi máy trong nhóm lại, máy nào cũng nhận kết quả."""
        if not self.distributed or self.size(dim) == 1:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group(dim))
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: str = "tp") -> list[torch.Tensor]:
        """Gom giá trị của mọi máy trong nhóm về mỗi máy."""
        if not self.distributed or self.size(dim) == 1:
            return [tensor]

        parts = [torch.empty_like(tensor) for _ in range(self.size(dim))]
        dist.all_gather(parts, tensor.contiguous(), group=self.group(dim))
        return parts

    def broadcast(self, tensor: torch.Tensor, src: int = 0, dim: str = "tp") -> torch.Tensor:
        """Một máy đọc lên, cả nhóm nghe theo."""
        if not self.distributed or self.size(dim) == 1:
            return tensor

        dist.broadcast(tensor, src=src, group=self.group(dim))
        return tensor

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        dist.send(tensor.contiguous(), dst=dst, group=self.pp_group)

    def recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        dist.recv(tensor, src=src, group=self.pp_group)
        return tensor

    def barrier(self, dim: str = "dp") -> None:
        if self.distributed and self.size(dim) > 1:
            dist.barrier(group=self.group(dim))

    def __repr__(self) -> str:
        return (
            f"Parallel(rank={self.rank}/{self.world_size}, "
            f"tp={self.tp_rank}/{self.tp_size}, "
            f"pp={self.pp_rank}/{self.pp_size}, "
            f"dp={self.dp_rank}/{self.dp_size})"
        )
