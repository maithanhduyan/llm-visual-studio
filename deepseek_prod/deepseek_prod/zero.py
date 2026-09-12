"""
zero.py — Chia trạng thái optimizer (ZeRO giai đoạn 1).

Adam giữ cho MỖI tham số hai con số nữa (trung bình động và phương sai), tức
là gấp đôi bộ nhớ của model. Model 1 tỷ thông số thì optimizer chiếm thêm
8 GB — chỉ để nhớ "tham số này đang đi theo hướng nào".

Ý tưởng của ZeRO: các máy trong nhóm dữ liệu vốn giữ BẢN SAO giống hệt nhau
của model, nên chúng đang giữ những trạng thái optimizer giống hệt nhau. Vô
ích. Chia ra:

    máy 0 giữ trạng thái của 1/2 số tham số
    máy 1 giữ trạng thái của 1/2 còn lại

Một bước học diễn ra như sau:

    1. Mọi máy tính gradient trên lô dữ liệu của mình
    2. all_reduce gradient  -> máy nào cũng có gradient đầy đủ
    3. MỖI MÁY CHỈ CẬP NHẬT PHẦN THAM SỐ CỦA MÌNH (dùng optimizer riêng)
    4. broadcast phần vừa cập nhật cho các máy còn lại
    5. Quay lại bước 1 — model trên mọi máy lại giống hệt nhau

Bộ nhớ tiết kiệm được: trạng thái optimizer chia cho dp lần.

ZeRO-1 chỉ chia trạng thái optimizer. ZeRO-2 chia thêm gradient.
ZeRO-3 chia cả tham số (mỗi máy chỉ giữ một mẩu model).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .parallel import Parallel


class ShardedOptimizer:
    """Bọc một optimizer thường, nhưng mỗi máy chỉ giữ một phần trạng thái."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        parallel: Parallel,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        self.parallel = parallel
        self.params = [p for p in parameters if p.requires_grad]
        self.lr = lr

        dp_size = parallel.size("dp")
        if dp_size < 2:
            raise ValueError(
                "ZeRO chỉ có nghĩa khi có từ 2 máy trở lên trong nhóm dữ liệu "
                f"(dp đang là {dp_size})."
            )

        # Chia danh sách tham số thành dp phần. Máy đầu nhận thêm nếu lẻ.
        base, extra = divmod(len(self.params), dp_size)
        rank = parallel.rank_in("dp")
        start = rank * base + min(rank, extra)
        count = base + (1 if rank < extra else 0)

        self.start = start
        self.count = count
        self.mine = self.params[start : start + count]

        self.optimizer = torch.optim.AdamW(self.mine, lr=lr, weight_decay=weight_decay)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.params:
            param.grad = None if set_to_none else torch.zeros_like(param)

    @torch.no_grad()
    def step(self) -> None:
        dp_size = self.parallel.size("dp")

        # 1. Cộng gradient của mọi máy rồi chia đều — giống hệt khi chạy
        #    một máy với lô to bằng dp lần.
        for param in self.params:
            if param.grad is None:
                continue
            self.parallel.all_reduce(param.grad, dim="dp")
            param.grad /= dp_size

        # 2. Chỉ cập nhật phần của mình. Đây là chỗ tiết kiệm bộ nhớ:
        #    optimizer chỉ giữ trạng thái cho `self.mine`.
        self.optimizer.step()

        # 3. Phát phần vừa cập nhật cho các máy khác, để model lại giống nhau.
        #
        #    Phải lặp qua MỌI tham số trên MỌI máy, theo cùng một thứ tự, và
        #    cùng một `src` cho mỗi tham số. Nếu máy nào cũng chỉ phát phần
        #    của mình thì hai máy gọi broadcast trên hai tensor khác nhau —
        #    không khớp nhau và treo. Đây là lỗi đã sập một lần khi làm file này.
        for index, param in enumerate(self.params):
            owner = self._owner_of(index)
            self.parallel.broadcast(
                param.data,
                src=self.parallel.global_rank_of_dp(owner),
                dim="dp",
            )

    def _owner_of(self, index: int) -> int:
        """Tham số thứ `index` trong danh sách thuộc về máy nào của nhóm DP."""
        dp_size = self.parallel.size("dp")
        base, extra = divmod(len(self.params), dp_size)

        # Máy đầu nhận thêm 1 tham số, nên phải trừ phần dôi ra khi tra ngược.
        if index < (base + 1) * extra:
            return index // (base + 1)
        return extra + (index - (base + 1) * extra) // base

    @property
    def state_bytes(self) -> int:
        """Trạng thái optimizer trên MÁY NÀY tốn bao nhiêu byte."""
        total = 0
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total
