"""
pipeline.py — Cắt DỌC: mỗi máy giữ vài tầng.

Tensor parallel cắt NGANG một tầng (mọi máy đều chạy mọi tầng).
Pipeline parallel cắt DỌC: máy 0 giữ tầng 0-1, máy 1 giữ tầng 2-3, ...

    máy 0                 máy 1
    embedding             (không có)
    tầng 0                tầng 2
    tầng 1                tầng 3
    (không có)            norm + lm_head
        |                     ^
        +---- activation -----+

Vấn đề: nếu làm tuần tự thì lúc nào cũng chỉ một máy chạy, còn lại ngồi chơi.
Giải pháp: chia lô ra nhiều VI LÔ (micro-batch) rồi bơm lần lượt:

    thời gian ->
    máy 0:  [m0][m1][m2][m3]          [b3][b2][b1][b0]
    máy 1:      [m0][m1][m2][m3]  [b3][b2][b1][b0]
    máy 2:          [m0][m1][m2][m3] ...
                     ^ lúc này máy 0 đã sang vi lô tiếp theo

Cách này gọi là GPipe. Bản tinh chỉnh hơn là 1F1B (xen kẽ forward/backward
để đỡ tốn bộ nhớ) — xem docs/song-song.md.

Chiều ngược lại cũng y hệt, chỉ đổi vai: gradient đi từ máy cuối về máy đầu.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel import Parallel
from .tensor_parallel import divide


def layer_range(n_layers: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    """Chia `n_layers` tầng cho `pp_size` máy, không cần chia hết.

    3 tầng cho 2 máy thì máy 0 nhận 2 tầng, máy 1 nhận 1 tầng. Hệ thật
    cũng làm vậy — bắt số tầng chia hết cho số máy thì quá cứng.
    """
    base, extra = divmod(n_layers, pp_size)
    start = pp_rank * base + min(pp_rank, extra)
    count = base + (1 if pp_rank < extra else 0)
    return start, count


class _RecvFromPrevious(torch.autograd.Function):
    """Nhận activation từ máy trước.

    Phải là một `autograd.Function` chứ không phải gọi `recv` trần: nếu chỉ
    `recv` vào một tensor thường thì đồ thị tính toán ĐỨT ở đó, `loss.backward()`
    sẽ dừng lại và không bao giờ gửi gradient về máy trước — máy trước ngồi
    chờ mãi. Đây đúng là lỗi đã sập một lần khi làm file này.

    Chiều xuôi: nhận activation.
    Chiều ngược: gửi gradient về máy trước.
    """

    @staticmethod
    def forward(ctx, placeholder: torch.Tensor, parallel: Parallel, src: int) -> torch.Tensor:
        ctx.parallel = parallel
        ctx.src = src

        buffer = torch.empty_like(placeholder)
        parallel.recv(buffer, src=src)
        return buffer

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ctx.parallel.send(grad_output.contiguous(), dst=ctx.src)
        return None, None, None


class PipelineStage(nn.Module):
    """Đoạn model mà máy NÀY giữ."""

    def __init__(self, model: nn.Module, parallel: Parallel) -> None:
        super().__init__()

        self.parallel = parallel
        self.first = parallel.is_pp_first
        self.last = parallel.is_pp_last

        self.n_streams = model.config.n_streams
        self.d_model = model.config.d_model

        start, count = layer_range(model.config.n_layers, parallel.pp_size, parallel.pp_rank)

        self.embedding = model.embedding if self.first else None
        self.blocks = nn.ModuleList(list(model.blocks)[start : start + count])
        self.norm = model.norm if self.last else None
        self.lm_head = model.lm_head if self.last else None

        self.layer_range = (start, start + count)

        # `deepseek_lite` dùng chung trọng số giữa bảng tra và LM Head.
        self.tied = model.lm_head.weight is model.embedding.weight

    # ------------------------------------------------------------------
    # Đường ống
    # ------------------------------------------------------------------

    def _activate(self, tokens: torch.Tensor) -> torch.Tensor:
        """Từ token -> các dòng suy nghĩ, ở máy đầu tiên."""
        x = self.embedding(tokens)
        batch, length = tokens.shape
        return x.unsqueeze(2).expand(batch, length, self.n_streams, self.d_model).contiguous()

    def _shape_like(self, tokens: torch.Tensor) -> torch.Tensor:
        """Tạo chỗ trống đúng kích thước để nhận activation từ máy trước.

        `requires_grad=True` là bắt buộc, không phải cho vui: autograd chỉ
        dựng đồ thị cho một `autograd.Function` khi có ÍT NHẤT MỘT đầu vào
        cần gradient. Nếu chỗ trống này không cần gradient thì đầu ra của
        hàm cũng không có `grad_fn`, đồ thị đứt ngay tại đây, và gradient
        không bao giờ được gửi về máy trước. Đây là lỗi thứ hai đã sập khi
        làm file này.
        """
        batch, length = tokens.shape
        return torch.empty(
            batch, length, self.n_streams, self.d_model, requires_grad=True
        )

    def forward_micro(self, tokens: torch.Tensor):
        """Chạy MỘT vi lô qua đoạn của mình.

        Trả về logits nếu là máy cuối, ngược lại trả về tensor đã gửi đi
        (giữ lại để lát nữa backward qua nó).
        """
        if self.first:
            streams = self._activate(tokens)
        else:
            streams = _RecvFromPrevious.apply(
                self._shape_like(tokens),
                self.parallel,
                self.parallel.rank_in("pp") - 1,
            )

        for block in self.blocks:
            streams = block(streams)

        if self.last:
            x = streams.mean(dim=2)
            x = self.norm(x)
            return self.lm_head(x)

        self.parallel.send(streams.contiguous(), dst=self.parallel.rank_in("pp") + 1)
        return streams

    def backward_micro(self, output: torch.Tensor) -> None:
        """Nhận gradient từ máy sau rồi chạy ngược qua đoạn của mình."""
        if self.last:
            return  # máy cuối tự gọi loss.backward()

        grad = torch.empty_like(output)
        self.parallel.recv(grad, src=self.parallel.rank_in("pp") + 1)
        output.backward(grad)

    def share_tied_gradient(self) -> None:
        """Cộng gradient của bảng tra dùng chung giữa máy đầu và máy cuối.

        Model dùng chung trọng số giữa Token Embedding (máy đầu) và LM Head
        (máy cuối). Khi cắt dọc, hai thứ đó nằm ở hai máy khác nhau, nên mỗi
        máy chỉ nhận được MỘT NỬA gradient của bảng tra. Không cộng lại thì
        bảng tra học sai — và sai rất khó thấy, vì loss vẫn ra đúng.

        Đây là chi tiết hệ thật cũng phải xử lý (Megatron gọi nó là
        "gradient all-reduce cho embedding dùng chung").
        """
        if self.parallel.pp_size == 1 or not self.tied:
            return

        if self.last:
            weight = self.lm_head.weight
        elif self.first:
            weight = self.embedding.weight
        else:
            return

        if weight.grad is None:
            return

        if self.last:
            self.parallel.send(weight.grad.contiguous(), dst=self.parallel.global_rank_of_pp(0))
        else:
            incoming = torch.empty_like(weight.grad)
            self.parallel.recv(
                incoming, src=self.parallel.global_rank_of_pp(self.parallel.pp_size - 1)
            )
            weight.grad += incoming

    def __repr__(self) -> str:
        lo, hi = self.layer_range
        return (
            f"PipelineStage(máy {self.parallel.pp_rank}/{self.parallel.pp_size}, "
            f"tầng {lo}-{hi - 1}"
            + (", giữ embedding" if self.first else "")
            + (", giữ norm + lm_head" if self.last else "")
            + ")"
        )


def run_pipeline_batch(
    stage: PipelineStage,
    tokens: torch.Tensor,
    targets: torch.Tensor | None,
    micro_batches: int,
) -> float | None:
    """Chạy trọn một lô qua cả đường ống, trả về loss (chỉ máy cuối có).

    Cách làm: forward HẾT các vi lô, rồi backward HẾT. Nhờ vậy máy 0 có thể
    chạy vi lô 2 trong khi máy 1 còn đang chạy vi lô 1.
    """
    token_chunks = list(torch.chunk(tokens, micro_batches, dim=0))
    target_chunks = list(torch.chunk(targets, micro_batches, dim=0)) if targets is not None else None

    outputs = [stage.forward_micro(chunk) for chunk in token_chunks]

    if stage.last:
        if targets is None:
            return None

        losses = [
            F.cross_entropy(out.reshape(-1, out.shape[-1]), target.reshape(-1))
            for out, target in zip(outputs, target_chunks)
        ]

        # Chạy ngược TỪNG vi lô, theo đúng thứ tự xuôi.
        #
        # Không cộng hết loss rồi backward một lần: làm vậy thì autograd sẽ
        # đi ngược theo thứ tự ngược lại, trong khi máy trước đang chờ theo
        # thứ tự xuôi — hai bên lệch nhau và gradient bị ghép sai vi lô.
        for loss in losses:
            (loss / len(losses)).backward()

        stage.share_tied_gradient()
        return sum(loss.item() for loss in losses) / len(losses)

    for output in outputs:
        stage.backward_micro(output)

    stage.share_tied_gradient()
    return None
