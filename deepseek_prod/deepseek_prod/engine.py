"""
engine.py — Máy phục vụ: gom nhiều câu hỏi vào một lô (continuous batching).

Cách thường (static batching): gom đủ 8 câu rồi chạy. Câu nào xong trước
thì chỗ đó ngồi không cho tới khi cả lô xong. Người thứ 9 phải đợi.

Continuous batching: mỗi bước chạy, câu nào xong thì trả kết quả và NHẢ CHỖ
NGAY, câu đang đợi được nhận vào luôn. Không ai phải đợi ai.

    static:      [câu 1][câu 2 dài gấp ba.............][câu 3]
                 .....................chỗ trống..............   <- lãng phí

    continuous:  [1][2][4][7]   câu 1 xong -> [5] vào ngay
                 [2][4][7][5]   câu 7 xong -> [9] vào ngay
                 ...

Muốn làm được vậy thì đệm K/V phải chia khối (paged_cache.py), vì mỗi câu
vào lô ở một thời điểm khác nhau, dài ngắn khác nhau, không thể xin sẵn
một vùng cố định cho cả lô.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .base_model import Tokenizer
from .paged_cache import PagedKVCache


@dataclass
class Request:
    request_id: int
    prompt_ids: list[int]
    max_new_tokens: int = 20
    temperature: float = 0.8

    output: list[int] = field(default_factory=list)
    prefilled: bool = False
    finished: bool = False
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output)

    @property
    def latency(self) -> float:
        return self.end_time - self.start_time


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class ContinuousEngine:
    """Máy phục vụ chạy nhiều câu hỏi cùng lúc."""

    def __init__(
        self,
        model,
        tokenizer: Tokenizer,
        cache: PagedKVCache,
        max_batch: int = 4,
        name: str = "continuous",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.max_batch = max_batch
        self.name = name

        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []

    # ------------------------------------------------------------------

    def submit(self, request: Request) -> None:
        self.waiting.append(request)

    @property
    def active(self) -> int:
        return len(self.running)

    def _admit(self) -> None:
        """Nhận thêm câu mới vào lô, chừng nào còn chỗ."""
        while self.waiting and len(self.running) < self.max_batch:
            request = self.waiting[0]

            if not self.cache.add_sequence(request.request_id):
                break  # hết khối K/V, đành để câu sau đợi

            self.waiting.pop(0)
            request.start_time = time.perf_counter()
            self.running.append(request)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _prefill(self, request: Request) -> None:
        """Đọc cả câu mở đầu của MỘT câu, một mình.

        Chưa gom chung được vì các câu mở đầu dài ngắn khác nhau. Hệ thật
        cũng tách riêng bước này (gọi là chunked prefill).
        """
        self.cache.begin_batch([request.request_id])

        tokens = torch.tensor([request.prompt_ids])
        logits, _ = self.model(tokens, cache=self.cache, start_pos=0)

        request.prefilled = True
        self._emit(request, logits[:, -1, :])

    @torch.no_grad()
    def _decode_batch(self) -> None:
        """Một bước sinh chữ cho TẤT CẢ các câu đang chạy."""
        if not self.running:
            return

        seq_ids = [request.request_id for request in self.running]
        self.cache.begin_batch(seq_ids)

        # Mỗi câu đang ở một vị trí khác nhau -> start_pos là một tensor.
        positions = torch.tensor(
            [self.cache.committed_lengths[seq_id] for seq_id in seq_ids],
            dtype=torch.long,
        )

        # Câu vừa đọc xong câu mở đầu thì chữ cuối của nó đã sinh rồi,
        # nên bước này phải đưa vào chữ VỪA SINH, không phải chữ cuối câu mở đầu.
        last_tokens = torch.tensor(
            [[request.output[-1]] for request in self.running], dtype=torch.long
        )

        logits, _ = self.model(last_tokens, cache=self.cache, start_pos=positions)

        for index, request in enumerate(self.running):
            self._emit(request, logits[index, -1, :])

    def _emit(self, request: Request, logits: torch.Tensor) -> None:
        """Bốc một chữ từ logits, ghép vào câu, và xem đã xong chưa."""
        next_token = _sample(logits.unsqueeze(0), request.temperature)
        request.output.append(int(next_token.item()))

        if len(request.output) >= request.max_new_tokens:
            self._finish(request)

    def _finish(self, request: Request) -> None:
        request.finished = True
        request.end_time = time.perf_counter()

        self.cache.free_sequence(request.request_id)  # nhả khối NGAY
        self.running.remove(request)
        self.finished.append(request)

        # Còn đang chờ thì bước sau được nhận vào.

    # ------------------------------------------------------------------

    def run(self) -> list[Request]:
        """Chạy tới khi mọi câu hỏi xong."""
        started = time.perf_counter()

        while self.waiting or self.running:
            self._admit()

            # Câu mới vào thì đọc câu mở đầu trước, một mình.
            for request in list(self.running):
                if not request.prefilled:
                    self._prefill(request)

            self._decode_batch()

        self.elapsed = time.perf_counter() - started
        return self.finished


class SequentialEngine(ContinuousEngine):
    """Cách thường, để so sánh: lần lượt từng câu một, không gom lô."""

    def run(self) -> list[Request]:
        started = time.perf_counter()

        while self.waiting or self.running:
            # Chỉ nhận ĐÚNG MỘT câu, và chạy cho xong mới nhận câu sau.
            if not self.running:
                self.max_batch = 1
                self._admit()
                self._prefill(self.running[0])

            self._decode_batch()

        self.elapsed = time.perf_counter() - started
        return self.finished


def text_of(tokenizer: Tokenizer, request: Request) -> str:
    return tokenizer.decode(request.prompt_ids + request.output)
