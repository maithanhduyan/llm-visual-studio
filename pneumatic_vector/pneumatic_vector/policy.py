"""
policy.py — Bộ ra quyết định bằng model đã huấn luyện.

Model đọc bốn con số trạng thái rồi viết ra một lệnh. Nó chỉ là một model
sinh chữ bình thường — không có gì đặc biệt về "điều khiển" cả. Cái làm nó
thành bộ điều khiển là **cách ta dùng nó**: cho nó đọc trạng thái, để nó
viết tiếp, rồi hiểu chữ nó viết ra thành lệnh.

    "13.4 0.5 1.2 5.5 -> "   model viết tiếp   "GIU 13.0"

---

## Điều gì xảy ra nếu model viết sai định dạng

Model nhỏ thì sẽ có lúc viết ra thứ vô nghĩa. Ví dụ `"HAM 999"` hoặc
`"XYZ 3"`. Nếu đưa thẳng cho PID thì con tàu rơi.

Nên có một lớp chắn: đọc lệnh ra, kiểm tra hợp lệ, và nếu không hiểu thì
**cắt ga** — hành động an toàn nhất. Đây là điều mọi hệ thống thật đều làm:
LLM ra quyết định, nhưng luôn có một lớp kiểm tra giữa nó và cơ cấu chấp
hành.
"""

from __future__ import annotations

import torch

from .base import Config, DeepSeekLite, Tokenizer
from .expert import COMMANDS, Decision
from .paths import RUNS_DIR

# Model nhỏ. Việc ở đây dễ hơn làm toán nhiều: chỉ cần đọc bốn số và
# chọn một trong bốn lệnh.
PILOT_CONFIG = dict(
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    n_layers=2,
    max_seq_len=48,
    window=0,
    n_experts=4,
    top_k=2,
    n_shared=1,
    expert_hidden=48,
    n_streams=1,
)


def build_model(vocab_size: int) -> DeepSeekLite:
    return DeepSeekLite(Config(vocab_size=vocab_size, **PILOT_CONFIG))


class LearnedPilot:
    """Bộ ra quyết định dùng model đã học. Cùng giao diện với `ExpertPilot`."""

    def __init__(self, model, tokenizer: Tokenizer, max_new: int = 16) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new = max_new

        self.phase = "?"
        self.peak_altitude = 0.0

        # Đề bài. `simulate.fly` đặt lại trước mỗi chuyến bay.
        self.mission = None

        # Đếm xem model viết ra lệnh không hiểu được bao nhiêu lần.
        self.invalid = 0
        self.total = 0

        self.newline = tokenizer.stoi.get("\n")

    @classmethod
    def load(cls, name: str = "pilot") -> "LearnedPilot":
        checkpoint = torch.load(RUNS_DIR / f"{name}.pt", map_location="cpu")

        tokenizer = Tokenizer.load(checkpoint["chars"])
        model = build_model(tokenizer.vocab_size)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        return cls(model, tokenizer)

    @torch.no_grad()
    def write(self, prompt: str) -> str:
        """Cho model viết tiếp từ `prompt`, dừng ở xuống dòng."""
        ids = self.tokenizer.encode(prompt)[-self.model.config.max_seq_len + 1 :]

        if not ids:
            return ""

        tokens = torch.tensor([ids])
        written: list[int] = []

        for _ in range(self.max_new):
            logits, _ = self.model(tokens)
            next_id = int(logits[0, -1, :].argmax(-1).item())

            if next_id == self.newline:
                break

            written.append(next_id)
            tokens = torch.cat([tokens, torch.tensor([[next_id]])], dim=1)

        return self.tokenizer.decode(written)

    def decide(self, state, tank):
        """Đọc trạng thái, hỏi model, rồi kiểm tra lệnh trước khi dùng."""
        from .expert import state_text

        target = getattr(self.mission, "target_altitude", 12.0)
        prompt = f"{state_text(state, tank, target)} -> "
        answer = self.write(prompt)

        self.total += 1
        try:
            decision = Decision.parse(answer)
        except Exception:  # noqa: BLE001
            decision = Decision("ROI")

        # Đếm xem model có viết ra chữ không hiểu được không.
        #
        # Phải xem CHỮ model viết, không phải lệnh đã đọc ra: `Decision.parse`
        # đã tự đổi mọi thứ lạ thành `ROI` rồi, nên nếu đọc `decision.command`
        # thì lúc nào cũng thấy hợp lệ và bộ đếm mãi mãi bằng 0.
        first_word = answer.strip().split()[0].upper() if answer.strip() else ""

        if first_word not in COMMANDS:
            self.invalid += 1

        # Kiểm tra lệnh trước khi đưa xuống PID.
        #
        # `LEN` và `GIU` không kèm con số (độ cao mục tiêu lấy từ đề bài),
        # nên chỉ có `HAM` mới cần kiểm tra con số. `HAM 999` là lệnh vô
        # nghĩa và nếu đưa thẳng xuống thì con tàu rơi.
        if decision.command == "HAM" and not (1.0 <= decision.value <= 15.0):
            self.invalid += 1
            decision = Decision("HAM", 6.5)

        self.phase = decision.command
        self.peak_altitude = max(self.peak_altitude, state.z)

        return decision

    @property
    def invalid_rate(self) -> float:
        return self.invalid / max(self.total, 1)


__all__ = ["LearnedPilot", "PILOT_CONFIG", "build_model"]
