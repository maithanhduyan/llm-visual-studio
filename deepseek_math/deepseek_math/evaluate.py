"""
evaluate.py — Đo xem model làm được bao nhiêu phần trăm.

Điểm quan trọng: **bài để thử luôn là bài MỚI**, sinh bằng một hạt giống
khác với lúc học. Nếu lấy lại bài cũ thì chỉ đang đo trí nhớ, không đo
việc học.

Với bậc 1 và 2 thì không có "bài mới" — cả bảng cộng chỉ có 100 bài. Nên
điểm cao ở hai bậc đó KHÔNG có nghĩa là model biết làm toán, chỉ có nghĩa
là nó thuộc bảng cửu chương. Bậc 5 sẽ cho thấy sự khác nhau đó.
"""

from __future__ import annotations

import random

import torch

from .base import Tokenizer
from .curriculum import Level
from .problems import Problem, read_answer


@torch.no_grad()
def solve(model, tokenizer: Tokenizer, prompt: str, max_new: int = 48) -> str:
    """Cho model viết tiếp, dừng khi gặp xuống dòng hoặc hết chỗ."""
    ids = tokenizer.encode(prompt)
    if not ids:
        return ""

    limit = model.config.max_seq_len
    ids = ids[: limit - 1]

    tokens = torch.tensor([ids])
    newline = tokenizer.stoi.get("\n")

    written: list[int] = []

    for _ in range(max_new):
        if tokens.shape[1] >= limit:
            break

        logits, _ = model(tokens)
        next_id = int(logits[0, -1, :].argmax(-1).item())

        if next_id == newline:
            break

        written.append(next_id)
        tokens = torch.cat([tokens, torch.tensor([[next_id]])], dim=1)

    return tokenizer.decode(written)


def evaluate(
    model,
    tokenizer: Tokenizer,
    level: Level,
    count: int = 200,
    seed: int = 999,
) -> dict:
    """Thử `count` bài MỚI của một bậc. Trả về độ chính xác và vài ví dụ."""
    rng = random.Random(seed)
    problems: list[Problem] = level.make(count, rng)

    correct = 0
    examples: list[dict] = []
    empty = 0

    for problem in problems:
        prompt = level.prompt_for(problem)
        written = solve(model, tokenizer, prompt)

        # Phải ghép câu hỏi vào trước khi đọc đáp án: `solve` chỉ trả về phần
        # model VIẾT THÊM, mà với kiểu trả lời thẳng thì phần đó chỉ có mấy
        # chữ số, không có dấu '=' nào. Đây là lỗi đã sập một lần khi làm.
        got = read_answer(prompt + written)
        want = str(problem.answer)

        if not got:
            empty += 1

        if got == want:
            correct += 1
        elif len(examples) < 4:
            examples.append(
                {
                    "problem": f"{problem.a}{problem.op}{problem.b}",
                    "want": want,
                    "got": got or "(không viết gì)",
                    "raw": written[:60],
                }
            )

    return {
        "level": level.number,
        "name": level.name,
        "count": count,
        "correct": correct,
        "accuracy": correct / count,
        "empty": empty,
        "examples": examples,
    }


def accuracy_for_digits(
    model,
    tokenizer: Tokenizer,
    digits: int,
    style: str,
    count: int = 200,
    seed: int = 777,
) -> float:
    """Thử với số có `digits` chữ số — dùng để xem có suy rộng ra không."""
    from .curriculum import Level as _Level

    probe = _Level(0, f"{digits} chữ số", digits, style)
    return evaluate(model, tokenizer, probe, count=count, seed=seed)["accuracy"]
