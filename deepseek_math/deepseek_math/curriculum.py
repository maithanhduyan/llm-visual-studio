"""
curriculum.py — Năm bậc học, và vì sao phải theo đúng thứ tự này.

    Bậc  Tên                            Số chữ số  Cách viết       Số bài
    ───  ─────────────────────────────  ─────────  ──────────────  ───────
     1   Cộng một chữ số, không nhớ     1          trả lời thẳng   55
     2   Cộng một chữ số, có nhớ        1          trả lời thẳng   45
     3   Cộng hai chữ số, trả lời thẳng 2          trả lời thẳng   vô hạn
     4   Cộng hai chữ số, từng bước     2          từng bước       vô hạn
     5   Cộng ba chữ số, từng bước      3          từng bước       vô hạn

**Vì sao bậc 1 và 2 tách riêng:** bảng cộng một chữ số chỉ có 100 bài. Model
học thuộc lòng được, không cần hiểu gì. Bậc 1 chỉ có 55 bài (tổng ≤ 9), bậc 2
là 45 bài còn lại (có nhớ). Tách ra để thấy rõ: học thuộc thì dễ, mà học thuộc
thì KHÔNG suy ra được bài dài hơn.

**Vì sao bậc 3 và 4 giống nhau về bài toán, chỉ khác cách viết:** đây là thí
nghiệm chính của cả dự án. Cùng một tập bài, cùng một model, chỉ khác chỗ có
bắt model viết ra từng bước hay không. Rồi đem CẢ HAI đi thử với số ba chữ số.

    Dự đoán:  bậc 3 (trả lời thẳng) sẽ thất bại ở ba chữ số,
              bậc 4 (viết từng bước) sẽ làm được.

Vì sao dự đoán vậy: cộng hai số nhiều chữ số là bài toán **tuần tự** — cột sau
phải biết số nhớ của cột trước. Model chỉ có số tầng cố định, không có vòng
lặp, nên không thể tính ngầm trong đầu. Viết ra từng bước thì phép tính trở
thành việc "đọc bước trước rồi viết bước sau" — mà việc đó thì model làm được.
"""

from __future__ import annotations

from dataclasses import dataclass

from .problems import Problem, all_single_digit, direct, random_problem, scratchpad

import random


@dataclass
class Level:
    number: int
    name: str
    digits: int
    style: str  # "direct" hoặc "scratchpad"
    carry: str = "all"  # "all" | "no_carry" | "carry" (chỉ dùng cho một chữ số)
    finite: bool = False  # True = chỉ có hữu hạn bài (bảng cộng)
    # Nếu có, mỗi bài chọn ngẫu nhiên một độ dài trong đây. Đây là cách dạy
    # model biết "số cột bằng số chữ số" — xem checks.py.
    digits_mix: tuple[int, ...] | None = None

    def render(self, problem: Problem) -> str:
        return direct(problem) if self.style == "direct" else scratchpad(problem)

    def prompt_for(self, problem: Problem) -> str:
        """Phần đưa cho model, chưa có đáp án."""
        return f"{problem.a}{problem.op}{problem.b}="

    def make(self, count: int, rng: random.Random) -> list[Problem]:
        if self.finite:
            pool = all_single_digit(self.carry)
            return [rng.choice(pool) for _ in range(count)]

        if self.digits_mix:
            return [random_problem(rng.choice(self.digits_mix), rng) for _ in range(count)]

        return [random_problem(self.digits, rng) for _ in range(count)]

    @property
    def pool_size(self) -> int | None:
        """Bao nhiêu bài khác nhau. None = vô hạn."""
        if self.finite:
            return len(all_single_digit(self.carry))
        return None


LEVELS: dict[int, Level] = {
    1: Level(1, "Cộng một chữ số, không nhớ", 1, "direct", carry="no_carry", finite=True),
    2: Level(2, "Cộng một chữ số, có nhớ", 1, "direct", carry="carry", finite=True),
    3: Level(3, "Cộng hai chữ số, trả lời thẳng", 2, "direct"),
    4: Level(4, "Cộng hai chữ số, viết từng bước", 2, "scratchpad"),
    5: Level(
        5,
        "Cộng 1-3 chữ số, viết từng bước",
        3,
        "scratchpad",
        digits_mix=(1, 2, 3),
    ),
}


def build_text(level: Level, count: int, seed: int = 0) -> str:
    """Sinh dữ liệu học cho một bậc: mỗi bài một dòng."""
    rng = random.Random(seed)
    problems = level.make(count, rng)
    return "\n".join(level.render(p) for p in problems) + "\n"


def build_mixed(level_numbers: list[int], count_each: int, seed: int = 0) -> str:
    """Trộn nhiều bậc vào một tập dữ liệu."""
    parts = [build_text(LEVELS[n], count_each, seed + n) for n in level_numbers]
    return "".join(parts)


def describe() -> str:
    lines = [f"{'bậc':>4}  {'tên':<34} {'chữ số':>7} {'cách viết':<12}"]
    lines.append("-" * 62)
    for number, level in LEVELS.items():
        lines.append(
            f"{number:>4}  {level.name:<34} {level.digits:>7} {level.style:<12}"
        )
    return "\n".join(lines)
