"""
problems.py — Sinh bài toán, và viết chúng ra thành chữ.

Có HAI cách viết cùng một bài toán, và sự khác nhau giữa chúng là toàn bộ
ý tưởng của dự án này:

    TRẢ LỜI THẲNG        17+25=42
    VIẾT TỪNG BƯỚC       17+25=[7+5=12][1+2+1=4]=42
                          ^^^^^^^^ cột đơn vị: 7+5=12
                                   ^^^^^^^^ cột chục: 1+2+1=4
                                              (số 1 là số nhớ từ "12")

Cách viết từng bước có một tính chất rất đẹp: **số nhớ không cần ghi riêng**.
Nó nằm sẵn trong kết quả của cột trước — chữ số hàng chục của "12" chính là
số nhớ. Nên bước sau chỉ việc cộng ba số: 1+2+1.

Và đáp án cũng đọc ra được từ chính các bước: lấy chữ số cuối của mỗi cột,
rồi thêm số nhớ cuối cùng.

    17+25  ->  [7+5=12]  [1+2+1=4]   ->  2, 4   ->  42
    95+15  ->  [5+5=10]  [9+1+1=11]  ->  0, 1, 1 -> 110
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Problem:
    a: int
    b: int
    op: str = "+"

    @property
    def answer(self) -> int:
        return self.a + self.b if self.op == "+" else self.a - self.b

    def __str__(self) -> str:
        return f"{self.a}{self.op}{self.b}"


# ======================================================================
# Hai cách viết
# ======================================================================


def direct(problem: Problem) -> str:
    """Cách 1: hỏi gì đáp nấy. Không có chỗ nào để mà tính."""
    return f"{problem.a}{problem.op}{problem.b}={problem.answer}"


def scratchpad(problem: Problem) -> str:
    """Cách 2: viết ra từng cột, từ phải sang trái, y như đặt tính rồi tính."""
    a_digits = [int(d) for d in reversed(str(problem.a))]
    b_digits = [int(d) for d in reversed(str(problem.b))]

    steps = []
    carry = 0

    for index in range(max(len(a_digits), len(b_digits))):
        left = a_digits[index] if index < len(a_digits) else 0
        right = b_digits[index] if index < len(b_digits) else 0

        if carry:
            steps.append(f"[{left}+{right}+{carry}={left + right + carry}]")
        else:
            steps.append(f"[{left}+{right}={left + right}]")

        carry = (left + right + carry) // 10

    return f"{problem.a}{problem.op}{problem.b}=" + "".join(steps) + f"={problem.answer}"


# ======================================================================
# Đọc đáp án ra khỏi câu model viết
# ======================================================================


def read_answer(text: str) -> str:
    """Lấy đáp án ở cuối câu. Trả về chuỗi rỗng nếu không đọc được.

    Với cách viết từng bước, trong câu có nhiều dấu '=' (một cho mỗi cột).
    Đáp án là đoạn sau dấu '=' CUỐI CÙNG — nhưng chỉ khi dấu đó nằm SAU dấu
    ngoặc vuông cuối. Nếu model viết dở dang, dấu '=' cuối cùng vẫn còn nằm
    trong một bước, và đọc theo nó sẽ ra một chữ số của bước đó — tưởng là
    đúng mà thật ra là model chưa làm xong.
    """
    if "=" not in text:
        return ""

    last_equal = text.rfind("=")

    if text.rfind("]") > last_equal:
        return ""  # viết dở, chưa ra tới đáp án

    digits = ""
    for ch in text[last_equal + 1 :]:
        if ch.isdigit():
            digits += ch
        else:
            break

    return digits


# ======================================================================
# Sinh bài toán
# ======================================================================


def random_problem(digits: int, rng: random.Random, allow_carry: bool = True) -> Problem:
    """Sinh một bài cộng có đúng `digits` chữ số ở mỗi số."""
    low = 10 ** (digits - 1) if digits > 1 else 0
    high = 10**digits - 1

    if not allow_carry and digits == 1:
        # Không nhớ: tổng phải nhỏ hơn 10.
        a = rng.randint(low, min(9, high - 1))
        b = rng.randint(0, 9 - a)
        return Problem(a, b)

    return Problem(rng.randint(low, high), rng.randint(low, high))


def all_single_digit(mode: str = "all") -> list[Problem]:
    """Hết cả bảng cộng một chữ số.

    Chỉ có 100 bài nên model HỌC THUỘC được, không cần hiểu gì.

        mode="no_carry"  tổng <= 9   -> 55 bài (bậc 1)
        mode="carry"     tổng >= 10  -> 45 bài (bậc 2)
        mode="all"                   -> 100 bài
    """
    problems = []
    for a in range(10):
        for b in range(10):
            total = a + b
            if mode == "no_carry" and total > 9:
                continue
            if mode == "carry" and total < 10:
                continue
            problems.append(Problem(a, b))
    return problems
