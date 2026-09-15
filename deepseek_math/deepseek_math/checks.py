"""
checks.py — Thí nghiệm chính của cả dự án.

Câu hỏi: **dạy model làm toán thì nên dạy cái gì, và theo thứ tự nào?**

Năm model được dạy, rồi đem thử với bài CHƯA TỪNG THẤY ở ba độ dài khác nhau.

    bậc  học gì                              thử với
    ───  ───────────────────────────────────  ──────────────────────────
     1   cộng 1 chữ số, không nhớ (55 bài)    chính bảng đó
     2   cộng 1 chữ số, có nhớ (45 bài)       chính bảng đó
     3   cộng 2 chữ số, TRẢ LỜI THẲNG         2 / 3 / 4 chữ số
     4   cộng 2 chữ số, TỪNG BƯỚC             2 / 3 / 4 chữ số
     5   cộng 1-3 chữ số, TỪNG BƯỚC           2 / 3 / 4 chữ số

Ba câu hỏi mà bảng kết quả trả lời:

    1. Học thuộc bảng cộng có suy ra được bài dài hơn không?   (bậc 1, 2)
    2. Viết ra từng bước có giúp gì không?                     (bậc 3 vs 4)
    3. Học toàn số hai chữ số thì có làm được số ba chữ số
       không, dù có viết từng bước?                            (bậc 4 vs 5)

Câu 3 là chỗ dễ bị bất ngờ nhất. Xem phần kết luận ở cuối.
"""

from __future__ import annotations

import argparse

from .curriculum import LEVELS, Level
from .evaluate import evaluate
from .train import save, train_level

# Thử xa hơn lúc học, để xem có suy rộng ra không.
PROBE_LENGTHS = (2, 3, 4)


def _probe(digits: int, style: str) -> Level:
    return Level(0, f"{digits} chữ số", digits, style)


def _learn(number: int, args) -> tuple:
    level = LEVELS[number]
    print(f"[bậc {number}] {level.name}")

    model, tokenizer, history = train_level(
        level,
        steps=args.steps,
        examples=args.examples,
        seed=number,
        quiet=True,
    )

    print(f"    loss {history[0][1]:.3f} -> {history[-1][1]:.3f}")

    if not args.no_save:
        path = save(model, tokenizer, level, f"bac{number}")
        print(f"    lưu vào {path}")

    return model, tokenizer, level


def main_experiment(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepseek_math experiment")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--examples", type=int, default=20000)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--no-save", action="store_true", help="không lưu model")
    parser.add_argument("--skip", type=int, nargs="*", default=[], help="bỏ qua bậc nào")
    args = parser.parse_args(argv)

    print("=" * 72)
    print("THÍ NGHIỆM: dạy model làm toán theo từng bước")
    print("=" * 72)
    print(f"  {args.steps} bước học mỗi model · {args.examples:,} bài học · "
          f"{args.count} bài để thử\n")

    trained: dict[int, tuple] = {}

    for number in (1, 2, 3, 4, 5):
        if number in args.skip:
            continue
        trained[number] = _learn(number, args)

    # ------------------------------------------------------------------
    # Chấm điểm
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("KẾT QUẢ — tỉ lệ làm đúng, thử bằng bài CHƯA TỪNG THẤY")
    print("=" * 72)

    header = f"{'bậc':<4} {'học gì':<34} " + "".join(
        f"{d} chữ số".rjust(11) for d in PROBE_LENGTHS
    )
    print(header)
    print("-" * len(header))

    scores: dict[int, dict[int, float]] = {}

    for number, (model, tokenizer, level) in trained.items():
        style = level.style
        row: dict[int, float] = {}

        for digits in PROBE_LENGTHS:
            probe = _probe(digits, style)
            result = evaluate(model, tokenizer, probe, count=args.count, seed=1000 + digits)
            row[digits] = result["accuracy"]

        scores[number] = row

        label = level.name[:33]
        print(f"{number:<4} {label:<34} " + "".join(f"{row[d]:>10.1%} " for d in PROBE_LENGTHS))

    print()
    print("(bậc 3, 4, 5 học số HAI chữ số, nên cột '2 chữ số' là bài mới cùng độ dài;")
    print(" bậc 1, 2 học cả bảng một chữ số nên cột đó là bài dài hơn hẳn.)")

    # ------------------------------------------------------------------
    # Kết luận
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("KẾT LUẬN")
    print("=" * 72)

    _conclude(scores)

    return 0


def _conclude(scores: dict[int, dict[int, float]]) -> None:
    def get(level: int, digits: int) -> float:
        return scores.get(level, {}).get(digits, float("nan"))

    # 1. Học thuộc có suy ra được không?
    if 1 in scores:
        print("\n1. Học thuộc bảng cộng có suy ra được bài dài hơn không?")
        print(f"   bậc 1 học thuộc 55 bài, làm đúng {get(1, 2):.0%} bài hai chữ số.")
        if get(1, 2) < 0.2:
            print("   -> KHÔNG. Thuộc bảng cộng không giúp gì cho số dài hơn.")
            print("      Trong 'đầu' model, 47+25 chỉ là một dãy ký tự chưa từng thấy.")

    # 2. Viết từng bước có giúp không?
    if 3 in scores and 4 in scores:
        print("\n2. Viết ra từng bước có giúp gì không?")
        print(f"   cùng bài hai chữ số:  trả lời thẳng {get(3, 2):.0%}   "
              f"viết từng bước {get(4, 2):.0%}")
        if get(4, 2) > get(3, 2) + 0.2:
            print("   -> CÓ, rất rõ. Viết ra cho model chỗ để mà tính.")
            print("      Mỗi bước chỉ là 'đọc bước trước rồi viết tiếp' — việc")
            print("      mà attention làm được, còn tính ngầm thì không.")

    # 3. Bất ngờ: học toàn hai chữ số thì có làm được ba chữ số không?
    if 4 in scores and 5 in scores:
        print("\n3. Học toàn số HAI chữ số thì có làm được số BA chữ số không?")
        print(f"   bậc 4 (chỉ học 2 chữ số, viết từng bước): {get(4, 3):.0%}")
        print(f"   bậc 5 (học 1-3 chữ số, viết từng bước)  : {get(5, 3):.0%}")
        if get(5, 3) > get(4, 3) + 0.2:
            print("   -> KHÔNG, và đây là chỗ dễ bị bất ngờ nhất.")
            print("      Bậc 4 viết từng bước rất giỏi, nhưng nó học được đúng")
            print("      'bài toán này có HAI cột'. Gặp số ba chữ số nó vẫn viết")
            print("      hai cột — không phải vì không biết cộng, mà vì không")
            print("      biết là phải làm thêm cột.")
            print("      Muốn suy rộng ra thì phải cho nó thấy NHIỀU ĐỘ DÀI")
            print("      khác nhau lúc học (bậc 5), để nó học quy tắc")
            print("      'số cột = số chữ số' thay vì học 'có hai cột'.")

    if 5 in scores:
        print("\n4. Bậc 5 học tới 3 chữ số. Thử luôn 4 chữ số xem sao:")
        print(f"   4 chữ số (chưa từng học): {get(5, 4):.0%}")
        if get(5, 4) > 0.5:
            print("   -> Lần này thì suy rộng được. Vì nó đã học QUY TẮC,")
            print("      không phải học một độ dài cụ thể.")
        elif get(5, 4) > 0.1:
            print("   -> Suy rộng được một phần. Càng dài càng dễ sai.")
        else:
            print("   -> Chưa suy rộng ra được. Ba chữ số là mức cao nhất nó học.")

    print()
