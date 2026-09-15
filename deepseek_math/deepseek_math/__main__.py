"""
__main__.py — Chạy từng bước.

    python -m deepseek_math steps        xem năm bậc học
    python -m deepseek_math demo         xem vài bài ở mỗi bậc
    python -m deepseek_math data         xem dữ liệu huấn luyện đang có
    python -m deepseek_math export       ghi dữ liệu sinh ra xuống file
    python -m deepseek_math teach 1      dạy bậc 1
    python -m deepseek_math test 1       thử bậc 1
    python -m deepseek_math experiment   THÍ NGHIỆM CHÍNH: thẳng vs từng bước
"""

from __future__ import annotations

import argparse
import sys


def cmd_steps(argv=None) -> int:
    from .curriculum import LEVELS, describe

    print("=" * 68)
    print("Năm bậc học — vì sao phải theo đúng thứ tự này")
    print("=" * 68)
    print(describe())
    print()
    print("Bậc 1-2: bảng cộng một chữ số. Chỉ 100 bài, HỌC THUỘC được.")
    print("Bậc 3  : số hai chữ số, trả lời thẳng. Phải TÍNH, không thuộc được.")
    print("Bậc 4  : cũng số hai chữ số, nhưng viết ra từng bước.")
    print("Bậc 5  : số ba chữ số. Chưa học bao giờ — để xem có suy rộng ra không.")
    print()
    print("Thí nghiệm chính là bậc 3 so với bậc 4: cùng bài, khác cách viết.")
    _ = LEVELS
    return 0


def cmd_demo(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepseek_math demo")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args(argv)

    import random

    from .curriculum import LEVELS

    for number, level in LEVELS.items():
        print(f"\nBậc {number} — {level.name}")
        print("-" * 68)
        rng = random.Random(number)
        for problem in level.make(args.count, rng):
            print(f"  {level.render(problem)}")
    return 0


def cmd_export(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m deepseek_math export",
        description="Ghi dữ liệu sinh ra xuống file để mở mà đọc.",
    )
    parser.add_argument("--count", type=int, default=300, help="mỗi bậc bao nhiêu bài")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--level", type=int, default=None, help="chỉ ghi một bậc")
    args = parser.parse_args(argv)

    from .curriculum import LEVELS
    from .dataset import describe_file, export_all, export_level

    print("=" * 68)
    print("Ghi dữ liệu huấn luyện ra file")
    print("=" * 68)

    if args.level:
        paths = [export_level(LEVELS[args.level], args.count, args.seed)]
    else:
        paths = export_all(args.count, args.seed)

    for path in paths:
        print(f"  {describe_file(path)}")

    print()
    print("Mở file ra là đọc được: mỗi dòng một bài, dòng đầu là chú thích.")
    print("Model KHÔNG học từng dòng — nó học cả đoạn văn liền mạch, đoán ký tự tiếp theo.")
    return 0


def cmd_data(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m deepseek_math data",
        description="Xem dữ liệu huấn luyện đang có.",
    )
    parser.add_argument("--show", type=int, default=0, help="in ra bao nhiêu dòng đầu")
    args = parser.parse_args(argv)

    from .curriculum import LEVELS
    from .dataset import describe_file, load_text
    from .paths import DATA_DIR

    print("=" * 68)
    print("Dữ liệu huấn luyện")
    print("=" * 68)

    found = sorted(DATA_DIR.glob("bac*.txt")) if DATA_DIR.exists() else []

    if not found:
        print("  Chưa có file nào. Chạy trước:")
        print("      python -m deepseek_math export")
        print()
        print("  Bình thường thì KHÔNG cần file: dữ liệu được sinh ra ngay lúc học.")
        print("  Ghi ra file chỉ để mở mà đọc, hoặc để chạy lại đúng dữ liệu cũ.")
        return 0

    for path in found:
        print(f"  {describe_file(path)}")

    if args.show:
        for path in found:
            print()
            print(f"--- {path.name} ---")
            body = load_text(path).splitlines()
            for line in body[: args.show]:
                print(f"  {line}")

    _ = LEVELS
    return 0


def cmd_teach(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepseek_math teach")
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--examples", type=int, default=20000)
    parser.add_argument("--name", default=None, help="tên file lưu (mặc định: bacN)")
    parser.add_argument("--data", default=None, help="học từ file có sẵn thay vì sinh ra")
    args = parser.parse_args(argv)

    from .curriculum import LEVELS
    from .train import save, train_level

    level = LEVELS[args.level]
    print("=" * 68)
    print(f"Dạy bậc {level.number} — {level.name}")
    print(f"  cách viết: {level.style}   số chữ số: {level.digits}")
    print(f"  dữ liệu  : {args.data or f'sinh ra {args.examples:,} bài'}")
    print("=" * 68)

    model, tokenizer, history = train_level(
        level,
        steps=args.steps,
        examples=args.examples,
        data_file=args.data,
    )

    name = args.name or f"bac{level.number}"
    path = save(model, tokenizer, level, name)
    print(f"    đã lưu vào {path}")
    print(f"    loss: {history[0][1]:.3f} -> {history[-1][1]:.3f}")
    return 0


def cmd_test(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepseek_math test")
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--name", default=None)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args(argv)

    from .curriculum import LEVELS
    from .evaluate import evaluate
    from .train import load

    level = LEVELS[args.level]
    name = args.name or f"bac{level.number}"

    model, tokenizer = load(name)
    result = evaluate(model, tokenizer, level, count=args.count)

    print("=" * 68)
    print(f"Thử bậc {level.number} — {level.name}   (model: {name})")
    print("=" * 68)
    print(f"  đúng {result['correct']}/{result['count']}  =  {result['accuracy']:.1%}")
    if result["empty"]:
        print(f"  (model không viết gì ở {result['empty']} bài)")

    for example in result["examples"]:
        print(f"    {example['problem']}  muốn {example['want']}, "
              f"model viết {example['got']}   [{example['raw']}]")
    return 0


def cmd_experiment(argv=None) -> int:
    from .checks import main_experiment

    return main_experiment(argv)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        print(__doc__)
        return 0

    command, rest = argv[0], argv[1:]

    table = {
        "steps": cmd_steps,
        "demo": cmd_demo,
        "export": cmd_export,
        "data": cmd_data,
        "teach": cmd_teach,
        "test": cmd_test,
        "experiment": cmd_experiment,
    }

    if command not in table:
        print(f"Không biết lệnh {command!r}.")
        print(__doc__)
        return 2

    return table[command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
