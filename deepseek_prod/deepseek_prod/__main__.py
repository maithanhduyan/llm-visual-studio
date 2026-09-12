"""
__main__.py — Chạy mọi thứ từ một chỗ.

    python -m deepseek_prod info          xem model và cách chia
    python -m deepseek_prod check         chứng minh từng tính năng bằng số
    python -m deepseek_prod check tp      chỉ chạy phần tensor parallel
    python -m deepseek_prod bench         đo tốc độ
    python -m deepseek_prod serve         chạy máy phục vụ
"""

from __future__ import annotations

import argparse
import sys

CHECKS = {
    "tp": "tensor parallel — cắt ngang một tầng ra nhiều máy",
    "pp": "pipeline parallel — cắt dọc model ra nhiều máy",
    "ep": "expert parallel — mỗi máy giữ vài chuyên gia",
    "zero": "ZeRO-1 — chia trạng thái optimizer",
    "cache": "paged KV cache — đệm chia khối",
    "batch": "continuous batching — gom nhiều câu hỏi vào một lô",
    "quant": "quantization — nén model xuống 8 bit",
    "spec": "speculative decoding — đoán nhanh rồi kiểm lại",
    "rope": "RoPE scaling — kéo dài ngữ cảnh",
}


def cmd_info(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepseek_prod info", description="Xem model và cách chia.")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    args = parser.parse_args(argv)

    from .checks import load_level3
    from .config import ParallelConfig

    _, tokenizer, model = load_level3()
    total, active = model.parameter_counts()
    layout = ParallelConfig(tp=args.tp, pp=args.pp, dp=args.dp)

    print("=" * 68)
    print("deepseek_prod — Cấp 4")
    print("=" * 68)
    print("Model dùng lại nguyên của Cấp 3:")
    print(model)
    print()
    print(f"  tổng thông số    : {total:,}")
    print(f"  dùng mỗi token   : {active:,}  ({active / total:.0%})")
    print(f"  từ điển          : {len(tokenizer)} ký tự")
    print()
    print(f"Cách chia         : {layout.describe()}")
    if layout.is_parallel:
        print(f"  mỗi máy giữ khoảng {total / layout.tp / layout.pp:,.0f} thông số")
    print()
    print("Chạy 'python -m deepseek_prod check' để chứng minh từng tính năng.")
    return 0


def cmd_check(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepseek_prod check", description="Chứng minh bằng số.")
    parser.add_argument(
        "which",
        nargs="*",
        default=None,
        help=f"chọn một hoặc nhiều: {', '.join(CHECKS)}  (bỏ trống = chạy hết)",
    )
    parser.add_argument("--world", type=int, default=2, help="số tiến trình cho các phép thử song song")
    args = parser.parse_args(argv)

    from . import checks

    wanted = args.which or list(CHECKS)
    unknown = [name for name in wanted if name not in CHECKS]
    if unknown:
        print(f"Không biết phần {', '.join(unknown)}. Chọn trong: {', '.join(CHECKS)}")
        return 2

    print("=" * 68)
    print("Chứng minh từng tính năng của Cấp 4")
    print("=" * 68)

    failures = 0
    for name in wanted:
        print(f"\n[{name}] {CHECKS[name]}")
        print("-" * 68)

        runner = getattr(checks, f"check_{_function_name(name)}", None)
        if runner is None:
            print("  (chưa làm)")
            continue

        try:
            result = runner(world=args.world) if _needs_world(name) else runner()
        except Exception as exc:  # noqa: BLE001
            print(f"  LỖI: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        for line in result["details"]:
            print(f"  {line}")
        print(f"  -> {'ĐẠT' if result['ok'] else 'KHÔNG ĐẠT'}")
        if not result["ok"]:
            failures += 1

    print("\n" + "=" * 68)
    print("Tất cả đều đạt." if failures == 0 else f"{failures} phần KHÔNG đạt.")
    return 1 if failures else 0


def _function_name(name: str) -> str:
    return {
        "tp": "tensor_parallel",
        "pp": "pipeline",
        "ep": "expert_parallel",
        "zero": "zero",
        "cache": "paged_cache",
        "batch": "continuous_batching",
        "quant": "quantization",
        "spec": "speculative",
        "rope": "rope_scaling",
    }[name]


def _needs_world(name: str) -> bool:
    return name in {"tp", "pp", "ep", "zero"}


def cmd_bench(argv=None) -> int:
    """Đo tốc độ phần phục vụ — không cần nhiều tiến trình."""
    parser = argparse.ArgumentParser(
        prog="python -m deepseek_prod bench",
        description="Đo tốc độ phục vụ.",
    )
    parser.parse_args(argv)

    return cmd_check(["cache", "batch", "quant"])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        print(__doc__)
        return 0

    command, rest = argv[0], argv[1:]

    if command == "info":
        return cmd_info(rest)
    if command == "check":
        return cmd_check(rest)
    if command == "bench":
        return cmd_bench(rest)

    print(f"Không biết lệnh {command!r}.")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
