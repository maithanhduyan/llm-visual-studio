"""
__main__.py — Chạy mọi thứ từ một chỗ.

    python -m deepseek_lite info                 xem model có gì (chưa cần học)
    python -m deepseek_lite train                dạy nó học
    python -m deepseek_lite train --steps 500    học ít cho nhanh
    python -m deepseek_lite generate "Học mãi"   cho nó nói
    python -m deepseek_lite benchmark            đo tốc độ

Mọi tham số của từng lệnh đều được chuyển thẳng xuống module tương ứng,
nên `python -m deepseek_lite train --help` cũng chạy được.
"""

from __future__ import annotations

import argparse
import sys

from . import benchmark, generate, train
from .config import Config
from .model import DeepSeekLite

COMMANDS = ("info", "train", "generate", "benchmark")


def info(argv=None) -> int:
    """In ra model mặc định có gì. Chưa cần học, chưa cần dữ liệu."""
    parser = argparse.ArgumentParser(
        prog="python -m deepseek_lite info",
        description="Xem model có gì.",
    )
    parser.add_argument("--d-model", dest="d_model", type=int, default=None)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=None)
    parser.add_argument("--n-streams", dest="n_streams", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    args = parser.parse_args(argv)

    overrides = {
        name: getattr(args, name)
        for name in ("d_model", "n_layers", "n_streams", "window")
        if getattr(args, name) is not None
    }
    config = Config(**overrides)
    model = DeepSeekLite(config)

    total, active = model.parameter_counts()

    print("=" * 66)
    print("deepseek_lite — Cấp 3")
    print("=" * 66)
    print(model)
    print()
    print(f"  thông số        : {total:,}")
    print(f"  dùng mỗi token  : {active:,}  ({active / total:.0%})")
    print(f"  đệm K/V         : {model.cache_memory_bytes() / 1024:.1f} KB"
          f" cho {config.max_seq_len} token")
    print()
    print("Thử đổi vài con số xem model to nhỏ thế nào:")
    print("  python -m deepseek_lite info --n-layers 6")
    print("  python -m deepseek_lite info --d-model 384 --n-layers 6")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        print(__doc__)
        return 0

    command, rest = argv[0], argv[1:]

    if command not in COMMANDS:
        print(f"Không biết lệnh {command!r}. Chọn một trong: {', '.join(COMMANDS)}")
        print(__doc__)
        return 2

    if command == "info":
        return info(rest)
    if command == "train":
        return train.main(rest)
    if command == "generate":
        return generate.main(rest)
    return benchmark.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
