"""
benchmark.py — Đo xem mấy chiêu của Cấp 3 có thật sự giúp gì không.

Nói "nhanh hơn" mà không đo thì không ai tin. File này đo bốn thứ:

    1. Đệm K/V có cho ra KẾT QUẢ GIỐNG HỆT không?
       Nếu khác thì đệm vô dụng, dù nhanh đến đâu.

    2. Đệm K/V nhanh hơn bao nhiêu lần?

    3. Đệm tốn bao nhiêu bộ nhớ, và GQA tiết kiệm được bao nhiêu?

    4. Cửa sổ trượt tiết kiệm được gì khi câu đã dài?

Chạy:
    python -m deepseek_lite.benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .cache import KVCache
from .config import Config
from .generate import generate_ids, load_checkpoint
from .paths import DEFAULT_CHECKPOINT


def _time_decode(base_config, window: int, context: int, batch: int, steps: int = 150) -> float:
    """Đo thời gian MỘT tầng attention khi sinh từng chữ (T = 1).

    Đo riêng attention để thấy rõ cửa sổ trượt tác động tới đâu, thay vì
    đo cả model rồi bị lẫn với hàng trăm phép tính khác.

    Trả về số mili-giây cho một bước.
    """
    from .attention import Attention

    config = Config(**{**base_config.to_dict(), "window": window})
    attention = Attention(config, layer_index=0).eval()

    cache = KVCache(config, batch_size=batch)
    cache.length = context  # giả vờ đã cất sẵn `context` token vào đệm

    x = torch.randn(batch, 1, config.d_model)

    with torch.no_grad():
        for _ in range(10):  # chạy nóng
            attention(x, cache, start_pos=context)

        started = time.perf_counter()
        for _ in range(steps):
            attention(x, cache, start_pos=context)

    return (time.perf_counter() - started) / steps * 1000


@torch.no_grad()
def max_logit_difference(model, ids: torch.Tensor, config) -> float:
    """Chạy cả câu một lần, rồi chạy lại bằng đệm. Trả về mức lệch lớn nhất.

    Đây là cách kiểm tra chắc chắn nhất, vì nó so sánh thẳng điểm model cho
    từng chữ — không đi qua bước bốc thăm ngẫu nhiên.
    """
    model.eval()

    full, _ = model(ids)

    cache = KVCache(config, batch_size=1)
    split = max(1, ids.shape[1] // 2)

    prefill, _ = model(ids[:, :split], cache=cache, start_pos=0)
    parts = [prefill]

    for t in range(split, ids.shape[1]):
        step, _ = model(ids[:, t : t + 1], cache=cache, start_pos=t)
        parts.append(step)

    incremental = torch.cat(parts, dim=1)
    return (full - incremental).abs().max().item()


def _time_generation(model, prompt_ids, tokens, use_cache, device, seed=0) -> tuple[float, list[int]]:
    """Chạy một lần sinh chữ và bấm giờ. Trả về (số giây, dãy token sinh ra).

    Gieo lại hạt giống ngẫu nhiên trước mỗi lần chạy, để hai cách chạy
    bốc thăm ra cùng một dãy số. Không làm vậy thì không so sánh được.
    """
    # Chạy nóng một chút trước, để không đo cả thời gian khởi động.
    torch.manual_seed(seed)
    generate_ids(model, prompt_ids, max_new_tokens=3, temperature=0.8,
                 use_cache=use_cache, device=device)

    torch.manual_seed(seed)
    started = time.perf_counter()
    ids = generate_ids(model, prompt_ids, max_new_tokens=tokens, temperature=0.8,
                       use_cache=use_cache, device=device)
    return time.perf_counter() - started, ids


def run_benchmark(args) -> dict:
    device = torch.device(args.device)

    model, tokenizer = load_checkpoint(args.model, device=device)
    config = model.config

    print("=" * 66)
    print("deepseek_lite — đo tốc độ")
    print("=" * 66)
    print(f"Model: {config.n_layers} tầng | GQA {config.n_heads}Q/{config.n_kv_heads}KV"
          f" | cửa sổ {config.window or 'không'} | {config.n_streams} dòng")
    print(f"Sinh {args.tokens} token mỗi lần đo.")

    prompt_ids = tokenizer.encode(args.prompt)
    results = {}

    # --- 1. Đệm có đúng không? -----------------------------------------
    print("\n[1] Đệm K/V có cho ra kết quả giống hệt không?")
    print("-" * 66)

    # Cách chắc chắn nhất: so sánh thẳng điểm model cho từng chữ.
    probe = tokenizer.encode("Chuyên gia giỏi toán trả lời câu hỏi.")
    if len(probe) < 4:
        probe = torch.randint(0, config.vocab_size, (30,)).tolist()

    probe = probe[: config.max_seq_len]
    diff = max_logit_difference(model, torch.tensor([probe]), config)
    results["max_logit_diff"] = diff

    print("  So sánh điểm model cho từng chữ, giữa chạy cả câu và chạy có đệm:")
    print(f"    lệch lớn nhất: {diff:.2e}")
    print(f"    -> {'GIỐNG HỆT NHAU' if diff < 1e-4 else 'KHÁC NHAU — đệm bị lỗi!'}")

    # Rồi so sánh cả chữ sinh ra, với cùng một hạt giống ngẫu nhiên.
    _, with_cache = _time_generation(model, prompt_ids, args.tokens, True, device)
    _, without_cache = _time_generation(model, prompt_ids, args.tokens, False, device)

    same = with_cache == without_cache
    results["identical"] = same

    print("\n  Rồi so sánh chữ sinh ra (cùng hạt giống ngẫu nhiên):")
    print(f"    có đệm   : {tokenizer.decode(prompt_ids + with_cache)[:58]!r}")
    print(f"    không đệm: {tokenizer.decode(prompt_ids + without_cache)[:58]!r}")
    print(f"    -> {'GIỐNG HỆT NHAU' if same else 'KHÁC NHAU'}")

    if not same:
        first = next(
            (i for i, (a, b) in enumerate(zip(with_cache, without_cache)) if a != b),
            min(len(with_cache), len(without_cache)),
        )
        print(f"       Lệch từ token thứ {first}.")

    # --- 2. Nhanh hơn bao nhiêu? ---------------------------------------
    print("\n[2] Đệm K/V nhanh hơn bao nhiêu lần?")
    print("-" * 66)

    t_with, _ = _time_generation(model, prompt_ids, args.tokens, True, device)
    t_without, _ = _time_generation(model, prompt_ids, args.tokens, False, device)

    speed_with = args.tokens / t_with
    speed_without = args.tokens / t_without

    results["tok_per_s_cached"] = speed_with
    results["tok_per_s_plain"] = speed_without
    results["speedup"] = speed_with / speed_without

    print(f"  không đệm: {speed_without:7.1f} token/giây  ({t_without:.2f} giây)")
    print(f"  có đệm   : {speed_with:7.1f} token/giây  ({t_with:.2f} giây)")
    print(f"  -> nhanh hơn {speed_with / speed_without:.1f} lần")
    print("     Câu càng dài thì khoảng cách càng lớn: không đệm thì mỗi chữ")
    print("     phải tính lại cả câu, còn có đệm thì luôn chỉ tính 1 chữ.")

    # --- 3. Đệm tốn bao nhiêu bộ nhớ? ----------------------------------
    print("\n[3] Đệm K/V tốn bao nhiêu bộ nhớ?")
    print("-" * 66)

    head_dim = config.d_model // config.n_heads
    per_head = 2 * config.max_seq_len * head_dim * 4 * config.n_layers

    with_gqa = per_head * config.n_kv_heads
    without_gqa = per_head * config.n_heads

    results["cache_bytes"] = with_gqa

    print(f"  GQA: {config.n_kv_heads} đầu K/V thay vì {config.n_heads}.")
    print(f"  đệm cho {config.max_seq_len} token : {with_gqa / 1024:8.1f} KB")
    print(f"  nếu không dùng GQA  : {without_gqa / 1024:8.1f} KB")
    print(f"  -> tiết kiệm {without_gqa / with_gqa:.1f} lần")

    # --- 4. Cửa sổ trượt -------------------------------------------------
    print("\n[4] Cửa sổ trượt tiết kiệm được gì?")
    print("-" * 66)

    if config.window == 0:
        print("  Model này không dùng cửa sổ trượt (window = 0) nên bỏ qua phần này.")
    else:
        context = max(32, min(args.long_prompt, config.max_seq_len - 8))

        print("  Cửa sổ trượt KHÔNG phải lúc nào cũng nhanh hơn.")
        print("  Nó chỉ có lợi khi phần tính attention thật sự chiếm nhiều thời gian,")
        print("  nghĩa là khi phục vụ NHIỀU câu cùng lúc — đúng như hệ thống thật.")
        print(f"\n  Đo riêng một tầng attention, câu đã dài {context} token, sinh từng chữ:\n")
        print(f"  {'batch':>6} {'không cửa sổ':>14} {'cửa sổ ' + str(config.window):>12}"
              f" {'nhanh hơn':>11}")

        for batch in args.batches:
            plain = _time_decode(config, 0, context, batch)
            windowed = _time_decode(config, config.window, context, batch)

            results[f"decode_batch{batch}_plain_ms"] = plain
            results[f"decode_batch{batch}_window_ms"] = windowed

            note = "   <- chậm hơn!" if plain < windowed else ""
            print(f"  {batch:>6} {plain:>11.2f} ms {windowed:>9.2f} ms"
                  f" {plain / windowed:>10.2f}x{note}")

        print("\n  Ở batch 1, cắt bớt K/V còn tốn hơn phần tiết kiệm được: mỗi phép")
        print("  tính quá nhỏ nên chi phí gọi hàm lấn át. Batch càng lớn thì phần")
        print("  tính toán càng lớn, và cửa sổ trượt càng có lợi.")
        print("  (Lúc HỌC, cửa sổ chỉ che bằng mặt nạ nên không bỏ được tính toán.)")

    print("\n" + "=" * 66)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepseek_lite.benchmark",
        description="Đo tốc độ và kiểm tra đệm K/V.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="Học mãi thì")
    parser.add_argument("--tokens", type=int, default=100, help="sinh bao nhiêu token mỗi lần đo")
    parser.add_argument("--long-prompt", type=int, default=500, help="độ dài câu cho phần 4")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64],
                        help="các batch size để đo ở phần 4")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
