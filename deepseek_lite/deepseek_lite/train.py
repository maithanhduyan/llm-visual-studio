"""
train.py — Làm sao AI học? (bản Cấp 3)

Cấp 2 chỉ có một vòng lặp đơn giản. Cấp 3 thêm ba thứ mà mọi LLM thật đều có:

1. LỊCH HỌC (learning rate schedule)
   Lúc đầu học chậm để khỏi "vấp", giữa chừng học nhanh, cuối thì học
   chậm lại để ổn định. Học với một tốc độ suốt thì model khó hội tụ.

2. CHIA BÀI ĐỂ HỌC VÀ ĐỂ THI (train/val split)
   Phần "thi" không bao giờ được dùng lúc học.
   Nếu điểm học rất thấp mà điểm thi rất cao -> model đang HỌC VẸT.
   Cấp 2 không có phần thi nên không thấy được điều này.

3. GOM NHIỀU LÔ (gradient accumulation)
   Muốn học một lô to mà máy không đủ nhớ, thì chia lô to thành nhiều
   lô nhỏ rồi cộng dồn gradient lại trước khi sửa trọng số.

Chạy:
    python -m deepseek_lite.train
    python -m deepseek_lite.train --steps 500
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from .config import Config
from .data import get_batch, load_text, split_train_val
from .model import DeepSeekLite
from .paths import DEFAULT_CHECKPOINT, DEFAULT_DATA_FILE, ensure_run_dir
from .tokenizer import Tokenizer


def pick_device(name: str = "auto") -> torch.device:
    """Chọn nơi chạy: GPU nếu có, không thì CPU."""
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def lr_at(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Tốc độ học ở bước thứ `step`.

    Lên dần trong `warmup_steps` bước đầu, rồi xuống theo đường cosin.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    # Từ 1.0 xuống 0.1 — không xuống 0 hẳn, để cuối vẫn học được chút.
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=20) -> dict:
    """Đo loss trên cả phần học lẫn phần thi."""
    model.eval()  # tắt dropout, tắt đếm chuyên gia

    result = {}
    for name, tokens in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(tokens, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        result[name] = losses.mean().item()

    model.train()
    return result


def build_config(args, vocab_size: int) -> Config:
    """Dựng Config từ tham số dòng lệnh."""
    overrides = {
        "vocab_size": vocab_size,
        # Câu lúc HỌC chỉ dài bằng block_size, nhưng lúc SINH CHỮ có thể dài
        # hơn nhiều. Để dư ra, nếu không sinh quá vài chục chữ là hết chỗ.
        "max_seq_len": max(args.block_size, 512),
    }

    for name in ("d_model", "n_layers", "n_experts", "n_streams", "window"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value

    return Config(**overrides)


def train(args) -> dict:
    """Toàn bộ quá trình học. Trả về kết quả để test hoặc để vẽ biểu đồ."""
    torch.manual_seed(args.seed)

    device = pick_device(args.device)

    # --- 1. Đọc bài ----------------------------------------------------
    text = load_text(args.data)
    tokenizer = Tokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    train_tokens, val_tokens = split_train_val(tokens, args.val_fraction)
    splits = {"học": train_tokens}
    if len(val_tokens) > args.block_size + 1:
        splits["thi"] = val_tokens

    # --- 2. Dựng model -------------------------------------------------
    config = build_config(args, tokenizer.vocab_size)
    model = DeepSeekLite(config).to(device)

    total_params, active_params = model.parameter_counts()

    print("=" * 66)
    print("deepseek_lite — Cấp 3")
    print("=" * 66)
    print(f"Bài học   : {len(text)} ký tự, {tokenizer.vocab_size} ký tự khác nhau")
    print(f"            {len(train_tokens)} token để học, {len(val_tokens)} token để thi")
    print(f"Model     : {total_params:,} thông số (mỗi token chỉ dùng {active_params:,})")
    print(f"            {config.n_layers} tầng | GQA {config.n_heads}Q/{config.n_kv_heads}KV"
          f" | cửa sổ {config.window or 'không'} | {config.n_streams} dòng")
    print(f"            {config.n_experts} chuyên gia riêng + {config.n_shared} dùng chung,"
          f" chọn {config.top_k}")
    print(f"Đệm K/V   : {model.cache_memory_bytes() / 1024:.1f} KB cho {config.max_seq_len} token")
    print(f"Thiết bị  : {device}")
    print(f"Học       : {args.steps} bước, lô {args.batch_size}x{args.block_size}"
          f"{f' x{args.grad_accum} (gom)' if args.grad_accum > 1 else ''}, lr {args.lr}")

    # --- 3. Học --------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    history: list[list[float]] = []
    started = time.time()

    print(f"\nBắt đầu học...\n")
    header = f"{'bước':>6} {'loss':>8} {'lr':>9}"
    if "thi" in splits:
        header += f" {'loss thi':>9}"
    header += f" {'chia đều':>9}"
    print(header)
    print("-" * len(header))

    model.train()

    for step in range(args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gom nhiều lô nhỏ thành một lô to.
        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_tokens, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            step_loss += loss.item() / args.grad_accum

        # Chặn gradient quá to, để một lô xấu không làm hỏng cả model.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Đọc bảng chia việc TRƯỚC khi nó bị xoá.
        loads = [share.max().item() for share in model.router_loads()]
        worst_load = max(loads) if loads else 0.0

        # share của mọi chuyên gia cộng lại = 1, nên chia đều hoàn hảo
        # là mỗi chuyên gia nhận đúng 1/n_experts.
        perfect = 1.0 / config.n_experts

        # Rồi mới chỉnh điểm thiên vị cho bước sau.
        model.update_router_bias()

        history.append([step, round(step_loss, 4), round(lr, 8), round(worst_load, 4)])

        if step % args.log_every == 0 or step == args.steps:
            line = f"{step:>6} {step_loss:>8.3f} {lr:>9.2e}"
            if "thi" in splits and (step % args.eval_every == 0 or step == args.steps):
                losses = estimate_loss(
                    model, splits, args.block_size, args.batch_size, device, args.eval_iters
                )
                line += f" {losses['thi']:>9.3f}"
            else:
                line += f" {'':>9}"
            # 1.00x = chia đều hoàn hảo. 4.00x = chuyên gia đông nhất
            # đang nhận gấp 4 lần phần của nó.
            line += f" {worst_load / perfect:>8.2f}x"
            print(line)

    elapsed = time.time() - started

    # --- 4. Kết quả cuối -----------------------------------------------
    final = estimate_loss(model, splits, args.block_size, args.batch_size, device, args.eval_iters)

    print(f"\nHọc xong sau {elapsed:.1f} giây ({args.steps / max(elapsed, 1e-9):.1f} bước/giây)")
    print(f"  loss phần học : {final['học']:.3f}")

    if "thi" in splits:
        print(f"  loss phần thi : {final['thi']:.3f}")
        gap = final["thi"] - final["học"]
        if gap > 0.7:
            print(f"  -> chênh {gap:.2f}: model đang HỌC VẸT.")
            print("     Bài học còn ít so với sức chứa của model.")
            print("     Muốn học thật thì cho nó đọc thêm nhiều bài khác nhau.")
        else:
            print(f"  -> chênh {gap:.2f}: học thật, không phải học vẹt.")

    # --- 5. Lưu lại ----------------------------------------------------
    if args.out:
        args.out = Path(args.out)
        ensure_run_dir()
        args.out.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "config": config.to_dict(),
                "chars": tokenizer.chars,
                "model": model.state_dict(),
                "history": history,
                "steps": args.steps,
            },
            args.out,
        )
        print(f"\nĐã lưu model vào {args.out}")

    return {
        "history": history,
        "final_loss": final,
        "seconds": elapsed,
        "total_params": total_params,
        "active_params": active_params,
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepseek_lite.train",
        description="Dạy deepseek_lite học đọc tiếng Việt.",
    )

    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE, help="file bài học")
    parser.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT, help="nơi lưu model")

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1, help="gom mấy lô nhỏ thành một lô to")

    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--clip", type=float, default=1.0)

    parser.add_argument("--val-fraction", type=float, default=0.1, help="phần bài để thi")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")

    # Mấy núm để thử nghiệm cho vui.
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=None)
    parser.add_argument("--n-experts", dest="n_experts", type=int, default=None)
    parser.add_argument("--n-streams", dest="n_streams", type=int, default=None)
    parser.add_argument("--window", type=int, default=None, help="0 = nhìn hết cả câu")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
