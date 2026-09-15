"""
train.py — Dạy model nhỏ bắt chước chuyên gia.

Đây là học bắt chước (imitation learning): không có phần thưởng, không có
thử-sai. Chuyên gia làm gì thì model học làm đúng như vậy.

Ưu điểm: học nhanh, ổn định, không cần mô phỏng triệu chuyến bay.
Nhược điểm: **model không bao giờ giỏi hơn thầy**. Nó chỉ bắt chước, kể cả
những chỗ thầy làm dở. Muốn giỏi hơn thì phải dùng học tăng cường — và đó
là một dự án khác.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .base import Tokenizer
from .dataset import generate, save, split
from .paths import RUNS_DIR, ensure_dirs
from .policy import build_model
from .vehicle import Vehicle


def get_batch(tokens: torch.Tensor, block_size: int, batch_size: int):
    starts = torch.randint(max(len(tokens) - block_size - 1, 1), (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in starts])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train_pilot(
    vehicle: Vehicle | None = None,
    flights: int = 60,
    steps: int = 1200,
    batch_size: int = 32,
    block_size: int = 32,
    lr: float = 3e-3,
    seed: int = 0,
    quiet: bool = False,
    examples: list[str] | None = None,
    data_name: str = "pilot",
):
    """Bay bằng chuyên gia để lấy dữ liệu, rồi dạy model bắt chước.

    `examples`: nếu có, dùng luôn đống dữ liệu này thay vì bay lại. Dùng cho
    các vòng DAgger.
    """
    ensure_dirs()
    torch.manual_seed(seed)

    vehicle = vehicle or Vehicle()

    if examples is None:
        if not quiet:
            print(f"    đang bay {flights} chuyến bằng chuyên gia...")

        examples = generate(vehicle, flights=flights, seed=seed)

    train_text, val_text = split(examples)

    data_path = save(examples, RUNS_DIR.parent / "data" / f"{data_name}.txt")

    tokenizer = Tokenizer(train_text + val_text)

    train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    model = build_model(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if not quiet:
        print(f"    {len(examples):,} ví dụ · {len(tokenizer)} ký tự · "
              f"{sum(p.numel() for p in model.parameters()):,} thông số")
        print(f"    dữ liệu lưu ở {data_path}")

    history: list[list[float]] = []
    started = time.time()

    model.train()

    for step in range(steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps, lr, warmup=min(100, steps // 10))

        x, y = get_batch(train_tokens, block_size, batch_size)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_router_bias()

        history.append([step, round(loss.item(), 4)])

        if not quiet and (step % 300 == 0 or step == steps):
            print(f"    bước {step:5d}   loss = {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        x, y = get_batch(val_tokens, block_size, batch_size)
        _, val_loss = model(x, y)

    if not quiet:
        print(f"    xong sau {time.time() - started:.0f} giây · "
              f"loss học {history[-1][1]:.4f} · loss thi {val_loss.item():.4f}")

    return model, tokenizer, history, val_loss.item()


def save_pilot(model, tokenizer, name: str = "pilot") -> str:
    ensure_dirs()
    path = RUNS_DIR / f"{name}.pt"

    torch.save(
        {
            "config": dict(model.config.__dict__),
            "chars": tokenizer.chars,
            "model": model.state_dict(),
        },
        path,
    )

    return str(path)
