"""
train.py — Dạy model học một bậc.

Khác các cấp trước ở một chỗ quan trọng: **dữ liệu sinh ra được, vô hạn**.
Cấp 1-4 phải đi nhặt một bài văn rồi học thuộc nó. Ở đây muốn bao nhiêu bài
cũng có, và luôn sinh được bài MỚI để thử.

Nhờ vậy mới đo được thứ đáng đo: **bài chưa từng thấy**.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .base import Tokenizer
from .curriculum import Level, build_text
from .model import build_model
from .paths import ensure_run_dir


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


def train_level(
    level: Level,
    steps: int = 1500,
    examples: int = 20000,
    batch_size: int = 32,
    block_size: int = 48,
    lr: float = 3e-3,
    seed: int = 0,
    quiet: bool = False,
    data_file=None,
    **model_overrides,
):
    """Dạy model một bậc. Trả về (model, tokenizer, lịch sử loss).

    `data_file`: nếu có, đọc dữ liệu từ file thay vì sinh ra. Dùng khi muốn
    chạy lại ĐÚNG dữ liệu cũ, hoặc khi muốn tự soạn bài cho model học.
    """
    torch.manual_seed(seed)

    if data_file:
        from .dataset import load_text

        text = load_text(data_file)
    else:
        text = build_text(level, examples, seed=seed)

    tokenizer = Tokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    if len(tokens) < block_size + 2:
        raise ValueError(
            f"Dữ liệu chỉ có {len(tokens)} token, không đủ cắt đoạn dài {block_size}. "
            "Sinh nhiều bài hơn, hoặc giảm --block-size."
        )

    model = build_model(tokenizer.vocab_size, **model_overrides)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history: list[list[float]] = []
    started = time.time()

    model.train()

    for step in range(steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps, lr, warmup=min(100, steps // 10))

        x, y = get_batch(tokens, block_size, batch_size)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_router_bias()

        history.append([step, round(loss.item(), 4)])

        if not quiet and (step % 250 == 0 or step == steps):
            print(f"    bước {step:5d}   loss = {loss.item():.4f}")

    model.eval()
    elapsed = time.time() - started

    if not quiet:
        print(f"    xong sau {elapsed:.0f} giây · {len(tokenizer)} ký tự · "
              f"{sum(p.numel() for p in model.parameters()):,} thông số")

    return model, tokenizer, history


def save(model, tokenizer, level: Level, name: str) -> str:
    import torch as _torch

    ensure_run_dir()
    from .paths import RUNS_DIR

    path = RUNS_DIR / f"{name}.pt"
    _torch.save(
        {
            "config": {k: v for k, v in model.config.__dict__.items()},
            "chars": tokenizer.chars,
            "model": model.state_dict(),
            "level": level.number,
            "style": level.style,
        },
        path,
    )
    return str(path)


def load(name: str):
    import torch as _torch

    from .base import Config as _Config
    from .base import DeepSeekLite as _Model
    from .paths import RUNS_DIR

    checkpoint = _torch.load(RUNS_DIR / f"{name}.pt", map_location="cpu")

    tokenizer = Tokenizer.load(checkpoint["chars"])
    model = _Model(_Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, tokenizer


__all__ = ["train_level", "save", "load", "get_batch", "lr_at", "F"]
