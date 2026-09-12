"""
generate.py — Làm sao AI nói? (bản Cấp 3)

Cấp 2 sinh mỗi chữ bằng cách chạy lại CẢ CÂU từ đầu. Viết 200 chữ thì
chữ thứ 200 phải tính 200 lần — càng viết càng chậm.

Cấp 3 dùng đệm K/V (xem cache.py):

    Đọc câu mở đầu một lần   ->  cất K, V vào đệm
    Mỗi chữ tiếp theo        ->  chỉ tính 1 token mới, tra K/V trong đệm

Nhờ vậy viết chữ thứ 200 nhanh gần bằng viết chữ thứ nhất.

Chạy:
    python -m deepseek_lite.generate "Học mãi thì"
    python -m deepseek_lite.generate --prompt "Chuyên gia" --tokens 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .cache import KVCache
from .config import Config
from .model import DeepSeekLite
from .paths import DEFAULT_CHECKPOINT
from .tokenizer import Tokenizer


@torch.no_grad()
def generate_ids(
    model: DeepSeekLite,
    prompt_ids: list[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    use_cache: bool = True,
    device=None,
) -> list[int]:
    """Sinh ra một dãy token mới. Trả về đúng những token MỚI, không kèm câu mở đầu.

    use_cache=False để so sánh — kết quả phải giống hệt nhau.
    """
    model.eval()
    config = model.config

    if not prompt_ids:
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = KVCache(config, batch_size=1, device=device) if use_cache else None

    # Đọc cả câu mở đầu một lần.
    logits, _ = model(idx, cache=cache, start_pos=0)
    pos = idx.shape[1]

    generated: list[int] = []

    for _ in range(max_new_tokens):
        next_id = _sample(logits[:, -1, :], temperature, top_k)
        generated.append(int(next_id.item()))

        if use_cache:
            if pos >= config.max_seq_len:
                break
            # Chỉ đưa vào đúng một token mới.
            logits, _ = model(next_id.view(1, 1), cache=cache, start_pos=pos)
            pos += 1
        else:
            idx = torch.cat([idx, next_id.view(1, 1)], dim=1)
            if idx.shape[1] >= config.max_seq_len:
                break
            # Chạy lại cả câu từ đầu — chậm, nhưng để so sánh.
            logits, _ = model(idx)

    return generated


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    """Bốc một chữ theo xác suất.

    temperature nhỏ (0.2) -> nói chắc chắn, hay lặp lại
    temperature to (1.5)  -> nói sáng tạo, hay linh tinh
    temperature = 0       -> luôn chọn chữ có điểm cao nhất
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model: DeepSeekLite,
    tokenizer: Tokenizer,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    use_cache: bool = True,
    device=None,
) -> str:
    """Sinh chữ rồi trả về cả câu (kèm phần mở đầu)."""
    prompt_ids = tokenizer.encode(prompt)
    new_ids = generate_ids(
        model, prompt_ids, max_new_tokens, temperature, top_k, use_cache, device
    )
    return tokenizer.decode(prompt_ids + new_ids)


def load_checkpoint(path: str | Path | None = None, device=None):
    """Đọc lại model đã học xong."""
    path = Path(path) if path else DEFAULT_CHECKPOINT

    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có model: {path}\n"
            "Hãy chạy  python -m deepseek_lite.train  trước."
        )

    checkpoint = torch.load(path, map_location="cpu")

    tokenizer = Tokenizer.load(checkpoint["chars"])
    config = Config.from_dict(checkpoint["config"])

    model = DeepSeekLite(config)
    model.load_state_dict(checkpoint["model"])

    if device is not None:
        model = model.to(device)

    model.eval()
    return model, tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepseek_lite.generate",
        description="Cho deepseek_lite nói.",
    )

    parser.add_argument("prompt", nargs="?", default="Học mãi thì", help="câu mở đầu")
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokens", type=int, default=200, help="viết thêm bao nhiêu chữ")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                        help="tắt đệm K/V (chậm hơn nhiều, để so sánh)")
    parser.add_argument("--device", default="cpu")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    model, tokenizer = load_checkpoint(args.model, device=args.device)

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_cache=args.use_cache,
        device=args.device,
    )

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
