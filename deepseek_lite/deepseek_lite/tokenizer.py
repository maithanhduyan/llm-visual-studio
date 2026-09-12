"""
tokenizer.py — Chữ biến thành số thế nào?

Giống hệt Cấp 2: mỗi ký tự là một số.

    "Học"  ->  [41, 62, 88]

Model thật của DeepSeek dùng BPE (mỗi mảnh chữ là một số, 129.000 mảnh).
Ở đây dùng ký tự cho dễ hiểu. Ý tưởng y hệt: chữ -> số.
"""

from __future__ import annotations

from collections.abc import Iterable


class Tokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))

        self.stoi = {ch: i for i, ch in enumerate(chars)}  # chữ -> số
        self.itos = {i: ch for ch, i in self.stoi.items()}  # số -> chữ
        self.chars = chars  # để lưu vào file model

    def encode(self, text: str) -> list[int]:
        """Chữ -> số. Ký tự lạ (model chưa từng thấy) thì bỏ qua."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens: Iterable[int]) -> str:
        """Số -> chữ."""
        return "".join(self.itos[t] for t in tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size})"

    @classmethod
    def load(cls, chars: list[str]) -> "Tokenizer":
        """Dựng lại tokenizer đã lưu trong file model."""
        return cls("".join(chars))
