"""
data.py — Đọc bài học và cắt thành từng đoạn để học.

Một bước học cần hai thứ:

    x = một đoạn văn
    y = cũng đoạn đó nhưng dịch lên 1 chữ  ->  đó là "đáp án" phải đoán

Ví dụ với đoạn "Học mãi thì giỏi":

    x = Học mãi thì giỏ
    y = ọc mãi thì giỏi

Chia bài thành hai phần: phần để HỌC và phần để THI.
Phần thi không bao giờ được dùng lúc học, nếu không điểm thi sẽ vô nghĩa.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .paths import DEFAULT_DATA_FILE


def load_text(path: str | Path | None = None) -> str:
    """Đọc bài học từ file. Mặc định là data/data.txt."""
    path = Path(path) if path else DEFAULT_DATA_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy bài học: {path}\n"
            "Hãy tạo file đó, hoặc trỏ --data sang file khác."
        )

    return path.read_text(encoding="utf-8")


def split_train_val(tokens: torch.Tensor, val_fraction: float = 0.1):
    """Cắt bài ra làm hai: phần học và phần thi.

    Lấy phần CUỐI làm phần thi, để phần học vẫn liền mạch từ đầu.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction ({val_fraction}) phải nằm trong [0, 1).")

    cut = int(len(tokens) * (1.0 - val_fraction))
    return tokens[:cut], tokens[cut:]


def get_batch(
    tokens: torch.Tensor,
    block_size: int,
    batch_size: int,
    device=None,
):
    """Bốc ngẫu nhiên vài đoạn văn trong bài."""
    if len(tokens) <= block_size + 1:
        raise ValueError(
            f"Bài học chỉ có {len(tokens)} token, không đủ để cắt đoạn dài {block_size}.\n"
            "Hãy viết thêm, hoặc giảm --block-size."
        )

    starts = torch.randint(len(tokens) - block_size - 1, (batch_size,))

    x = torch.stack([tokens[i : i + block_size] for i in starts])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in starts])

    if device is not None:
        x, y = x.to(device), y.to(device)

    return x, y
