"""
dataset.py — Ghi dữ liệu sinh ra xuống file, và đọc lại.

Vì sao dữ liệu ở dự án này KHÔNG có sẵn trong repo:

    Các cấp trước phải đi nhặt một bài văn rồi học thuộc nó, nên phải có
    file data.txt. Ở đây bài toán SINH RA ĐƯỢC, vô hạn, nên không cần chứa
    sẵn — và đó chính là điều làm thí nghiệm này làm được: muốn thử thì cứ
    sinh bài MỚI, chắc chắn model chưa từng thấy.

Nhưng sinh ra lúc chạy thì có hai cái thiếu, và file này bù vào:

    1. KHÔNG NHÌN THẤY ĐƯỢC. Muốn biết model học cái gì thì phải mở ra xem.
    2. KHÔNG LẶP LẠI ĐƯỢC. Cùng một hạt giống thì ra cùng dữ liệu, nhưng chỉ
       khi nào còn nhớ hạt giống đó. Ghi ra file thì khỏi phải nhớ.

Nên: sinh ra lúc chạy là mặc định, còn muốn xem hay muốn cố định thì ghi ra.
"""

from __future__ import annotations

from pathlib import Path

from .curriculum import LEVELS, Level, build_text
from .paths import DATA_DIR


def export_level(level: Level, count: int, seed: int, path: Path | None = None) -> Path:
    """Sinh dữ liệu của một bậc rồi ghi ra file để mở mà đọc."""
    path = Path(path) if path else DATA_DIR / f"bac{level.number}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# Bậc {level.number} — {level.name}\n"
        f"# {count} bài, hạt giống {seed}\n"
        f"# Mỗi dòng là một bài. Model học bằng cách đoán ký tự tiếp theo\n"
        f"# trong cả đoạn văn này, không phải học từng dòng một.\n"
    )

    path.write_text(header + build_text(level, count, seed=seed), encoding="utf-8")
    return path


def export_all(count: int = 300, seed: int = 7) -> list[Path]:
    """Ghi dữ liệu của cả năm bậc. Đây là thứ có sẵn trong repo."""
    return [
        export_level(LEVELS[number], count, seed=seed + number)
        for number in sorted(LEVELS)
    ]


def load_text(path: str | Path) -> str:
    """Đọc dữ liệu đã ghi ra. Bỏ qua các dòng chú thích ở đầu file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.startswith("#")) + "\n"


def describe_file(path: str | Path) -> str:
    """Vài con số về một file dữ liệu, để biết nó chứa gì."""
    path = Path(path)

    if not path.exists():
        return f"{path} — chưa có"

    body = load_text(path)
    lines = [line for line in body.splitlines() if line.strip()]
    chars = sorted(set(body))

    return (
        f"{path.name}: {len(lines)} bài, {len(body)} ký tự, "
        f"{len(chars)} ký tự khác nhau, dài nhất {max((len(l) for l in lines), default=0)}"
    )
