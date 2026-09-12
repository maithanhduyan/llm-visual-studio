"""
deepseek_prod — Cấp 4: đưa một LLM ra chạy thật.

Cấp 1, 2, 3 trả lời câu hỏi "model hoạt động thế nào".
Cấp 4 trả lời câu hỏi khác hẳn: **làm sao để nó chạy được ở quy mô lớn?**

    Model không đổi. Cái thay đổi là mọi thứ bao quanh nó.

Cấp 4 KHÔNG viết lại model. Nó lấy nguyên model của Cấp 3
(`deepseek_lite`) rồi dựng hai nửa hệ thống quanh nó:

    A. HUẤN LUYỆN Ở QUY MÔ        model to hơn RAM của một máy
       ├── parallel.py            chia các tiến trình thành nhóm
       ├── tensor_parallel.py     cắt NGANG một tầng ra nhiều máy
       ├── pipeline.py            cắt DỌC model ra nhiều máy
       ├── expert_parallel.py     mỗi máy giữ vài chuyên gia
       ├── zero.py                chia trạng thái optimizer
       └── checkpoint.py          lưu model đã cắt thành nhiều mảnh

    B. PHỤC VỤ Ở QUY MÔ           nhiều người hỏi cùng lúc
       ├── paged_cache.py         đệm K/V chia thành khối, không phân mảnh
       ├── scheduler.py           gom nhiều câu hỏi vào một lô (continuous batching)
       ├── engine.py              máy phục vụ
       ├── quantize.py            nén model xuống 8 bit
       ├── speculative.py         đoán nhanh bằng model nhỏ rồi kiểm lại
       └── rope_scaling.py        kéo dài ngữ cảnh vượt lúc huấn luyện

Chạy:
    python -m deepseek_prod info
    python -m deepseek_prod bench
    torchrun ... python -m deepseek_prod train     (xem README)
"""

from .config import ParallelConfig, ProdConfig
from .parallel import Parallel, is_distributed

__version__ = "0.1.0"

__all__ = [
    "Parallel",
    "ParallelConfig",
    "ProdConfig",
    "is_distributed",
    "__version__",
]
