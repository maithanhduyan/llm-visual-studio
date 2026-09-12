"""
quantize.py — Nén model xuống 8 bit.

Mỗi thông số đang chiếm 4 byte (float32). Nhưng để chọn chữ tiếp theo thì
4 byte là thừa — 1 byte với 256 mức cũng đủ. Nén lại thì model nhỏ đi 4 lần,
nhét vừa card đồ họa nhỏ hơn, và đọc từ bộ nhớ nhanh hơn.

    float32   ████████████████████████████████  4 byte / thông số
    int8      ████████                          1 byte / thông số

Cái giá: mỗi lần nhân phải giải nén rồi lại nén, nên kết quả không còn
giống hệt bản gốc — chỉ gần đúng.

Ở đây dùng "dynamic quantization" của PyTorch: trọng số nén sẵn thành int8,
còn con số scale thì tính ngay lúc chạy. Chỉ áp cho các lớp Linear — đó là
chỗ có nhiều thông số nhất.

FP8 (định dạng 8 bit của card H100 trở lên) không chạy được trên CPU, nên
file này chỉ làm int8. Xem docs/han-che.md.
"""

from __future__ import annotations

import io

import torch
import torch.nn as nn


def model_bytes(model: nn.Module) -> int:
    """Model chiếm bao nhiêu byte khi lưu xuống đĩa."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def quantize_int8(model: nn.Module) -> nn.Module:
    """Nén mọi lớp Linear xuống int8. Phần còn lại giữ nguyên."""
    model.eval()

    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def quantize_int8_linear_only(model: nn.Module) -> nn.Module:
    """Như trên, nhưng bỏ qua LM Head.

    LM Head quyết định điểm của từng chữ. Nén nó xuống 8 bit thì chữ sinh ra
    dễ sai hơn hẳn, mà nó chỉ chiếm một phần nhỏ thông số. Nên giữ nó ở
    float32 — đây cũng là cách các hệ thật làm.
    """
    model.eval()

    keep = {"lm_head"}
    to_quantize = nn.Sequential()

    for name, module in model.named_children():
        if name in keep:
            continue
        to_quantize.add_module(name, module)

    quantized = torch.ao.quantization.quantize_dynamic(
        to_quantize, {nn.Linear}, dtype=torch.qint8
    )

    for name, module in quantized.named_children():
        setattr(model, name, module)

    return model
