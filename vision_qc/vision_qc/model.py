"""
model.py — Mạng tích chập, viết từ đầu. Không dùng pretrained, không torchvision.

Vì sao một logit chứ không phải hai?

    Hai lớp + softmax cho ra "OK 97% / NG 3%". Nhưng dây chuyền cần một câu
    trả lời khác: *"đặt ngưỡng ở đâu để tốn ít tiền nhất"*. Ngưỡng nằm ở đâu
    trong không gian softmax? Không rõ ràng.

    Một logit + sigmoid cho ra thẳng P(NG) — một con số chạy liên tục từ 0 tới
    1. `evaluate.py` quét con số đó để tìm ngưỡng rẻ nhất. Đó là toàn bộ lý do.

Kiến trúc: 4 khối, mỗi khối hai lớp tích chập rồi gộp cực đại. Cổ điển (kiểu
VGG thu nhỏ) và có lý do: nó chạy được trên CPU. Một ViT nhỏ cần nhiều dữ liệu
hơn và nhiều thời gian hơn để tới cùng độ chính xác — với vài nghìn ảnh thì CNN
thắng, và đó chính là tình huống trong xưởng.

    192x192x3
      -> 96x96x16     khối 1
      -> 48x48x32     khối 2
      -> 24x24x64     khối 3
      -> 12x12x128    khối 4
      -> gộp trung bình toàn cục -> 128 số -> 1 logit
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class ConvBlock(nn.Module):
    """hai lớp tích chập 3x3 rồi gộp cực đại.

    BatchNorm đứng sau mỗi lớp tích chập, trước ReLU. Không có nó thì mạng này
    không hội tụ được với tốc độ học 1e-3.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.pool(x)


class DefectNet(nn.Module):
    """Ảnh sản phẩm vào, một điểm "mức độ lỗi" ra."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()

        in_channels = 1 if self.cfg.grayscale else 3
        width = self.cfg.width

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            ConvBlock(width, width),
            ConvBlock(width, width * 2),
            ConvBlock(width * 2, width * 4),
            ConvBlock(width * 4, width * 8),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(width * 8, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo Kaiming cho tích chập — mặc định của PyTorch hợp với tanh
        hơn là ReLU, và chênh lệch đó làm chậm hội tụ vài chục epoch."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về logit thô, shape (B,). Dùng sigmoid để ra P(NG)."""
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)

    # -- tiện dụng ----------------------------------------------------

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> list[tuple[str, str, int]]:
        """(tên, cỡ đầu ra, số tham số) cho từng phần — để in ra xem."""
        size = self.cfg.image_size
        rows: list[tuple[str, str, int]] = []
        channels = 1 if self.cfg.grayscale else 3

        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        rows.append(("vào", f"{channels}x{size}x{size}", 0))
        rows.append(("stem", f"{self.cfg.width}x{size}x{size}", count(self.stem)))

        size //= 2
        for index, block in enumerate(self.blocks, start=1):
            out = self.cfg.width * (2 ** (index - 1))
            rows.append((f"khối {index}", f"{out}x{size}x{size}", count(block)))
            size //= 2

        rows.append(("gộp + đầu ra", "1", count(self.head)))
        return rows


# ---------------------------------------------------------------------
# Lưu / nạp
# ---------------------------------------------------------------------


def save_model(model: DefectNet, path, extra: dict | None = None) -> None:
    """Lưu cả trọng số lẫn cấu hình — nạp lại không cần đoán kiến trúc."""
    from pathlib import Path

    payload = {
        "config": model.cfg.__dict__,
        "state_dict": model.state_dict(),
        "num_params": model.num_params,
    }
    if extra:
        payload.update(extra)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model(path) -> tuple[DefectNet, dict]:
    """Nạp model đã lưu. Trả về (model, phần thông tin kèm theo)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)

    cfg = ModelConfig(**payload["config"])
    model = DefectNet(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    extra = {k: v for k, v in payload.items() if k not in {"config", "state_dict"}}
    return model, extra


def restore_camera(extra: dict, camera_cfg):
    """Khôi phục vùng cắt ROI đã dùng lúc huấn luyện.

    ROI phải ĐI CÙNG MODEL, không nằm rời trong file cấu hình.

    Nếu lúc dạy model nhìn vùng cắt (0,25 0,15 0,75 0,85) mà lúc chạy thật lại
    nhìn cả khung, model nhận một loại ảnh hoàn toàn khác — sản phẩm nhỏ hơn
    nhiều lần trong khung — và độ chính xác sụp mà không có lỗi nào báo. Đây
    đúng là loại lỗi mà việc gom tiền xử lý vào một hàm duy nhất
    (`imageops.preprocess`) sinh ra để chặn, nhưng ROI thì nằm ngoài hàm đó.
    """
    roi = extra.get("camera_roi")
    if roi is not None:
        camera_cfg.roi = tuple(roi)
    return camera_cfg
