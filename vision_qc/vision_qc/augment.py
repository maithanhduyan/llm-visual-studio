"""
augment.py — Tăng cường ảnh, viết từ đầu bằng tensor của PyTorch.

Vì sao cần: một CNN huấn luyện từ đầu có ~300 nghìn tham số, còn trong xưởng
thì có vài trăm ảnh tốt và vài chục ảnh lỗi. Không tăng cường thì model học
thuộc lòng 300 ảnh đó trong 10 epoch rồi đạt 100% trên tập học và 60% trên
tập thi. Tăng cường chính là cách duy nhất bịa thêm dữ liệu từ dữ liệu đang có.

Vì sao viết tay: torchvision không có trong máy, và cài nó chỉ để dùng 6 phép
biến đổi là không đáng. `affine_grid` + `grid_sample` của PyTorch làm được toàn
bộ phần hình học, chạy theo lô, nhanh hơn hẳn vòng lặp PIL.

MỘT CẢNH BÁO QUAN TRỌNG
-----------------------
Tăng cường màu mạnh có thể **xoá mất chính thứ cần tìm**. Nếu lỗi là "sai màu
sơn" mà ta lại dạy model rằng màu nào cũng như nhau (đổi sáng/tương phản/gamma
ngẫu nhiên), thì model học được đúng một điều: bỏ qua màu. Với lỗi hình dạng
(xước, móp, thiếu chi tiết) thì ngược lại, tăng cường màu giúp ích.

Nên `AugmentConfig.color_strength` có ba nấc, và mặc định là nấc giữa.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    """Mức độ tăng cường. Đặt tất cả về 0 để tắt hoàn toàn."""

    # --- hình học: an toàn cho mọi loại lỗi ---
    flip_horizontal: bool = True
    flip_vertical: bool = False
    max_rotation_deg: float = 10.0
    max_translate: float = 0.06      # tỉ lệ cạnh ảnh
    scale_range: tuple[float, float] = (0.92, 1.08)

    # --- ánh sáng: "weak" | "medium" | "strong" ---
    color_strength: str = "medium"
    noise_sigma: float = 0.02

    # --- che ô: chỉ bật khi ảnh đủ nhiều ---
    erase_prob: float = 0.0
    erase_area: tuple[float, float] = (0.02, 0.08)

    @property
    def color_jitter(self) -> tuple[float, float, float]:
        """(độ lệch độ sáng, độ lệch tương phản, độ lệch gamma)."""
        return {
            "off": (0.0, 0.0, 0.0),
            "weak": (0.06, 0.06, 0.05),
            "medium": (0.15, 0.15, 0.12),
            "strong": (0.30, 0.30, 0.25),
        }.get(self.color_strength, (0.15, 0.15, 0.12))


# ---------------------------------------------------------------------
# Hình học
# ---------------------------------------------------------------------


def random_affine(
    x: torch.Tensor,
    max_rotation_deg: float,
    max_translate: float,
    scale_range: tuple[float, float],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Xoay, dịch, thu phóng ngẫu nhiên — cả lô một lần.

    `padding_mode="reflection"` chứ không phải "zeros": viền đen quanh ảnh là
    một tín hiệu giả mà model sẽ học ngay ("ảnh có viền đen = NG"), vì chỉ ảnh
    bị biến đổi mới có viền.
    """
    batch = x.shape[0]
    device = x.device

    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.empty(batch, device=device).uniform_(low, high, generator=generator)

    angle = uniform(-max_rotation_deg, max_rotation_deg) * (torch.pi / 180.0)
    scale = torch.empty(batch, device=device).uniform_(*scale_range, generator=generator)
    translate_x = uniform(-max_translate, max_translate) * 2.0
    translate_y = uniform(-max_translate, max_translate) * 2.0

    cos = torch.cos(angle) * scale
    sin = torch.sin(angle) * scale

    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = translate_x
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = translate_y

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


# ---------------------------------------------------------------------
# Ánh sáng
# ---------------------------------------------------------------------


def _to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] để áp gamma cho đúng."""
    return (x + 1.0) * 0.5


def _from_unit(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def random_photometric(
    x: torch.Tensor,
    brightness: float,
    contrast: float,
    gamma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Đổi độ sáng, tương phản, gamma. Cả lô dùng chung một mức (nhất quán
    giữa các ảnh trong lô giúp BatchNorm không bị nhiễu)."""
    if brightness == contrast == gamma == 0.0:
        return x

    def scalar(amount: float) -> float:
        return float(torch.empty(1).uniform_(-amount, amount, generator=generator).item())

    b = scalar(brightness)
    c = scalar(contrast)
    g = scalar(gamma)

    out = x
    if b:
        out = out + b
    if c:
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = (out - mean) * (1.0 + c) + mean
    if g:
        unit = _to_unit(out).clamp(0.0, 1.0)
        out = _from_unit(unit.pow(1.0 + g))

    return out


def add_noise(
    x: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Nhiễu Gauss. Có thật trong xưởng: cảm biến nóng lên là nhiễu tăng."""
    if sigma <= 0:
        return x
    noise = torch.randn(x.shape, device=x.device, generator=generator) * sigma
    return x + noise


def random_erase(
    x: torch.Tensor,
    prob: float,
    area: tuple[float, float],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Che một ô chữ nhật bằng giá trị trung bình.

    Cân nhắc kỹ trước khi bật: nếu ô bị che đúng vào chỗ vết lỗi, ta vừa dạy
    model rằng "vết lỗi có thể biến mất mà sản phẩm vẫn là NG". Với ảnh ít
    (dưới ~500 tấm) thì hại nhiều hơn lợi. Mặc định tắt.
    """
    if prob <= 0:
        return x

    batch, _, height, width = x.shape
    out = x.clone()

    for index in range(batch):
        if float(torch.rand(1, generator=generator).item()) > prob:
            continue

        fraction = float(
            torch.empty(1).uniform_(area[0], area[1], generator=generator).item()
        )
        aspect = float(torch.empty(1).uniform_(0.5, 2.0, generator=generator).item())

        erase_h = min(height, max(1, int((fraction * height * width / aspect) ** 0.5)))
        erase_w = min(width, max(1, int(erase_h * aspect)))

        top = int(torch.randint(0, max(1, height - erase_h), (1,), generator=generator).item())
        left = int(torch.randint(0, max(1, width - erase_w), (1,), generator=generator).item())

        out[index, :, top:top + erase_h, left:left + erase_w] = out[index].mean()

    return out


# ---------------------------------------------------------------------
# Gộp lại
# ---------------------------------------------------------------------


def augment(
    x: torch.Tensor,
    cfg: AugmentConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Tăng cường cả lô. `x` shape (B, C, H, W), giá trị trong khoảng -1..1.

    Gọi trong vòng lặp huấn luyện, KHÔNG gọi lúc đánh giá.
    """
    cfg = cfg or AugmentConfig()
    if cfg is None:
        return x

    out = x

    if cfg.flip_horizontal:
        mask = torch.rand(out.shape[0], generator=generator) < 0.5
        if mask.any():
            out = out.clone()
            out[mask] = torch.flip(out[mask], dims=[3])

    if cfg.flip_vertical:
        mask = torch.rand(out.shape[0], generator=generator) < 0.5
        if mask.any():
            out = out.clone()
            out[mask] = torch.flip(out[mask], dims=[2])

    if cfg.max_rotation_deg > 0 or cfg.max_translate > 0 or cfg.scale_range != (1.0, 1.0):
        out = random_affine(
            out, cfg.max_rotation_deg, cfg.max_translate, cfg.scale_range, generator
        )

    brightness, contrast, gamma = cfg.color_jitter
    out = random_photometric(out, brightness, contrast, gamma, generator)
    out = add_noise(out, cfg.noise_sigma, generator)
    out = random_erase(out, cfg.erase_prob, cfg.erase_area, generator)

    # Biến đổi có thể đẩy giá trị ra ngoài -1..1; kẹp lại cho khớp với ảnh
    # thật lúc chạy, nếu không model nhận hai loại đầu vào khác nhau.
    return out.clamp(-1.0, 1.0)
