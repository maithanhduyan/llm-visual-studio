"""
synth.py — Sinh ảnh sản phẩm giả, có lỗi và không lỗi, để thử cả đường ống.

Vì sao cần, khi đã có camera thật: **để thử được ngay hôm nay**. Một dây chuyền
thật cần vài trăm sản phẩm tốt và vài chục sản phẩm lỗi mới dạy được model —
đó là vài ngày. Trong lúc chờ, bộ sinh này cho phép chạy thử và đo được toàn bộ
đường ống: huấn luyện, đánh giá, ngưỡng, giao tiếp PLC, ngân sách thời gian.

ĐIỀU QUAN TRỌNG NHẤT Ở FILE NÀY
--------------------------------
Độ sáng được lấy ngẫu nhiên **độc lập với nhãn**. Nghĩa là ảnh OK và ảnh NG
có cùng phân bố độ sáng. Nếu không làm vậy, bộ dữ liệu giả sẽ có một lối tắt
("sáng = OK") và model sẽ đạt 99% mà chưa hề nhìn vào sản phẩm.

Có lối tắt đó thì tiện cho việc khoe số đẹp, nhưng nó dạy đúng một điều sai.
Xem `brightness_check` trong `checks.py` để biết cách phát hiện lối tắt ấy
trên dữ liệu THẬT.

Bộ sinh này KHÔNG thay thế được dữ liệu thật. Nó không có vết dầu, không có
bụi, không có phản chiếu kim loại thật, không có rung băng tải. Độ chính xác
trên ảnh giả không nói lên điều gì về độ chính xác trong xưởng.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFECT_KINDS = ("scratch", "dent", "stain", "chip", "spot")


@dataclass
class SynthConfig:
    """Tham số bộ sinh."""

    width: int = 1280
    height: int = 720

    # Độ rõ của lỗi, 0..1. 0,2 là lỗi thoáng qua rất khó thấy; 0,9 là hỏng
    # nặng nhìn là biết. Trộn nhiều mức mới ra được bài toán thật.
    severity: float = 0.5

    kinds: tuple[str, ...] = DEFECT_KINDS

    # Sản phẩm chiếm bao nhiêu phần cạnh ngắn của ảnh.
    part_scale: float = 0.55

    # Nhiễu cảm biến và độ nhòe chuyển động — hai thứ luôn có thật.
    noise: float = 4.0
    motion_blur_prob: float = 0.3

    # -- lưới an toàn cho nhãn --
    # Một ảnh NG phải khác ảnh OK cùng cảnh ít nhất ngần này điểm ảnh (mức xám
    # lệch hơn 5). Nếu không, đó là nhãn sai chứ không phải lỗi khó.
    defect_attempts: int = 8
    min_defect_pixels: int = 30


# ---------------------------------------------------------------------
# Các lớp của bức ảnh
# ---------------------------------------------------------------------


def _belt(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Băng tải: nền xám, có vệt dọc và hạt."""
    base = 70.0 + rng.uniform(-15, 25)

    image = np.full((height, width), base, dtype=np.float32)

    # Vệt dọc chạy dọc băng tải (băng tải hay có sọc)
    for _ in range(rng.integers(3, 9)):
        x = rng.integers(0, width)
        thickness = rng.integers(2, 14)
        brightness = rng.uniform(-18, 18)
        image[:, max(0, x - thickness):x + thickness] += brightness

    # Hạt lấm tấm
    grain = rng.normal(0, 6.0, size=(height // 4, width // 4)).astype(np.float32)
    image += cv2.resize(grain, (width, height), interpolation=cv2.INTER_LINEAR)

    return image


def _lighting(
    rng: np.random.Generator, width: int, height: int, centre: tuple[float, float]
) -> np.ndarray:
    """Đèn chiếu: sáng hơn ở một phía, tối dần ở rìa.

    Lấy ngẫu nhiên ĐỘC LẬP với nhãn — đây là chỗ quyết định bộ dữ liệu giả có
    dạy model một lối tắt hay không.
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    # Đèn lệch khỏi tâm một chút
    cx = centre[0] + rng.uniform(-0.25, 0.25) * width
    cy = centre[1] + rng.uniform(-0.25, 0.25) * height
    radius = np.hypot(xs - cx, ys - cy) / max(width, height)

    falloff = 1.0 - rng.uniform(0.25, 0.55) * radius
    tilt = 1.0 + rng.uniform(-0.10, 0.10) * (xs / width - 0.5)

    return (falloff * tilt).astype(np.float32)


def _product_mask(
    rng: np.random.Generator, width: int, height: int, cfg: SynthConfig
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float]:
    """Mặt nạ sản phẩm: một đĩa tròn, có lỗ ở giữa, lệch chỗ đôi chút."""
    scale = cfg.part_scale * rng.uniform(0.92, 1.08)
    radius = min(width, height) * scale / 2

    cx = width / 2 + rng.uniform(-0.06, 0.06) * width
    cy = height / 2 + rng.uniform(-0.06, 0.06) * height

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.hypot(xs - cx, ys - cy)

    mask = (distance <= radius).astype(np.float32)

    # Viền mềm 2 điểm ảnh, giống ảnh thật
    mask = np.clip((radius - distance) / 2.0 + 0.5, 0.0, 1.0).astype(np.float32)

    # Lỗ ở giữa
    hole_radius = radius * rng.uniform(0.16, 0.30)
    hole = np.clip((hole_radius - distance) / 2.0 + 0.5, 0.0, 1.0).astype(np.float32)
    mask = np.clip(mask - hole, 0.0, 1.0)

    return mask, distance, (cx, cy), radius


def _shade_product(
    rng: np.random.Generator, distance: np.ndarray, centre: tuple[float, float], radius: float
) -> np.ndarray:
    """Kim loại: sáng ở trên-trái, tối ở dưới-phải, cộng vân gia công."""
    ys, xs = np.mgrid[0:distance.shape[0], 0:distance.shape[1]].astype(np.float32)

    # Ánh sáng nghiêng
    normal_x = (xs - centre[0]) / radius
    normal_y = (ys - centre[1]) / radius
    shade = 0.55 + 0.45 * np.clip(-normal_x * 0.6 - normal_y * 0.8, -1.0, 1.0)

    # Vân tiện tròn đồng tâm
    rings = np.sin(distance / max(radius, 1.0) * rng.uniform(28, 60)) * 0.04
    shade = shade + rings

    # Vết bẩn nhẹ rải rác — quan trọng: đây là thứ khiến ảnh OK không hề
    # đồng nhất, nên model không thể học vẹt "cứ sạch trơn là OK".
    for _ in range(rng.integers(0, 4)):
        bx = rng.uniform(0, distance.shape[1])
        by = rng.uniform(0, distance.shape[0])
        br = rng.uniform(6, 30)
        blob = np.exp(-(((xs - bx) ** 2 + (ys - by) ** 2) / (2 * br**2)))
        shade = shade - blob * rng.uniform(0.02, 0.10)

    base = rng.uniform(120, 190)
    return (shade * base).astype(np.float32)


# ---------------------------------------------------------------------
# Lỗi
#
# TẤT CẢ kích thước lỗi đều tính theo `radius` (bán kính sản phẩm trong ảnh),
# KHÔNG tính theo điểm ảnh tuyệt đối.
#
# Đây không phải chuyện thẩm mỹ. Một vết xước rộng 2 điểm ảnh trên ảnh
# 1280x720, sau khi thu về 192x192, còn lại 0,5 điểm ảnh — nó biến mất. Bộ
# sinh vẽ ra một vết xước mà model không thể thấy, rồi ta kết luận "model dở".
# Kết luận đó sai, và sai một cách tốn kém: nó khiến ta đi sửa model trong khi
# vấn đề nằm ở độ phân giải.
#
# Ngoài xưởng, vết xước là một tính chất CỦA SẢN PHẨM: nó rộng bao nhiêu phần
# trăm đường kính. Nên ở đây cũng vậy.
#
# Đo được (sản phẩm bán kính 198 px trên ảnh 1280x720, model nhận 192x192,
# tức thu nhỏ 3,75 lần):
#
#     vết xước 2 px  -> 0,5 điểm ảnh ở đầu vào model   KHÔNG THỂ THẤY
#     vết xước 8 px  -> 2,1 điểm ảnh                   thấy được
#
# Tỉ lệ 0,015-0,05 lần bán kính cho ra 3-10 px, tức 0,8-2,7 điểm ảnh ở đầu
# vào model. Đó là mức "nhìn thấy được nhưng vẫn khó" — đúng thứ cần để thử.
# ---------------------------------------------------------------------


def _defect_scratch(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    severity: float,
    radius: float,
    background: np.ndarray | None = None,
) -> None:
    """Vết xước: một đường mảnh, sáng hoặc tối hơn nền.

    Điểm bắt đầu nằm BÊN TRONG sản phẩm, không phải ngẫu nhiên khắp ảnh.

    Bản đầu tiên lấy điểm bắt đầu ngẫu nhiên trên cả ảnh 1280x720 rồi kéo một
    đoạn dài 158-475 px. Phần lớn số lần, đoạn thẳng đó không chạm vào sản
    phẩm — sản phẩm chỉ chiếm một vòng tròn bán kính 198 px ở giữa. Kết quả:
    ảnh được dán nhãn NG nhưng giống hệt ảnh OK. Đo được: `max|Δ| = 0,000`.

    Đó là **nhãn sai**, và nó tệ hơn cả một vết xước mờ: model được dạy rằng
    những tấm ảnh này là lỗi, nên nó học cách bỏ qua chúng, và độ chính xác
    trên thực tế tụt mà không có gì báo động.
    """
    height, width = image.shape

    # Bắt đầu trong lòng sản phẩm
    angle_in = rng.uniform(0, 2 * np.pi)
    offset = radius * rng.uniform(0.0, 0.7)
    x0 = width / 2 + np.cos(angle_in) * offset
    y0 = height / 2 + np.sin(angle_in) * offset

    angle = rng.uniform(0, np.pi)
    length = rng.uniform(0.8, 2.4) * radius

    x1 = x0 + np.cos(angle) * length
    y1 = y0 + np.sin(angle) * length

    thickness = max(2, int(radius * rng.uniform(0.015, 0.05) * (0.6 + severity)))

    overlay = np.zeros((height, width), dtype=np.float32)
    cv2.line(overlay, (int(x0), int(y0)), (int(x1), int(y1)),
             color=1.0, thickness=thickness)

    overlay = cv2.GaussianBlur(overlay, (0, 0), max(0.8, radius * 0.004))
    overlay *= mask

    contrast = rng.choice([-1.0, 1.0]) * (35.0 + 95.0 * severity)
    image += overlay * contrast


def _defect_dent(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    severity: float,
    radius: float,
    background: np.ndarray | None = None,
) -> None:
    """Vết móp: một vùng tối hình bầu dục, có bóng đổ."""
    height, width = image.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    cx = width / 2 + rng.uniform(-0.5, 0.5) * radius
    cy = height / 2 + rng.uniform(-0.5, 0.5) * radius
    rx = radius * rng.uniform(0.12, 0.35) * (0.6 + severity)
    ry = rx * rng.uniform(0.5, 1.4)
    angle = rng.uniform(0, np.pi)

    dx, dy = xs - cx, ys - cy
    xr = dx * np.cos(angle) + dy * np.sin(angle)
    yr = -dx * np.sin(angle) + dy * np.cos(angle)

    blob = np.exp(-((xr / rx) ** 2 + (yr / ry) ** 2))
    image -= blob * mask * (35.0 + 75.0 * severity)


def _defect_stain(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    severity: float,
    radius: float,
    background: np.ndarray | None = None,
) -> None:
    """Vết bẩn: mảng tối, viền nham nhở."""
    height, width = image.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    cx = width / 2 + rng.uniform(-0.5, 0.5) * radius
    cy = height / 2 + rng.uniform(-0.5, 0.5) * radius
    blob_radius = radius * rng.uniform(0.15, 0.40) * (0.6 + severity)

    distance = np.hypot(xs - cx, ys - cy)
    wobble = 1.0 + 0.35 * np.sin(np.arctan2(ys - cy, xs - cx) * rng.uniform(3, 7))
    stain = np.clip((blob_radius * wobble - distance) / 3.0 + 0.5, 0.0, 1.0)

    image -= stain * mask * (40.0 + 80.0 * severity)


def _defect_chip(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    severity: float,
    radius: float,
    background: np.ndarray | None = None,
) -> None:
    """Sứt rìa: khoét một miếng khỏi mép sản phẩm — lỗi hình dạng, không phải
    lỗi màu. Model phải nhìn vào HÌNH DẠNG mới thấy được.

    Chỗ sứt phải lộ ra **băng tải** phía sau, không phải màu đen. Khoét thành
    màu đen thì model chỉ cần học "có mảng đen = NG" là xong, và nó sẽ không
    bao giờ nhìn thấy một vết sứt thật (vốn cùng màu với nền).
    """
    height, width = image.shape
    edge = mask - cv2.erode(mask, np.ones((5, 5), np.uint8))

    ys, xs = np.nonzero(edge > 0.5)
    if len(xs) == 0:
        return

    pick = rng.integers(0, len(xs))
    cx, cy = float(xs[pick]), float(ys[pick])

    size = radius * (0.12 + 0.30 * severity)
    angles = rng.integers(5, 12)
    points = []
    for index in range(angles):
        theta = 2 * np.pi * index / angles
        r = size * rng.uniform(0.5, 1.3)
        points.append([int(cx + np.cos(theta) * r), int(cy + np.sin(theta) * r)])

    overlay = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], 1.0)

    if background is None:
        image *= (1.0 - overlay)
    else:
        image *= (1.0 - overlay)
        image += background * overlay


def _defect_spot(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    severity: float,
    radius: float,
    background: np.ndarray | None = None,
) -> None:
    """Đốm nhỏ, sáng — kiểu bụi kim loại hoặc cháy xước."""
    height, width = image.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    for _ in range(rng.integers(1, 4)):
        cx = width / 2 + rng.uniform(-0.6, 0.6) * radius
        cy = height / 2 + rng.uniform(-0.6, 0.6) * radius
        blob_radius = radius * rng.uniform(0.03, 0.12) * (0.5 + severity)
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * blob_radius**2)))
        image += blob * mask * rng.choice([-1.0, 1.0]) * (45.0 + 85.0 * severity)


DEFECT_FUNCS = {
    "scratch": _defect_scratch,
    "dent": _defect_dent,
    "stain": _defect_stain,
    "chip": _defect_chip,
    "spot": _defect_spot,
}


def _apply_defect(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    cfg: SynthConfig,
    radius: float,
    belt: np.ndarray,
) -> tuple[str, np.ndarray]:
    """Vẽ một lỗi lên ảnh, và KIỂM TRA rằng nó thật sự thay đổi ảnh.

    Bất biến của bộ sinh: **một ảnh dán nhãn NG phải khác ảnh OK cùng cảnh.**
    Không có bất biến này thì bộ sinh âm thầm tạo ra nhãn sai, và nhãn sai thì
    không cách nào sửa được ở phía model — model chỉ học được cách bỏ qua
    chúng, hoặc tệ hơn là học nhầm.

    Nếu lần vẽ đầu không để lại dấu vết gì (lỗi rơi ra ngoài sản phẩm, hoặc bị
    `mask` che mất), thử lại bằng loại lỗi khác. Đây là lưới an toàn, không
    phải cách chính để vẽ đúng.
    """
    best: tuple[str, np.ndarray, int] | None = None

    for _ in range(cfg.defect_attempts):
        kind = str(rng.choice(list(cfg.kinds)))
        severity = float(np.clip(rng.normal(cfg.severity, 0.18), 0.05, 1.0))

        trial = image.copy()
        DEFECT_FUNCS[kind](trial, mask, rng, severity, radius, belt)

        changed = int((np.abs(trial - image) > 5.0).sum())
        if changed >= cfg.min_defect_pixels:
            return kind, trial

        if best is None or changed > best[2]:
            best = (kind, trial, changed)

    # Không lần nào đủ rõ. Dùng lần rõ nhất, nhưng nói to lên: đây là dấu hiệu
    # cấu hình sai (sản phẩm quá nhỏ, hoặc `severity` quá thấp).
    assert best is not None
    kind, trial, changed = best
    if changed == 0:
        raise RuntimeError(
            f"Không vẽ được lỗi nào lên sản phẩm sau {cfg.defect_attempts} lần thử.\n"
            f"  Sản phẩm bán kính {radius:.0f} px, `severity` = {cfg.severity}.\n"
            f"  Tăng `severity`, hoặc tăng `part_scale`, hoặc giảm `image_size` "
            f"của model để lỗi không bị thu nhỏ mất."
        )
    return kind, trial


# ---------------------------------------------------------------------
# Ghép lại
# ---------------------------------------------------------------------


def render(
    ok: bool,
    rng: np.random.Generator,
    cfg: SynthConfig | None = None,
) -> tuple[np.ndarray, str]:
    """Sinh một ảnh. Trả về (ảnh BGR uint8, tên loại lỗi hoặc "OK")."""
    cfg = cfg or SynthConfig()
    width, height = cfg.width, cfg.height

    mask, distance, centre, radius = _product_mask(rng, width, height, cfg)

    belt = _belt(rng, width, height)
    shade = _shade_product(rng, distance, centre, radius)

    image = belt * (1.0 - mask) + shade * mask
    image *= _lighting(rng, width, height, centre)

    kind = "OK"
    if not ok:
        kind, image = _apply_defect(image, mask, rng, cfg, radius, belt)

    # Nhiễu cảm biến — luôn có, và giống nhau ở cả hai lớp
    image += rng.normal(0, cfg.noise, size=image.shape).astype(np.float32)

    # Nhòe chuyển động: băng tải đang chạy thì ảnh không bao giờ nét tuyệt đối
    if rng.random() < cfg.motion_blur_prob:
        length = int(rng.integers(3, 11))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        angle = rng.uniform(0, 180)
        matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (length, length))
        total = kernel.sum()
        if total > 0:
            kernel /= total
            image = cv2.filter2D(image, -1, kernel)

    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), kind


def generate(
    directory,
    count_ok: int,
    count_ng: int,
    seed: int = 0,
    cfg: SynthConfig | None = None,
    prefix_ok: str = "OK",
    prefix_ng: str = "NG",
    quiet: bool = False,
) -> dict[str, int]:
    """Ghi ảnh giả ra `directory/OK/` và `directory/NG/`.

    Tên file theo đúng quy ước `capture.py`: mỗi ảnh một mã sản phẩm riêng, vì
    ảnh giả không có nhiều khung hình của cùng một sản phẩm.
    """
    from pathlib import Path

    from .paths import CLASSES

    cfg = cfg or SynthConfig()
    root = Path(directory)
    rng = np.random.default_rng(seed)

    written: dict[str, int] = {name: 0 for name in CLASSES}
    kinds: dict[str, int] = {}

    for label, count, prefix in ((True, count_ok, prefix_ok), (False, count_ng, prefix_ng)):
        folder = root / ("OK" if label else "NG")
        folder.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            image, kind = render(label, rng, cfg)
            kinds[kind] = kinds.get(kind, 0) + 1

            name = f"{prefix}_p{index:05d}_20260101_000000_00.jpg"
            cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written["OK" if label else "NG"] += 1

    if not quiet:
        print(f"  đã ghi vào {root}")
        for name in CLASSES:
            print(f"    {name}: {written[name]}")
        if len(kinds) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()) if k != "OK")
            print(f"    loại lỗi: {detail}")

    return written
