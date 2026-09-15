"""
dataset.py — Đọc ảnh từ đĩa, chia tập, và cảnh báo về những cái bẫy.

Ba cái bẫy được xử lý ở đây, và cả ba đều làm điểm kiểm tra **đẹp giả tạo**:

1. **Rò rỉ giữa tập học và tập thi.** Chụp 5 khung hình của cùng một sản phẩm
   đang đứng yên, rồi chia ngẫu nhiên 4 tấm vào tập học và 1 tấm vào tập thi.
   Năm tấm đó giống nhau tới từng điểm ảnh. Model chỉ cần học thuộc lòng là
   được 100% — và sẽ sai thảm hại ngoài xưởng. Cách chữa: chia theo **sản
   phẩm** (`part_id`), không chia theo ảnh. Cả 5 tấm đi cùng một tập.

2. **Ảnh trùng.** Bấm nút hai lần cho cùng một sản phẩm. Xem `imageops.dhash`.

3. **Model học ánh sáng thay vì học lỗi.** Nếu ảnh OK được chụp buổi sáng và
   ảnh NG buổi chiều, model sẽ học "sáng = OK". Nhìn thì đạt 99%, nhưng nó
   chưa hề nhìn vào sản phẩm. `brightness_gap()` đo đúng chuyện này.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AugmentConfig, augment
from .config import CameraConfig, ModelConfig
from .imageops import crop_roi, dhash, hamming, preprocess

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

# Tên file do `capture.py` sinh ra:
#     OK_p00007_20260915_214530_03.jpg
#       |  |          |       |
#       |  |          |       +-- số thứ tự khung trong lần bấm đó
#       |  |          +---------- ngày giờ
#       |  +--------------------- mã sản phẩm  <-- khoá để chia tập
#       +------------------------ nhãn
NAME_PATTERN = re.compile(
    r"^(?P<label>[A-Za-z]+)_p(?P<part>\d+)_(?P<stamp>\d{8}_\d{6})_(?P<seq>\d+)"
)


# ---------------------------------------------------------------------
# Một mẫu dữ liệu
# ---------------------------------------------------------------------


@dataclass
class Sample:
    path: Path
    label: int          # 0 = OK, 1 = NG
    group: str          # mã sản phẩm — đơn vị chia tập
    brightness: float = 0.0


@dataclass
class DatasetStats:
    counts: dict[int, int] = field(default_factory=dict)
    groups: dict[int, int] = field(default_factory=dict)
    duplicates: list[tuple[str, str, int]] = field(default_factory=list)
    brightness: dict[int, float] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def brightness_gap(self) -> float:
        """Chênh lệch độ sáng trung bình giữa hai lớp, tính theo phần trăm.

        Lớn hơn ~5% là dấu hiệu nguy hiểm: model có thể đang đoán theo độ sáng
        chứ không nhìn sản phẩm. Cách chữa không phải là sửa model, mà là
        **chụp lại**: trộn lẫn hai lớp trong cùng một buổi, cùng một điều kiện
        sáng.
        """
        if 0 not in self.brightness or 1 not in self.brightness:
            return 0.0
        ok, ng = self.brightness[0], self.brightness[1]
        base = (ok + ng) / 2
        return abs(ok - ng) / base * 100 if base else 0.0


# ---------------------------------------------------------------------
# Quét thư mục
# ---------------------------------------------------------------------


def parse_filename(path: Path) -> tuple[int, str]:
    """Tên file -> (nhãn, mã sản phẩm). Không khớp thì suy ra từ thư mục cha."""
    match = NAME_PATTERN.match(path.stem)
    if match:
        label = 0 if match.group("label").upper() == "OK" else 1
        return label, f"p{match.group('part')}"

    # Không theo quy ước: lấy nhãn từ tên thư mục cha, và coi mỗi ảnh là một
    # sản phẩm riêng. Chia tập sẽ kém an toàn hơn, nhưng vẫn chạy được.
    label = 0 if path.parent.name.upper() == "OK" else 1
    return label, f"file:{path.stem}"


def scan(root: Path, compute_stats: bool = True) -> list[Sample]:
    """Quét `root/OK/` và `root/NG/`, trả về danh sách mẫu."""
    samples: list[Sample] = []

    for folder in ("OK", "NG"):
        directory = root / folder
        if not directory.is_dir():
            continue

        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            parsed_label, group = parse_filename(path)
            samples.append(Sample(path=path, label=parsed_label, group=group))

    if compute_stats:
        for sample in samples:
            image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
            sample.brightness = float(image.mean()) if image is not None else 0.0

    return samples


def statistics(samples: list[Sample], duplicates: bool = True) -> DatasetStats:
    """Đếm, tìm ảnh trùng, tính độ sáng trung bình mỗi lớp."""
    stats = DatasetStats()
    brightness: dict[int, list[float]] = {0: [], 1: []}

    seen: dict[int, Sample] = {}

    for sample in samples:
        stats.counts[sample.label] = stats.counts.get(sample.label, 0) + 1
        stats.groups[sample.label] = stats.groups.get(sample.label, 0) + 1
        brightness[sample.label].append(sample.brightness)

        if not duplicates:
            continue
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        digest = dhash(image)
        for other_digest, other in seen.items():
            distance = hamming(digest, other_digest)
            if distance <= 2:  # cho phép lệch vài bit vì nén JPEG
                stats.duplicates.append((sample.path.name, other.path.name, distance))
                break
        seen[digest] = sample

    for label, values in brightness.items():
        if values:
            stats.brightness[label] = float(np.mean(values))

    return stats


# ---------------------------------------------------------------------
# Chia tập — theo sản phẩm, không theo ảnh
# ---------------------------------------------------------------------


def split_by_group(
    samples: list[Sample],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    """Chia tập học / tập thi theo NHÓM sản phẩm.

    Mọi ảnh của cùng một sản phẩm đi về cùng một phía. Đây là điểm khác biệt
    quan trọng nhất so với `random_split` — và là điểm hay bị bỏ qua nhất.
    """
    if not samples:
        return [], []

    by_group: dict[str, list[Sample]] = {}
    for sample in samples:
        by_group.setdefault(sample.group, []).append(sample)

    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    target = int(round(len(groups) * val_fraction))
    target = max(1, min(target, len(groups) - 1)) if len(groups) > 1 else 0
    val_groups = set(groups[:target])

    train = [s for s in samples if s.group not in val_groups]
    val = [s for s in samples if s.group in val_groups]
    return train, val


# ---------------------------------------------------------------------
# Dataset cho PyTorch
# ---------------------------------------------------------------------


class ImageDataset(Dataset):
    """Ảnh sản phẩm -> tensor.

    Ảnh được nạp sẵn vào RAM ở cỡ `image_size` (uint8). Ảnh 2MP đọc từ đĩa và
    giải nén mất ~15 ms mỗi tấm; giải nén lại mỗi epoch thì 3000 ảnh × 40 epoch
    là 30 phút chỉ để đọc lại đúng những byte cũ.
    """

    def __init__(
        self,
        samples: list[Sample],
        model_cfg: ModelConfig | None = None,
        camera_cfg: CameraConfig | None = None,
        train: bool = False,
        augment_cfg: AugmentConfig | None = None,
        cache: bool = True,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.model_cfg = model_cfg or ModelConfig()
        self.camera_cfg = camera_cfg or CameraConfig()
        self.train = train
        self.augment_cfg = augment_cfg
        self.cache = cache

        self.labels = np.array([s.label for s in samples], dtype=np.float32)
        self.groups = [s.group for s in samples]
        self.images: np.ndarray | None = None
        self.generator = torch.Generator().manual_seed(seed)

        if cache and samples:
            self.images = self._load_all()

    # -- nạp ----------------------------------------------------------

    def _load_one(self, sample: Sample) -> np.ndarray:
        """Đọc -> cắt ROI -> chuẩn hoá về cùng cỡ. Dùng CHUNG hàm với lúc chạy
        thật, nên hai bên không thể lệch nhau."""
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(
                f"Không đọc được ảnh: {sample.path}\n"
                f"  File hỏng, hoặc ổ đĩa mất kết nối giữa chừng."
            )

        image = crop_roi(image, self.camera_cfg.roi)
        tensor = preprocess(image, self.model_cfg.image_size, self.model_cfg.grayscale)

        # (C, H, W) float32 -1..1 -> (H, W, C) uint8 0..255 để tiết kiệm RAM
        return np.clip((tensor.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)

    def _load_all(self) -> np.ndarray:
        rows = [self._load_one(sample) for sample in self.samples]
        return np.stack(rows)

    # -- giao diện Dataset -------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.images is not None:
            row = self.images[index]
        else:
            row = self._load_one(self.samples[index])

        tensor = torch.from_numpy(row.copy()).permute(2, 0, 1).float()
        tensor = tensor / 127.5 - 1.0  # uint8 -> -1..1

        if self.train and self.augment_cfg is not None:
            tensor = augment(tensor.unsqueeze(0), self.augment_cfg, self.generator)[0]

        return tensor, torch.tensor(self.labels[index])

    # -- tiện dụng ----------------------------------------------------

    @property
    def num_ok(self) -> int:
        return int((self.labels == 0).sum())

    @property
    def num_ng(self) -> int:
        return int((self.labels == 1).sum())

    @property
    def pos_weight(self) -> float:
        """Trọng số cho lớp NG. Mất cân bằng thì model đoán bừa "OK" là đạt
        95% ngay, mà đó lại đúng là kiểu sai tốn tiền nhất."""
        ng = max(self.num_ng, 1)
        return self.num_ok / ng

    def memory_mb(self) -> float:
        return self.images.nbytes / 1e6 if self.images is not None else 0.0
