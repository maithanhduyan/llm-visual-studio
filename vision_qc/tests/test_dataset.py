"""
test_dataset.py — Chia tập, tiền xử lý, và những cái bẫy làm điểm thi đẹp giả tạo.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision_qc.config import CameraConfig, ModelConfig
from vision_qc.dataset import (
    ImageDataset,
    Sample,
    parse_filename,
    scan,
    split_by_group,
    statistics,
)
from vision_qc.imageops import crop_roi, dhash, hamming, preprocess

# ---------------------------------------------------------------------
# Tên file
# ---------------------------------------------------------------------


def test_doc_ten_file_theo_quy_uoc():
    from pathlib import Path

    label, group = parse_filename(Path("OK_p00007_20260915_214530_03.jpg"))
    assert label == 0
    assert group == "p00007"

    label, group = parse_filename(Path("NG_p00042_20260915_214531_00.jpg"))
    assert label == 1
    assert group == "p00042"


def test_ten_file_la_thi_suy_ra_tu_thu_muc():
    from pathlib import Path

    label, group = parse_filename(Path("data/raw/NG/anh_gi_do.jpg"))
    assert label == 1
    assert group == "file:anh_gi_do"


# ---------------------------------------------------------------------
# Bất biến quan trọng nhất: không rò rỉ sản phẩm giữa hai tập
# ---------------------------------------------------------------------


def _make_samples() -> list[Sample]:
    from pathlib import Path

    samples = []
    for part in range(20):
        # 5 khung hình cho mỗi sản phẩm — giống hệt nhau, như khi chụp burst
        for sequence in range(5):
            label = part % 2
            name = f"{'NG' if label else 'OK'}_p{part:05d}_20260101_000000_{sequence:02d}.jpg"
            samples.append(Sample(path=Path(name), label=label, group=f"p{part:05d}"))
    return samples


def test_chia_tap_khong_ro_ri_san_pham():
    """Mọi ảnh của cùng một sản phẩm phải đi về CÙNG một tập.

    Đây là bất biến quan trọng nhất của cả file. Nếu 4 khung hình của một sản
    phẩm vào tập học và khung thứ 5 vào tập thi, model chỉ cần học thuộc lòng
    là được 100% — và con số đó hoàn toàn vô nghĩa.
    """
    samples = _make_samples()
    train, val = split_by_group(samples, val_fraction=0.25, seed=0)

    train_groups = {s.group for s in train}
    val_groups = {s.group for s in val}

    assert not (train_groups & val_groups), (
        f"rò rỉ! các sản phẩm nằm ở cả hai tập: {train_groups & val_groups}"
    )
    assert len(train) + len(val) == len(samples)


def test_chia_tap_on_dinh_voi_cung_seed():
    samples = _make_samples()
    a_train, a_val = split_by_group(samples, 0.25, seed=7)
    b_train, b_val = split_by_group(samples, 0.25, seed=7)

    assert [s.path for s in a_train] == [s.path for s in b_train]
    assert [s.path for s in a_val] == [s.path for s in b_val]


def test_chia_tap_can_bang_hai_lop():
    """Sản phẩm chẵn là OK, lẻ là NG — cả hai tập phải có đủ hai lớp."""
    samples = _make_samples()
    train, val = split_by_group(samples, 0.25, seed=0)

    assert {s.label for s in train} == {0, 1}
    assert {s.label for s in val} == {0, 1}


def test_chia_tap_khi_chi_co_mot_san_pham():
    from pathlib import Path

    samples = [Sample(path=Path("a.jpg"), label=0, group="p1")]
    train, val = split_by_group(samples, 0.2, seed=0)

    assert len(train) == 1 and len(val) == 0


# ---------------------------------------------------------------------
# Tiền xử lý
# ---------------------------------------------------------------------


def test_tien_xu_ly_ra_dung_hinh_dang():
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = preprocess(image, image_size=64)

    assert out.shape == (3, 64, 64)
    assert out.dtype == np.float32
    assert -1.01 <= out.min() and out.max() <= 1.01


def test_tien_xu_ly_anh_xam():
    image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    out = preprocess(image, image_size=32, grayscale=True)

    assert out.shape == (1, 32, 32)


def test_cat_vuong_KHONG_lam_meo_hinh_hoc():
    """Ép 16:9 thẳng về vuông nén hai trục lệch nhau 1,78 lần.

    Đo trên hình tròn: ép vuông cho 40x71 (méo 43,7%), cắt vuông cho 71x71
    (méo 0%). Bài test này giữ cho lỗi đó không quay lại.
    """
    canvas = np.zeros((720, 1280), dtype=np.uint8)
    cv2.circle(canvas, (640, 360), 200, 255, -1)

    fixed = preprocess(canvas, image_size=128)[0]
    mask = fixed > 0.5
    ys, xs = np.nonzero(mask)
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1

    ratio = width / height
    assert abs(ratio - 1.0) < 0.05, f"hình tròn thành {width}x{height}, tỉ lệ {ratio:.2f}"

    # Và cách cũ thì phải méo — nếu không, bài test này vô nghĩa
    squashed = preprocess(canvas, image_size=128, fit="squash")[0]
    mask = squashed > 0.5
    ys, xs = np.nonzero(mask)
    old_ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    assert old_ratio < 0.7, "cách ép vuông lẽ ra phải méo rõ rệt"


def test_cat_roi():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, 100:] = 255

    right = crop_roi(image, (0.5, 0.0, 1.0, 1.0))
    assert right.shape == (100, 100, 3)
    assert right.mean() > 250

    assert crop_roi(image, None) is image


# ---------------------------------------------------------------------
# Băm ảnh
# ---------------------------------------------------------------------


def test_bam_anh_giong_nhau_thi_giong_nhau():
    image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    assert dhash(image) == dhash(image.copy())
    assert hamming(dhash(image), dhash(image.copy())) == 0


def test_bam_phan_biet_duoc_chieu_doc():
    """Sản phẩm sáng trên nền tối và sản phẩm tối trên nền sáng KHÁC nhau.

    Băm khác biệt một chiều chỉ so với điểm bên phải, nên mọi ảnh chỉ biến
    thiên theo chiều dọc — mỗi hàng phẳng — đều cho ra băm bằng 0. Hai ảnh
    dưới đây sẽ không phân biệt được. Đó là lý do `dhash` ở đây băm hai chiều.
    """
    sang_tren = np.zeros((200, 200), dtype=np.uint8)
    sang_tren[:100, :] = 255     # nửa trên sáng

    toi_tren = np.zeros((200, 200), dtype=np.uint8)
    toi_tren[100:, :] = 255      # nửa dưới sáng

    assert hamming(dhash(sang_tren), dhash(toi_tren)) > 5


def test_bam_khong_phu_thuoc_chieu_ngang():
    """Đối xứng: trái sáng và phải sáng cũng phải khác nhau."""
    trai_sang = np.zeros((200, 200), dtype=np.uint8)
    trai_sang[:, :100] = 255

    phai_sang = np.zeros((200, 200), dtype=np.uint8)
    phai_sang[:, 100:] = 255

    assert hamming(dhash(trai_sang), dhash(phai_sang)) > 5


# ---------------------------------------------------------------------
# Đọc từ đĩa
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_dataset(tmp_path):
    """Mười sản phẩm, mỗi sản phẩm hai khung hình."""
    for part in range(10):
        label = "NG" if part % 2 else "OK"
        folder = tmp_path / label
        folder.mkdir(exist_ok=True)

        for sequence in range(2):
            image = np.full((64, 64, 3), 40 + part * 15, dtype=np.uint8)
            if label == "NG":
                image[30:34, 30:34] = 250
            cv2.imwrite(
                str(folder / f"{label}_p{part:05d}_20260101_000000_{sequence:02d}.jpg"),
                image,
            )
    return tmp_path


def test_quet_thu_muc(tiny_dataset):
    samples = scan(tiny_dataset, compute_stats=True)

    assert len(samples) == 20
    assert sum(1 for s in samples if s.label == 0) == 10
    assert sum(1 for s in samples if s.label == 1) == 10
    assert len({s.group for s in samples}) == 10
    assert all(s.brightness > 0 for s in samples)


def test_thong_ke_va_lech_sang(tiny_dataset):
    samples = scan(tiny_dataset, compute_stats=True)
    stats = statistics(samples, duplicates=True)

    assert stats.total == 20
    assert stats.counts[0] == 10
    assert stats.counts[1] == 10

    # Ảnh trong bài test này lệch sáng theo nhãn một cách cố ý -> phải bắt được
    assert stats.brightness_gap() > 5.0


def test_dataset_tra_ve_tensor(tiny_dataset):
    samples = scan(tiny_dataset, compute_stats=False)
    model_cfg = ModelConfig(image_size=32, width=4)

    dataset = ImageDataset(samples, model_cfg, CameraConfig(roi=None), train=False)
    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype.__str__().startswith("torch.float32")
    assert label.item() in (0.0, 1.0)
    assert dataset.memory_mb() > 0


def test_pos_weight_cho_du_lieu_mat_can_bang(tiny_dataset):
    samples = scan(tiny_dataset, compute_stats=False)
    dataset = ImageDataset(samples, ModelConfig(image_size=32), CameraConfig())

    assert dataset.num_ok == 10
    assert dataset.num_ng == 10
    assert dataset.pos_weight == pytest.approx(1.0)
