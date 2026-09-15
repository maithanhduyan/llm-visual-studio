"""
test_capture.py — Phần thu ảnh, kiểm tra mà không cần camera.

`capture.py` có một vòng lặp giao diện không test được nếu không có người ngồi
trước màn hình. Nhưng phần thật sự quan trọng thì test được: tên file, chống
trùng, hoàn tác, và — quan trọng nhất — **tên file do `capture.py` ghi ra phải
đọc lại được bằng `dataset.py`**. Hai bên lệch nhau thì cả dự án đứt mà không
có lỗi nào báo.
"""

from __future__ import annotations

import numpy as np

from vision_qc.capture import CaptureSession, deduplicate, save_frame
from vision_qc.dataset import parse_filename, scan
from vision_qc.imageops import dhash


def _image(value: int = 100, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------
# Tên file — hợp đồng giữa capture.py và dataset.py
# ---------------------------------------------------------------------


def test_ten_file_doc_lai_duoc(tmp_path):
    """Vòng tròn: ghi ra rồi đọc lại, nhãn và mã sản phẩm phải khớp."""
    path = save_frame(_image(), "OK", part_id=7, sequence=0, raw_dir=tmp_path)
    label, group = parse_filename(path)

    assert label == 0
    assert group == "p00007"
    assert path.parent.name == "OK"


def test_ten_file_lop_ng(tmp_path):
    path = save_frame(_image(), "NG", part_id=42, sequence=3, raw_dir=tmp_path)
    label, group = parse_filename(path)

    assert label == 1
    assert group == "p00042"


def test_nhieu_khung_cung_san_pham(tmp_path):
    """Chụp burst: nhiều file, cùng một mã sản phẩm."""
    paths = [
        save_frame(_image(100 + i), "OK", part_id=3, sequence=i, raw_dir=tmp_path)
        for i in range(4)
    ]

    groups = {parse_filename(p)[1] for p in paths}
    assert groups == {"p00003"}
    assert len(set(paths)) == 4


def test_quet_lai_toan_bo(tmp_path):
    """Cả thư mục ghi ra phải quét lại được, đúng số lượng và nhãn."""
    for part in range(6):
        label = "NG" if part % 3 == 0 else "OK"
        save_frame(_image(50 + part * 10), label, part, 0, tmp_path)

    samples = scan(tmp_path, compute_stats=False)

    assert len(samples) == 6
    assert len({s.group for s in samples}) == 6
    assert sum(1 for s in samples if s.label == 1) == 2


# ---------------------------------------------------------------------
# Chống trùng
# ---------------------------------------------------------------------


def test_anh_giong_het_thi_bi_coi_la_trung():
    session = CaptureSession()
    image = _image(120)

    session.remember("OK", dhash(image), "OK_p00001_x_00.jpg")
    assert session.find_duplicate(dhash(image.copy()), "OK") is not None


def test_anh_khac_hang_thi_khong_bi_coi_la_trung():
    """Chống trùng không được quá tay — ảnh khác nhau phải được giữ."""
    session = CaptureSession()

    session.remember("OK", dhash(_image(60)), "OK_p00001_x_00.jpg")
    assert session.find_duplicate(dhash(_image(200)), "OK") is None


def test_trung_chi_tinh_trong_cung_mot_lop():
    """Ảnh OK giống ảnh NG vẫn phải được giữ — đó là mẫu khó, không phải lỗi."""
    session = CaptureSession()
    image = _image(120)

    session.remember("OK", dhash(image), "OK_p00001_x_00.jpg")
    assert session.find_duplicate(dhash(image.copy()), "NG") is None


def test_chong_trung_khong_nham_chieu_doc_voi_chieu_ngang():
    """Sản phẩm sáng trên nền tối và sản phẩm tối trên nền sáng khác nhau."""
    session = CaptureSession()

    sang_tren = np.zeros((200, 200), dtype=np.uint8)
    sang_tren[:100, :] = 255

    session.remember("NG", dhash(sang_tren), "NG_p00001_x_00.jpg")

    toi_tren = np.zeros((200, 200), dtype=np.uint8)
    toi_tren[100:, :] = 255

    assert session.find_duplicate(dhash(toi_tren), "NG") is None


# ---------------------------------------------------------------------
# Đếm và hoàn tác
# ---------------------------------------------------------------------


def test_dem_dung_theo_lop(tmp_path):
    session = CaptureSession()

    assert session.total() == 0

    for index in range(3):
        session.remember("OK", index, f"OK_p{index:05d}_x_00.jpg")
    session.remember("NG", 99, "NG_p00004_x_00.jpg")

    assert session.counts["OK"] == 3
    assert session.counts["NG"] == 1
    assert session.total() == 4


def test_hoan_tac_xoa_ca_lan_chup(tmp_path):
    """`u` phải xoá HẾT ảnh của lần bấm đó, không chỉ ảnh cuối."""
    session = CaptureSession()

    saved = []
    for sequence in range(3):
        path = save_frame(_image(100 + sequence), "OK", part_id=1,
                          sequence=sequence, raw_dir=tmp_path)
        session.remember("OK", dhash(_image(100 + sequence)), path.name)
        session.recent.append(path)
        saved.append(path)

    assert all(p.exists() for p in saved)
    removed = session.undo()

    assert removed == 3
    assert not any(p.exists() for p in saved)
    assert session.counts["OK"] == 0


def test_hoan_tac_khong_dung_toi_lan_chup_khac(tmp_path):
    session = CaptureSession()

    first = save_frame(_image(80), "OK", part_id=1, sequence=0, raw_dir=tmp_path)
    session.remember("OK", dhash(_image(80)), first.name)
    session.recent.append(first)

    second = save_frame(_image(180), "NG", part_id=2, sequence=0, raw_dir=tmp_path)
    session.remember("NG", dhash(_image(180)), second.name)
    session.recent.append(second)

    session.undo()

    assert first.exists(), "hoàn tác xoá nhầm lần chụp trước"
    assert not second.exists()
    assert session.counts["OK"] == 1
    assert session.counts["NG"] == 0


def test_hoan_tac_khi_chua_chup_gi():
    assert CaptureSession().undo() == 0


def test_ma_san_pham_tang_dan():
    session = CaptureSession()
    assert [session.next_part() for _ in range(3)] == [1, 2, 3]


# ---------------------------------------------------------------------
# Dọn ảnh trùng trên đĩa
# ---------------------------------------------------------------------


def test_don_anh_trung_tren_dia(tmp_path):
    ok_dir = tmp_path / "OK"
    ok_dir.mkdir()

    for index in range(5):
        save_frame(_image(130), "OK", part_id=index, sequence=0, raw_dir=tmp_path)

    assert len(list(ok_dir.glob("*.jpg"))) == 5

    removed, kept = deduplicate(tmp_path, tolerance=2)

    assert removed == 4, "4 ảnh giống hệt phải bị xoá, giữ lại 1"
    assert kept == 1
    assert len(list(ok_dir.glob("*.jpg"))) == 1


def test_don_anh_trung_giu_anh_khac_nhau(tmp_path):
    ok_dir = tmp_path / "OK"
    ok_dir.mkdir()

    for index, value in enumerate([40, 90, 140, 190, 240]):
        save_frame(_image(value), "OK", part_id=index, sequence=0, raw_dir=tmp_path)

    removed, kept = deduplicate(tmp_path, tolerance=2)

    assert removed == 0
    assert kept == 5
