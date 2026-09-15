"""
test_synth.py — Bộ sinh ảnh giả, và hai lỗi nhãn nó từng có.

Bộ sinh dữ liệu là chỗ nguy hiểm nhất của cả dự án: nó sai thì mọi con số phía
sau đều vô nghĩa, mà không có gì báo động. Hai bài test dưới đây khoá đúng hai
lỗi đã thật sự xảy ra.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_qc.imageops import preprocess
from vision_qc.synth import DEFECT_FUNCS, SynthConfig, generate, render


def _pair(kind: str | None, seed: int = 4242, **kwargs):
    """Một cặp ảnh OK / NG của CÙNG một cảnh. Khác biệt chỉ do lỗi."""
    base = dict(width=640, height=480, severity=0.6, motion_blur_prob=0.0, noise=0.0)
    base.update(kwargs)

    cfg_ok = SynthConfig(kinds=(kind,) if kind else DEFECT_FUNCS.keys(), **base)
    ok_img, _ = render(True, np.random.default_rng(seed), cfg_ok)

    ng_img, got = render(False, np.random.default_rng(seed), cfg_ok)
    return ok_img, ng_img, got


# ---------------------------------------------------------------------
# Bất biến 1: ảnh NG phải THẬT SỰ khác ảnh OK
# ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(DEFECT_FUNCS))
def test_moi_loai_loi_deu_de_lai_dau_vet(kind):
    """Lỗi phải được vẽ RA THẬT, không phải chỉ được đặt tên.

    Lỗi đã từng xảy ra: vết xước lấy điểm bắt đầu ngẫu nhiên trên cả ảnh, nên
    phần lớn số lần nó không chạm vào sản phẩm (sản phẩm chỉ chiếm một vòng
    tròn ở giữa). Ảnh được dán nhãn NG nhưng giống hệt ảnh OK — đo được
    `max|Δ| = 0,000`. Đó là nhãn sai, và model chỉ có thể học cách bỏ qua nó.
    """
    ok_img, ng_img, got = _pair(kind)

    delta = np.abs(ok_img.astype(np.int16) - ng_img.astype(np.int16))
    changed = int((delta > 5).sum())

    assert changed > 100, (
        f"lỗi {kind} chỉ đổi {changed} điểm ảnh — coi như không vẽ ra gì. "
        f"Ảnh NG này giống ảnh OK, tức là nhãn sai."
    )


def test_khong_bao_gio_tra_ve_anh_sai_nhan():
    """Sinh 50 ảnh NG, không ảnh nào được phép trùng ảnh OK cùng cảnh."""
    worst = 10**9

    for seed in range(50):
        ok_img, ng_img, _ = _pair(None, seed=seed, severity=0.25)
        delta = np.abs(ok_img.astype(np.int16) - ng_img.astype(np.int16))
        worst = min(worst, int((delta > 5).sum()))

    assert worst >= 30, f"có ảnh NG chỉ lệch {worst} điểm ảnh so với ảnh OK"


# ---------------------------------------------------------------------
# Bất biến 2: lỗi phải sống sót qua tiền xử lý
# ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(DEFECT_FUNCS))
def test_loi_con_thay_duoc_sau_khi_thu_nho(kind):
    """Lỗi phải còn nhìn thấy ở 192x192, không chỉ ở 1280x720.

    Lỗi đã từng xảy ra: kích thước lỗi tính bằng điểm ảnh TUYỆT ĐỐI, nên vết
    xước rộng 1-4 px trên ảnh 1280x720 bị thu nhỏ 3,75 lần còn 0,3-1,1 px ở
    đầu vào model — dưới ngưỡng nhiễu. Bộ sinh vẽ ra một vết xước mà model
    không thể thấy, rồi ta kết luận "model dở".
    """
    ok_img, ng_img, _ = _pair(kind, width=1280, height=720, severity=0.6)

    a = preprocess(ok_img, 192)
    b = preprocess(ng_img, 192)
    delta = np.abs(a - b)

    strong = int((delta > 0.1).sum())
    assert strong > 20, (
        f"lỗi {kind} chỉ còn {strong} điểm ảnh đủ tương phản ở đầu vào model. "
        f"Kích thước lỗi phải tính theo bán kính sản phẩm, không theo điểm ảnh."
    )


# ---------------------------------------------------------------------
# Bất biến 3: không có lối tắt độ sáng
# ---------------------------------------------------------------------


def test_do_sang_khong_tuong_quan_voi_nhan():
    """Ảnh OK và ảnh NG phải cùng phân bố độ sáng.

    Nếu không, bộ dữ liệu giả có một lối tắt ("sáng = tốt") và model sẽ đạt
    điểm cao mà chưa hề nhìn vào sản phẩm.
    """
    import cv2

    cfg = SynthConfig(width=320, height=240, severity=0.5)
    rng = np.random.default_rng(7)

    ok_means, ng_means = [], []
    for _ in range(120):
        image, _ = render(True, rng, cfg)
        ok_means.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        image, _ = render(False, rng, cfg)
        ng_means.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())

    ok, ng = float(np.mean(ok_means)), float(np.mean(ng_means))
    gap = abs(ok - ng) / ((ok + ng) / 2) * 100

    assert gap < 4.0, f"độ sáng lệch {gap:.1f}% — có lối tắt độ sáng"


# ---------------------------------------------------------------------
# Bộ sinh chạy được
# ---------------------------------------------------------------------


def test_sinh_ra_file(tmp_path):
    written = generate(tmp_path, count_ok=4, count_ng=4, seed=0,
                       cfg=SynthConfig(width=160, height=120), quiet=True)

    assert written["OK"] == 4
    assert written["NG"] == 4
    assert len(list((tmp_path / "OK").glob("*.jpg"))) == 4
    assert len(list((tmp_path / "NG").glob("*.jpg"))) == 4


def test_severity_cang_cao_loi_cang_ro():
    """Độ rõ phải thật sự điều khiển được độ khó của bài toán."""
    weak = []
    strong = []

    for seed in range(12):
        ok_img, ng_img, _ = _pair("dent", seed=seed, severity=0.15)
        delta = np.abs(ok_img.astype(np.int16) - ng_img.astype(np.int16))
        weak.append(float(delta.mean()))

        ok_img, ng_img, _ = _pair("dent", seed=seed, severity=0.95)
        delta = np.abs(ok_img.astype(np.int16) - ng_img.astype(np.int16))
        strong.append(float(delta.mean()))

    assert np.mean(strong) > np.mean(weak) * 1.5


def test_bao_loi_to_khi_khong_ve_duoc_loi():
    """Không vẽ được lỗi thì phải BÁO LỖI, không được trả về ảnh sai nhãn.

    Mặt nạ rỗng làm mọi hàm vẽ lỗi thành vô hiệu — đúng tình huống "ảnh NG
    giống hệt ảnh OK". Lúc đó im lặng trả về là gieo nhãn sai vào tập dữ liệu.
    """
    from vision_qc.synth import _apply_defect

    image = np.full((64, 64), 100.0, dtype=np.float32)
    empty_mask = np.zeros((64, 64), dtype=np.float32)
    cfg = SynthConfig(width=64, height=64, severity=0.5, defect_attempts=3)

    with pytest.raises(RuntimeError, match="Không vẽ được lỗi nào"):
        _apply_defect(image, empty_mask, np.random.default_rng(0), cfg, 20.0, image)


def test_loi_mo_van_duoc_chap_nhan():
    """Lỗi nhỏ nhưng CÓ thật thì trả về, không báo lỗi.

    `min_defect_pixels` là mức mong muốn, không phải điều kiện bắt buộc. Một
    vết xước chỉ đổi 5 điểm ảnh vẫn là một vết xước thật — đó chính là loại
    mẫu khó mà model cần học.
    """
    from vision_qc.synth import _apply_defect

    image = np.full((240, 320), 100.0, dtype=np.float32)
    mask = np.zeros((240, 320), dtype=np.float32)
    mask[80:160, 100:220] = 1.0

    cfg = SynthConfig(width=320, height=240, severity=0.5, min_defect_pixels=10**9)
    kind, out = _apply_defect(image, mask, np.random.default_rng(0), cfg, 60.0, image)

    assert kind in DEFECT_FUNCS
    assert not np.array_equal(out, image)
