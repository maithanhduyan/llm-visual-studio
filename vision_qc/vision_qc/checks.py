"""
checks.py — `python -m vision_qc check`: tự kiểm tra xem mọi thứ có đúng không.

Mỗi mục in ra một kết quả kèm SỐ ĐO, không phải chỉ "OK". Một dấu tích không
nói lên điều gì; "khung tin 21 byte khớp tài liệu" mới nói lên điều gì đó.

Những mục không cần phần cứng vẫn chạy được khi không có camera hay PLC, và
chúng ghi rõ là "bỏ qua" chứ không giả vờ đạt.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    skipped: bool = False

    @property
    def mark(self) -> str:
        if self.skipped:
            return "BO QUA"
        return "DAT   " if self.passed else "HONG  "


# ---------------------------------------------------------------------
# Từng mục
# ---------------------------------------------------------------------


def check_environment() -> CheckResult:
    """Máy này có đủ thứ để chạy không."""
    import torch

    gpu = "có" if torch.cuda.is_available() else "không (chạy CPU)"
    detail = (f"Python {platform.python_version()} · "
              f"PyTorch {torch.__version__} · GPU: {gpu}")
    return CheckResult("Môi trường", True, detail)


def check_model() -> CheckResult:
    """Model dựng được, chạy được, và có bao nhiêu thông số."""
    import torch

    from .config import ModelConfig
    from .model import DefectNet

    cfg = ModelConfig()
    model = DefectNet(cfg)

    channels = 1 if cfg.grayscale else 3
    dummy = torch.randn(2, channels, cfg.image_size, cfg.image_size)

    with torch.no_grad():
        out = model(dummy)

    shape_ok = tuple(out.shape) == (2,)
    return CheckResult(
        "Model",
        shape_ok,
        f"{model.num_params:,} thông số · vào "
        f"{channels}x{cfg.image_size}x{cfg.image_size} · ra {tuple(out.shape)}",
    )


def check_mc_frames() -> CheckResult:
    """Khung tin MC phải khớp TỪNG BYTE với tài liệu Mitsubishi."""
    from .mc import CMD_BATCH_READ, build_request, encode_read, parse_device

    sub, data = encode_read(parse_device("D100"), 1)
    frame = build_request(CMD_BATCH_READ, sub, data)

    expected = bytes([
        0x50, 0x00, 0x00, 0xFF, 0xFF, 0x03, 0x00, 0x0C, 0x00,
        0x10, 0x00, 0x01, 0x04, 0x00, 0x00, 0xA8, 0x64, 0x00, 0x00, 0x01, 0x00,
    ])

    ok = frame == expected
    detail = (f"đọc D100 -> {len(frame)} byte, "
              f"{'khớp' if ok else 'LỆCH'} tài liệu SH-080008")
    if not ok:
        detail += f"\n         nhận: {frame.hex(' ')}\n         cần:  {expected.hex(' ')}"
    return CheckResult("Khung tin MC", ok, detail)


def check_mc_roundtrip() -> CheckResult:
    """Nói chuyện thật qua TCP với PLC giả."""
    from .mc import McClient, SimMcPlc

    try:
        with SimMcPlc() as server:
            with McClient(server.host, server.port, timeout_s=2.0) as client:
                started = time.perf_counter()
                client.write_word("D100", 12345)
                word = client.read_word("D100")

                client.write_bit("M200", True)
                bit = client.read_bit("M200")

                client.write_words("D300", [0, -1, 32767, -32768])
                words = client.read_words("D300", 4)

                client.write_bits("M400", [1, 0, 1, 1, 0, 1, 0, 0, 1])
                bits = client.read_bits("M400", 9)

                elapsed = (time.perf_counter() - started) * 1000.0
    except Exception as exc:  # noqa: BLE001
        return CheckResult("MC qua TCP", False, f"lỗi: {exc}")

    ok = (word == 12345 and bit is True and words == [0, -1, 32767, -32768]
          and bits == [1, 0, 1, 1, 0, 1, 0, 0, 1])

    return CheckResult(
        "MC qua TCP",
        ok,
        f"16 lượt đọc/ghi trong {elapsed:.0f} ms · số âm và mảng bit nguyên vẹn",
    )


def check_plc_handshake() -> CheckResult:
    """Trình tự bắt tay với PLC: gửi kết quả, chờ xác nhận."""
    from .config import LineConfig
    from .plc import Decision, PlcError, SimRejectPlc

    plc = SimRejectPlc(LineConfig())
    plc.open()

    try:
        plc.fire_trigger(True)
        if not plc.wait_trigger(1.0):
            return CheckResult("Bắt tay PLC", False, "không thấy trigger")

        elapsed = plc.send(Decision(part_id=1, score=0.93, is_ng=True))
        plc.heartbeat()

        stats = plc.stats()
        ok = stats["kết quả đã gửi"] == 1 and stats["đã thổi"] == 1

        return CheckResult(
            "Bắt tay PLC",
            ok,
            f"gửi + xác nhận trong {elapsed:.1f} ms · "
            f"điểm 0,93 -> thổi · nhịp tim {stats['nhịp tim']}",
        )
    except PlcError as exc:
        return CheckResult("Bắt tay PLC", False, str(exc))
    finally:
        plc.close()


def check_plc_guard() -> CheckResult:
    """Ghi đè lên kết quả chưa được PLC nhận PHẢI bị chặn."""
    from .config import LineConfig
    from .plc import Decision, PlcError, SimRejectPlc

    plc = SimRejectPlc(LineConfig())
    plc.open()
    plc.result_valid = True  # giả vờ PLC chưa xử lý kết quả trước

    try:
        plc.send(Decision(part_id=1, score=0.9, is_ng=True))
        return CheckResult("Chặn ghi đè", False,
                           "ĐÃ CHO ghi đè lên kết quả chưa xử lý — nguy hiểm")
    except PlcError:
        return CheckResult("Chặn ghi đè", True,
                           "từ chối ghi đè khi kết quả trước chưa được nhận")
    finally:
        plc.ready = True


def check_timing() -> CheckResult:
    """Ngân sách thời gian có dương không, và nhịp dây chuyền là bao nhiêu."""
    from .config import LineConfig

    line = LineConfig()
    budget = line.time_budget_ms

    ok = budget > 0
    detail = (f"ngân sách {budget:.0f} ms · "
              f"{line.camera_to_reject_mm:.0f} mm / {line.conveyor_speed_mm_s:.0f} mm/s "
              f"- {line.plc_scan_ms:.0f} ms PLC - {line.valve_delay_ms:.0f} ms van · "
              f"{line.parts_per_minute:.0f} sản phẩm/phút")

    if not ok:
        detail += "  <- ÂM: van nằm quá gần camera, không kịp quyết định"

    return CheckResult("Ngân sách thời gian", ok, detail)


def check_station_end_to_end() -> CheckResult:
    """Cả trạm chạy thật: ảnh giả -> model chưa học -> PLC giả.

    Dùng model CHƯA huấn luyện, nên đừng trông vào độ chính xác. Mục này kiểm
    tra đường ống có thông không, và đo thời gian từng công đoạn.
    """
    from .config import CameraConfig, Config, ModelConfig
    from .model import DefectNet
    from .station import InspectionStation, SynthSource

    cfg = Config()
    cfg.model = ModelConfig(image_size=128, width=16)
    cfg.camera = CameraConfig(roi=None)
    cfg.line.plc_scan_ms = 2.0

    model = DefectNet(cfg.model)
    source = SynthSource(defect_rate=0.4, seed=3)
    source.config.width, source.config.height = 640, 480

    from .plc import make_plc

    plc = make_plc(cfg.plc, cfg.line)
    station = InspectionStation(model, source, plc, cfg, threshold=0.5)
    report = station.run(max_parts=12, trigger="interval")

    ok = report.parts == 12 and report.latency.get("tổng p95", 0) > 0
    detail = (f"12 sản phẩm · p95 {report.latency.get('tổng p95', 0):.1f} ms "
              f"(chụp {report.latency.get('chụp', 0):.1f} + "
              f"suy luận {report.latency.get('suy luận', 0):.1f} + "
              f"PLC {report.latency.get('ghi plc', 0):.1f})")

    if report.parts != 12:
        detail = f"chỉ chạy được {report.parts}/12 sản phẩm"

    return CheckResult("Trạm end-to-end", ok, detail)


def check_dataset() -> CheckResult:
    """Có dữ liệu thật chưa, và có dấu hiệu lối tắt độ sáng không."""
    from .dataset import scan, statistics
    from .paths import RAW_DIR

    samples = scan(RAW_DIR, compute_stats=True)
    if not samples:
        return CheckResult(
            "Dữ liệu thật", False,
            f"chưa có ảnh trong {RAW_DIR} — chạy `python -m vision_qc capture`",
            skipped=True,
        )

    stats = statistics(samples, duplicates=True)
    ok_count = stats.counts.get(0, 0)
    ng_count = stats.counts.get(1, 0)
    gap = stats.brightness_gap()

    problems: list[str] = []
    if ok_count == 0 or ng_count == 0:
        problems.append("thiếu hẳn một lớp")
    if gap > 5.0:
        problems.append(f"độ sáng hai lớp lệch {gap:.1f}% (nguy cơ học ánh sáng)")
    if stats.duplicates:
        problems.append(f"{len(stats.duplicates)} ảnh trùng")

    detail = (f"{stats.total} ảnh (OK {ok_count} · NG {ng_count}) · "
              f"lệch sáng {gap:.2f}% · {len(stats.duplicates)} ảnh trùng")

    return CheckResult("Dữ liệu thật", not problems, detail)


def check_camera() -> CheckResult:
    """Camera có cắm không, có lên 2MP không, có khoá được không."""
    from .camera import Camera, CameraError, probe
    from .config import CameraConfig

    found = probe(range(3))
    if not found:
        return CheckResult(
            "Camera", False, "không thấy camera nào — cắm camera rồi thử lại",
            skipped=True,
        )

    cfg = CameraConfig()
    camera = Camera(cfg)

    try:
        report = camera.open()
        fps = camera.measure_fps(20)
        stable, verdict = camera.verify_lock(20)
        camera.close()
    except CameraError as exc:
        return CheckResult("Camera", False, str(exc).split("\n")[0])

    # Cột `locks` chỉ là thông tin tham khảo — DirectShow hay nhận lệnh rồi trả
    # về -1 vì không có thang đo để đọc. Con số đáng tin là `verify_lock`, tức
    # là ĐO HẬU QUẢ: cảnh đứng yên thì độ sáng có đứng yên không.
    refused = report.refused

    detail = (f"{report.width}x{report.height} ({report.megapixels:.2f} MP, "
              f"xin {report.requested[0]}x{report.requested[1]}) · "
              f"{fps:.1f} fps · {report.fourcc}\n"
              f"         độ sáng {verdict}")

    if not report.resolution_ok:
        detail += "\n         <- driver không trả đúng cỡ yêu cầu"
    if fps <= 10:
        detail += "\n         <- quá chậm: kiểm tra FOURCC có phải MJPG không"
    if refused:
        detail += f"\n         driver từ chối: {', '.join(refused)}"

    return CheckResult("Camera", report.resolution_ok and fps > 10 and stable, detail)


# ---------------------------------------------------------------------
# Chạy tất cả
# ---------------------------------------------------------------------


CHECKS = (
    check_environment,
    check_model,
    check_mc_frames,
    check_mc_roundtrip,
    check_plc_handshake,
    check_plc_guard,
    check_timing,
    check_station_end_to_end,
    check_dataset,
    check_camera,
)


def run_all(verbose: bool = True) -> int:
    """Chạy mọi mục. Trả về 0 nếu không có mục nào hỏng thật."""
    print("Kiểm tra toàn hệ thống\n")

    results: list[CheckResult] = []

    for check in CHECKS:
        try:
            result = check()
        except Exception as exc:  # noqa: BLE001
            result = CheckResult(check.__name__.replace("check_", ""), False,
                                 f"lỗi không lường trước: {exc}")
        results.append(result)

        if verbose:
            print(f"  [{result.mark}] {result.name:20s} {result.detail}")

    passed = sum(1 for r in results if r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.passed and not r.skipped)

    print(f"\n  {passed} đạt · {failed} hỏng · {skipped} bỏ qua (thiếu phần cứng)")

    # Máy chưa có camera và chưa có dữ liệu là chuyện bình thường ở giai đoạn
    # đầu. Chỉ coi là hỏng khi thứ đáng lẽ phải chạy lại không chạy.
    return 1 if failed else 0
