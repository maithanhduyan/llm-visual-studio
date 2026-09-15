"""
__main__.py — Cửa vào của dự án.

    python -m vision_qc check                 kiểm tra toàn hệ thống
    python -m vision_qc camera                xem camera có gì, đo fps thật
    python -m vision_qc capture               THU ẢNH từ camera 2MP
    python -m vision_qc dataset               xem lại dữ liệu đã thu
    python -m vision_qc dedup                 xoá ảnh trùng
    python -m vision_qc synth                 sinh ảnh giả để thử
    python -m vision_qc train                 dạy model
    python -m vision_qc eval                  đo model và tìm ngưỡng
    python -m vision_qc station               chạy cả trạm, điều khiển PLC

Đường đi đầy đủ:

    capture  ->  train  ->  eval  ->  station
    (thu ảnh)    (dạy)     (ngưỡng)  (chạy thật)
"""

from __future__ import annotations

import argparse
import json
import sys

# Windows vẫn hay dùng cp1252, in tiếng Việt ra là lỗi ngay.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .config import Config  # noqa: E402
from .paths import MODELS_DIR, RAW_DIR, RUNS_DIR, ensure_dirs  # noqa: E402

# ---------------------------------------------------------------------
# Nạp cấu hình
# ---------------------------------------------------------------------


def load_config(path: str | None) -> Config:
    """Cấu hình từ file JSON, hoặc mặc định."""
    if not path:
        return Config()

    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config.from_dict(data)


def resolve_threshold(name: str, explicit: float | None) -> tuple[float, str]:
    """Ngưỡng để chạy thật.

    Ưu tiên: người dùng gõ tay > ngưỡng `eval` đã tìm ra > 0,5.

    Lấy ngưỡng từ báo cáo đánh giá là điểm nối quan trọng nhất giữa hai bước:
    `eval` quét để tìm ngưỡng rẻ nhất, `station` dùng đúng con số đó. Để hai
    bên tự chọn riêng thì model được đánh giá ở một ngưỡng mà lại chạy thật ở
    ngưỡng khác.
    """
    if explicit is not None:
        return explicit, "người dùng đặt"

    report_path = RUNS_DIR / f"{name}_eval.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        chosen = report.get("chosen", {})
        if "threshold" in chosen:
            return float(chosen["threshold"]), f"lấy từ {report_path.name}"

    return 0.5, "mặc định 0,5 (chưa chạy `eval`)"


# ---------------------------------------------------------------------
# Các lệnh
# ---------------------------------------------------------------------


def cmd_check(args) -> int:
    from .checks import run_all

    return run_all(verbose=not args.quiet)


def cmd_camera(args) -> int:
    from .camera import Camera, CameraError, probe
    from .config import CameraConfig

    print("Dò camera:\n")
    found = probe(range(4))
    if not found:
        print("  Không thấy camera nào.")
        print("  - Camera đã cắm chưa?")
        print("  - Có phần mềm khác đang giữ camera không (Zoom, Teams, OBS)?")
        return 1

    for report in found:
        print(f"  #{report.index}: {report.width}x{report.height} "
              f"({report.megapixels:.2f} MP) @ {report.fps:.0f} fps · {report.fourcc}")

    cfg = CameraConfig(index=args.index)
    camera = Camera(cfg)

    try:
        print(f"\nMở camera #{args.index} ở {cfg.width}x{cfg.height} {cfg.fourcc}:\n")
        report = camera.open()
        print(f"  {report.describe()}")

        if not report.resolution_ok:
            print(f"  CẢNH BÁO: xin {report.requested[0]}x{report.requested[1]}, "
                  f"driver trả {report.width}x{report.height}")

        print("\n  Khoá thông số:")
        for name, status in report.locks.items():
            mark = " " if status.startswith("OK") else "!"
            print(f"   {mark} {name:26s} {status}")

        print("\n  Đo thật (không tin con số driver khai):")
        fps = camera.measure_fps(30)
        print(f"    fps            {fps:6.1f}   (driver khai {report.fps:.1f})")

        mean, std = camera.measure_stability(20)
        print(f"    độ sáng        {mean:6.1f} ± {std:.2f}  ({std / mean * 100:.2f}%)")

        if std / mean > 0.01:
            print("      -> độ sáng dao động hơn 1%. Kiểm tra đèn nhấp nháy 50 Hz,")
            print("         hoặc phơi sáng chưa khoá được.")

        print(f"    cỡ model nhìn  {camera.frame_size}  (ROI {cfg.roi})")

        warning = cfg.exposure_warning()
        if warning:
            print(f"\n  CẢNH BÁO: {warning}")

        print(f"\n  Nhịp dây chuyền mặc định: {Config().line.parts_per_minute:.0f} "
              f"sản phẩm/phút")
        print(f"  Ngân sách mỗi quyết định: {Config().line.time_budget_ms:.0f} ms")

    except CameraError as exc:
        print(f"\nLỖI: {exc}")
        return 1
    finally:
        camera.close()

    return 0


def cmd_capture(args) -> int:
    from .capture import run_capture

    config = load_config(args.config)
    if args.roi:
        config.camera.roi = tuple(args.roi)  # type: ignore[assignment]
    if args.exposure is not None:
        config.camera.exposure = args.exposure

    run_capture(
        camera_cfg=config.camera,
        burst=args.burst,
        display=not args.no_display,
        min_ok=args.min_ok,
        min_ng=args.min_ng,
    )
    return 0


def cmd_dedup(args) -> int:
    from .capture import deduplicate

    print(f"Quét ảnh trùng trong {RAW_DIR} (dung sai {args.tolerance} bit):\n")
    removed, kept = deduplicate(RAW_DIR, args.tolerance)

    print(f"\n  đã xoá {removed} ảnh trùng · còn lại {kept}")
    return 0


def cmd_dataset(args) -> int:
    from .dataset import scan, split_by_group, statistics

    ensure_dirs()

    if args.path:
        from pathlib import Path

        root = Path(args.path)
    else:
        root = RAW_DIR

    samples = scan(root, compute_stats=True)
    if not samples:
        print(f"Chưa có ảnh nào trong {root}.")
        print("  Thu ảnh thật:  python -m vision_qc capture")
        print("  Hoặc thử ngay: python -m vision_qc synth")
        return 1

    stats = statistics(samples, duplicates=True)

    print(f"Dữ liệu trong {root}\n")
    print(f"  tổng          {stats.total:6d} ảnh")
    print(f"  OK            {stats.counts.get(0, 0):6d} ảnh")
    print(f"  NG            {stats.counts.get(1, 0):6d} ảnh")

    if stats.counts.get(0, 0) and stats.counts.get(1, 0):
        ratio = stats.counts[0] / stats.counts[1]
        print(f"  tỉ lệ OK/NG   {ratio:6.1f} : 1")
        if ratio > 5:
            print("    -> mất cân bằng. Huấn luyện vẫn được (có pos_weight),")
            print("       nhưng nên chụp thêm ảnh lỗi.")

    print(f"\n  độ sáng OK    {stats.brightness.get(0, 0):6.1f}")
    print(f"  độ sáng NG    {stats.brightness.get(1, 0):6.1f}")
    gap = stats.brightness_gap()
    print(f"  lệch          {gap:6.2f} %")
    if gap > 5.0:
        print("    -> NGUY HIỂM: hai lớp lệch sáng. Model rất dễ học")
        print("       \"sáng = tốt\" thay vì học lỗi. Chụp lại, trộn hai lớp")
        print("       trong cùng một lượt và cùng điều kiện sáng.")

    if stats.duplicates:
        print(f"\n  ẢNH TRÙNG: {len(stats.duplicates)} cặp")
        for first, second, distance in stats.duplicates[:5]:
            print(f"    {first}  ~  {second}  (lệch {distance} bit)")
        if len(stats.duplicates) > 5:
            print(f"    ... và {len(stats.duplicates) - 5} cặp nữa")
        print("    -> chạy `python -m vision_qc dedup` để xoá")

    train, val = split_by_group(samples, args.val_fraction, 0)
    print("\n  Chia theo sản phẩm (không theo ảnh):")
    print(f"    học  {len(train):6d} ảnh / {len({s.group for s in train})} sản phẩm")
    print(f"    thi  {len(val):6d} ảnh / {len({s.group for s in val})} sản phẩm")

    if len({s.group for s in samples}) < 20:
        print(f"\n  CHÚ Ý: chỉ có {len({s.group for s in samples})} sản phẩm khác nhau.")
        print("  Mỗi lần bấm là một sản phẩm — bấm nhiều lần cho cùng một chi tiết")
        print("  không tạo thêm dữ liệu, chỉ tạo thêm ảnh giống nhau.")

    return 0


def cmd_synth(args) -> int:
    from pathlib import Path

    from .synth import SynthConfig, generate

    root = Path(args.path) if args.path else RUNS_DIR.parent / "data" / "synth"

    print(f"Sinh ảnh giả vào {root}\n")
    generate(
        root,
        count_ok=args.ok,
        count_ng=args.ng,
        seed=args.seed,
        cfg=SynthConfig(width=args.width, height=args.height, severity=args.severity),
    )
    print("\n  Nhắc lại: ảnh giả KHÔNG thay được ảnh thật. Nó để thử đường ống.")
    return 0


def cmd_train(args) -> int:
    from pathlib import Path

    from .augment import AugmentConfig
    from .train import overfit_warning, run_training

    config = load_config(args.config)
    if args.image_size:
        config.model.image_size = args.image_size
    if args.width:
        config.model.width = args.width
    if args.epochs:
        config.model.epochs = args.epochs

    data_dir = Path(args.data) if args.data else RAW_DIR

    print(f"Dạy model trên {data_dir}\n")

    model, history = run_training(
        data_dir,
        name=args.name,
        model_cfg=config.model,
        camera_cfg=config.camera,
        augment_cfg=AugmentConfig(color_strength=args.color_strength),
        val_fraction=args.val_fraction,
    )

    warning = overfit_warning(history)
    if warning:
        print(f"\n  {warning}")

    print(f"\n  Bước tiếp: python -m vision_qc eval --name {args.name}")
    return 0


def cmd_eval(args) -> int:
    from pathlib import Path

    from .dataset import scan, split_by_group
    from .evaluate import run_evaluation
    from .model import load_model, restore_camera

    config = load_config(args.config)
    model_path = MODELS_DIR / f"{args.name}.pt"

    if not model_path.exists():
        print(f"Chưa có model {model_path}.")
        print(f"  Chạy trước: python -m vision_qc train --name {args.name}")
        return 1

    model, extra = load_model(model_path)

    # ROI của lúc huấn luyện, không phải ROI đang có trong cấu hình.
    restore_camera(extra, config.camera)

    data_dir = Path(args.data) if args.data else RAW_DIR

    samples = scan(data_dir, compute_stats=False)
    if not samples:
        print(f"Không có ảnh nào trong {data_dir}.")
        return 1

    # Cùng phép chia với lúc huấn luyện: cùng seed, cùng tỉ lệ.
    _, val_samples = split_by_group(samples, args.val_fraction, model.cfg.seed)

    roi_note = f" · ROI {config.camera.roi}" if config.camera.roi else ""
    print(f"Đo model {args.name} — {model.num_params:,} thông số · "
          f"ảnh vào {model.cfg.image_size}x{model.cfg.image_size}{roi_note}\n")

    run_evaluation(
        model,
        val_samples,
        model.cfg,          # cấu hình CỦA MODEL, không phải cấu hình hiện tại
        config.camera,
        cost_false_reject=args.cost_false_reject,
        cost_escape=args.cost_escape,
        max_false_reject_rate=args.max_false_reject,
        budget_ms=config.line.time_budget_ms,
        name=args.name,
    )

    print(f"\n  Bước tiếp: python -m vision_qc station --name {args.name}")
    return 0


def cmd_station(args) -> int:
    from pathlib import Path

    from .model import load_model, restore_camera
    from .plc import PlcError, make_plc
    from .station import (
        CameraSource,
        FolderSource,
        InspectionStation,
        SynthSource,
        print_report,
    )

    config = load_config(args.config)
    model_path = MODELS_DIR / f"{args.name}.pt"

    if not model_path.exists():
        print(f"Chưa có model {model_path}.")
        print(f"  Chạy trước: python -m vision_qc train --name {args.name}")
        return 1

    model, extra = load_model(model_path)

    # Vùng cắt phải giống hệt lúc huấn luyện.
    restore_camera(extra, config.camera)

    threshold, source_of_threshold = resolve_threshold(args.name, args.threshold)

    if args.plc:
        config.plc.backend = args.plc
    if args.host:
        config.plc.host = args.host
    if args.port:
        config.plc.port = args.port

    # -- nguồn ảnh --
    if args.source == "camera":
        source = CameraSource(config.camera)
    elif args.source == "folder":
        if not args.folder:
            print("--source folder cần --folder <thư mục>")
            return 1
        source = FolderSource(Path(args.folder), config.camera)
    else:
        source = SynthSource(defect_rate=args.defect_rate, seed=args.seed)

    plc = make_plc(config.plc, config.line)

    print("Chạy trạm kiểm tra\n")
    print(f"  model      {args.name} · {model.num_params:,} thông số")
    print(f"  ngưỡng     {threshold:.4f}  ({source_of_threshold})")
    print(f"  nguồn ảnh  {source.describe()}")
    print(f"  PLC        {'giả lập' if plc.is_simulated else f'{config.plc.host}:{config.plc.port}'}")
    print(f"  nhịp       {config.line.parts_per_minute:.0f} sản phẩm/phút "
          f"({config.line.part_interval_s:.2f} s một sản phẩm)")
    print(f"  ngân sách  {config.line.time_budget_ms:.0f} ms mỗi sản phẩm")
    print()

    station = InspectionStation(
        model, source, plc, config,
        threshold=threshold,
        frames_per_part=args.frames,
        fail_safe_reject=not args.fail_open,
        verbose=not args.quiet,
    )

    try:
        report = station.run(
            max_parts=args.parts,
            trigger=args.trigger,
            name=args.name,
        )
    except PlcError as exc:
        print(f"\nLỖI PLC: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n  dừng theo yêu cầu")
        return 0

    print_report(report, config.line)
    return 0


def cmd_info(args) -> int:
    """Xem model và cả dây chuyền mà không cần chạy gì."""
    from .config import Config

    config = Config()

    if args.config:
        config = load_config(args.config)

    print("Cấu hình dây chuyền\n")
    line = config.line
    print(f"  băng tải        {line.conveyor_speed_mm_s:.0f} mm/s "
          f"({line.conveyor_speed_mm_s * 60 / 1000:.1f} m/phút)")
    print(f"  khoảng cách     {line.part_pitch_mm:.0f} mm giữa hai sản phẩm")
    print(f"  camera -> van   {line.camera_to_reject_mm:.0f} mm")
    print(f"  chu kỳ quét PLC {line.plc_scan_ms:.0f} ms")
    print(f"  độ trễ van      {line.valve_delay_ms:.0f} ms")
    print(f"\n  -> năng suất     {line.parts_per_minute:.0f} sản phẩm/phút")
    print(f"  -> ngân sách     {line.time_budget_ms:.0f} ms cho mỗi quyết định")

    if line.time_budget_ms <= 0:
        print("     ÂM: van nằm quá gần camera. Không kịp quyết định.")

    print("\nCamera\n")
    cam = config.camera
    print(f"  {cam.width}x{cam.height} @ {cam.fps} fps · {cam.fourcc}")
    print(f"  phơi sáng       {cam.exposure:g} = 1/{1 / cam.exposure_seconds:.0f} s"
          f"   (dài nhất cho phép: {cam.max_exposure:.1f})")
    print(f"  độ lợi          {cam.gain:g}")
    print(f"  ROI             {cam.roi}")
    warning = cam.exposure_warning()
    if warning:
        print(f"  CẢNH BÁO: {warning}")

    print("\nPLC\n")
    plc = config.plc
    print(f"  backend         {plc.backend}")
    print(f"  địa chỉ         {plc.host}:{plc.port}")
    print(f"  M100 trigger    {plc.trigger}")
    print(f"  M200 kết quả    {plc.result_valid}")
    print(f"  M201 cờ lỗi     {plc.ng_flag}")
    print(f"  D200 điểm       {plc.score}")
    print(f"  D210 nhịp tim   {plc.heartbeat}")

    return 0


# ---------------------------------------------------------------------
# Phân tích tham số
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vision_qc",
        description="Kiểm tra lỗi sản phẩm bằng camera 2MP, điều khiển PLC loại sản phẩm.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str):
        return sub.add_parser(name, help=help_text)

    p = add("check", "kiểm tra toàn hệ thống")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_check)

    p = add("info", "xem cấu hình dây chuyền")
    p.add_argument("--config")
    p.set_defaults(func=cmd_info)

    p = add("camera", "xem camera có gì, đo fps thật")
    p.add_argument("--index", type=int, default=0)
    p.set_defaults(func=cmd_camera)

    p = add("capture", "thu ảnh từ camera")
    p.add_argument("--config")
    p.add_argument("--burst", type=int, default=1, help="số khung mỗi lần bấm")
    p.add_argument("--roi", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="vùng cắt, toạ độ tương đối 0..1")
    p.add_argument("--exposure", type=float)
    p.add_argument("--no-display", action="store_true",
                   help="không mở cửa sổ xem trước, dùng bàn phím")
    p.add_argument("--min-ok", type=int, default=150)
    p.add_argument("--min-ng", type=int, default=50)
    p.set_defaults(func=cmd_capture)

    p = add("dedup", "xoá ảnh trùng")
    p.add_argument("--tolerance", type=int, default=2, help="số bit khác nhau còn coi là trùng")
    p.set_defaults(func=cmd_dedup)

    p = add("dataset", "xem lại dữ liệu đã thu")
    p.add_argument("--path")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.set_defaults(func=cmd_dataset)

    p = add("synth", "sinh ảnh giả để thử đường ống")
    p.add_argument("--ok", type=int, default=200)
    p.add_argument("--ng", type=int, default=200)
    p.add_argument("--path")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--severity", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = add("train", "dạy model")
    p.add_argument("--config")
    p.add_argument("--data", help="thư mục dữ liệu (mặc định data/raw)")
    p.add_argument("--name", default="defect")
    p.add_argument("--epochs", type=int)
    p.add_argument("--image-size", type=int, help="cỡ ảnh đưa vào model")
    p.add_argument("--width", type=int, help="số kênh tầng đầu")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--color-strength", default="medium",
                   choices=["off", "weak", "medium", "strong"],
                   help="mức tăng cường màu (mạnh có thể xoá mất lỗi về màu)")
    p.set_defaults(func=cmd_train)

    p = add("eval", "đo model và tìm ngưỡng")
    p.add_argument("--config")
    p.add_argument("--data")
    p.add_argument("--name", default="defect")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--cost-false-reject", type=float, default=1.0,
                   help="loại oan một sản phẩm tốt tốn bao nhiêu")
    p.add_argument("--cost-escape", type=float, default=50.0,
                   help="để lọt một sản phẩm lỗi tốn bao nhiêu")
    p.add_argument("--max-false-reject", type=float, default=0.05,
                   help="trần tỉ lệ thổi oan; ngưỡng rẻ nhất phải nằm dưới mức này")
    p.set_defaults(func=cmd_eval)

    p = add("station", "chạy cả trạm, điều khiển PLC")
    p.add_argument("--config")
    p.add_argument("--name", default="defect")
    p.add_argument("--parts", type=int, default=50, help="số sản phẩm")
    p.add_argument("--source", default="synth", choices=["camera", "synth", "folder"])
    p.add_argument("--folder", help="thư mục ảnh khi --source folder")
    p.add_argument("--trigger", default="interval", choices=["interval", "plc"],
                   help="interval = theo nhịp băng tải, plc = chờ M100")
    p.add_argument("--threshold", type=float, help="mặc định: lấy từ `eval`")
    p.add_argument("--frames", type=int, default=1, help="số khung mỗi sản phẩm")
    p.add_argument("--defect-rate", type=float, default=0.3, help="tỉ lệ lỗi khi dùng ảnh giả")
    p.add_argument("--plc", choices=["sim", "mc"])
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--fail-open", action="store_true",
                   help="khi không suy luận được thì cho qua (mặc định là thổi)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_station)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
