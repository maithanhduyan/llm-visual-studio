"""
__main__.py — Cửa vào của dự án.

    python -m pneumatic_vector check                 kiểm tra mọi thứ có chạy đúng
    python -m pneumatic_vector expert                chuyên gia viết tay bay thử
    python -m pneumatic_vector train                 dạy model bắt chước chuyên gia
    python -m pneumatic_vector dagger                thêm các vòng DAgger cho đỡ rơi
    python -m pneumatic_vector eval                  đo tỉ lệ đạt của model đã lưu
    python -m pneumatic_vector fly                   ghi chuyến bay ra file cho trình xem
    python -m pneumatic_vector viewer                mở trình xem 3D
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

# Windows mặc định vẫn hay dùng cp1252, in tiếng Việt ra là lỗi ngay.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .expert import Mission  # noqa: E402
from .paths import FLIGHTS_DIR, VIEWER_DIR, ensure_dirs  # noqa: E402
from .simulate import fly, fly_expert, save_flight  # noqa: E402
from .vehicle import Vehicle  # noqa: E402

TARGETS = (10.0, 12.0, 15.0)


# ---------------------------------------------------------------------
# In kết quả cho gọn
# ---------------------------------------------------------------------


def evaluate(vehicle: Vehicle, policy, runs: int, targets=TARGETS, quiet: bool = False):
    """Cho `policy` bay `runs` chuyến cho mỗi đề bài. Trả về tỉ lệ đạt."""
    passed = 0
    total = 0

    for altitude in targets:
        mission = Mission(target_altitude=altitude)
        ok = 0
        speeds: list[float] = []
        peaks: list[float] = []

        for seed in range(runs):
            result = fly(vehicle, policy, mission, seed=seed)
            ok += result.success(mission)
            speeds.append(result.landing_speed)
            peaks.append(result.peak_altitude)

        passed += ok
        total += runs

        if not quiet:
            print(
                f"    {altitude:4.1f} m -> {ok:2d}/{runs} · "
                f"lên {min(peaks):5.2f}-{max(peaks):5.2f} m · "
                f"chạm đất {min(speeds):.2f}-{max(speeds):.2f} m/s"
            )

    return passed / max(total, 1)


# ---------------------------------------------------------------------
# Các lệnh
# ---------------------------------------------------------------------


def cmd_check(args) -> int:
    from .checks import run_all

    return run_all(verbose=not args.quiet)


def cmd_expert(args) -> int:
    vehicle = Vehicle()

    print("Chuyên gia viết tay — không học gì cả, chỉ là máy trạng thái\n")

    for altitude in args.targets:
        mission = Mission(target_altitude=altitude)
        result = fly_expert(vehicle, mission, seed=args.seed)

        print(
            f"  {altitude:4.1f} m · {result.summary()} · "
            f"{'ĐẠT' if result.success(mission) else 'HỎNG'}"
        )

    print(f"\n  khí mang theo {vehicle.air_mass * 1000:.0f} g · "
          f"khối lượng {vehicle.total_mass():.3f} kg")
    return 0


def cmd_train(args) -> int:
    from .train import save_pilot, train_pilot

    vehicle = Vehicle()

    print(f"Học bắt chước: {args.flights} chuyến bay của chuyên gia\n")

    model, tokenizer, history, val_loss = train_pilot(
        vehicle, flights=args.flights, steps=args.steps, seed=args.seed
    )

    path = save_pilot(model, tokenizer, args.name)
    print(f"\n  đã lưu vào {path}")

    from .policy import LearnedPilot

    print("\nĐo trên đường bay của chính model (đây mới là điều đáng quan tâm):")
    score = evaluate(vehicle, LearnedPilot(model, tokenizer), args.runs)
    print(f"  -> tỉ lệ đạt {score:.0%}")

    if score < 0.5:
        print("  (thấp thế này là bình thường với học bắt chước thuần — "
              "chạy `dagger` để chữa.)")

    return 0


def cmd_dagger(args) -> int:
    from .dagger import dagger_round
    from .dataset import generate
    from .policy import LearnedPilot
    from .train import save_pilot, train_pilot

    vehicle = Vehicle()

    print("DAgger — cho model bay rồi thầy sửa ngay trên đường model đi\n")
    print("Vòng 0: học bắt chước thuần")

    # Bay bằng chuyên gia một lần, giữ lại đống dữ liệu này.
    expert_data = generate(vehicle, flights=args.flights, seed=args.seed)

    model, tokenizer, _, _ = train_pilot(
        vehicle, steps=args.steps, seed=args.seed, examples=expert_data, quiet=False
    )

    score = evaluate(vehicle, LearnedPilot(model, tokenizer), args.runs)
    print(f"  -> đường model tự bay: {score:.0%}\n")

    # Đây là chữ "Aggregation" trong tên DAgger: dữ liệu của mọi vòng được
    # DỒN LẠI, không phải thay thế nhau.
    #
    # Bản đầu tiên của file này gọi `dagger_round(..., [], ...)` — tức là mỗi
    # vòng vứt sạch dữ liệu cũ và chỉ học trên những chỗ model vừa bay hỏng.
    # Kết quả đo được: vòng 0 đạt 7%, sau một vòng DAgger còn **0%**. Model
    # mất luôn đường bay chuẩn mà nó đã học được. Càng luyện càng dở.
    data = list(expert_data)

    for round_index in range(1, args.rounds + 1):
        print(f"Vòng {round_index}: DAgger")

        before = len(data)
        data = dagger_round(vehicle, model, tokenizer, data,
                            flights=args.dagger_flights, seed=round_index * 17)
        print(f"  thêm {len(data) - before:,} ví dụ tại những chỗ MODEL đi qua "
              f"(tổng {len(data):,})")

        model, tokenizer, _, val_loss = train_pilot(
            vehicle, steps=args.steps, seed=args.seed + round_index,
            examples=data, quiet=True,
        )
        print(f"  loss thi {val_loss:.4f}")

        score = evaluate(vehicle, LearnedPilot(model, tokenizer), args.runs)
        print(f"  -> đường model tự bay: {score:.0%}\n")

    path = save_pilot(model, tokenizer, args.name)
    print(f"đã lưu vào {path}")

    # Đo lần cuối kỹ hơn, để con số đem báo cáo không phải là may rủi.
    if args.final_runs > args.runs:
        print(f"\nĐo lần cuối, {args.final_runs} chuyến mỗi đề bài:")
        final = evaluate(vehicle, LearnedPilot(model, tokenizer), args.final_runs)
        print(f"  -> tỉ lệ đạt {final:.0%}")

    return 0


def cmd_eval(args) -> int:
    from .policy import LearnedPilot

    vehicle = Vehicle()
    pilot = LearnedPilot.load(args.name)

    print(f"Model {args.name} — {sum(p.numel() for p in pilot.model.parameters()):,} "
          f"thông số · {len(pilot.tokenizer)} ký tự\n")

    score = evaluate(vehicle, pilot, args.runs)
    print(f"\n  -> tỉ lệ đạt {score:.0%}")
    print(f"  lệnh không hiểu được: {pilot.invalid_rate:.2%} "
          f"({pilot.invalid}/{pilot.total})")
    return 0


def cmd_fly(args) -> int:
    """Ghi chuyến bay ra JSON để trình xem 3D phát lại."""
    ensure_dirs()

    vehicle = Vehicle()
    written: list[tuple[str, float, float, bool]] = []

    pilots: list[tuple[str, object]] = [("chuyentay", None)]  # None = chuyên gia

    if args.with_model:
        from .policy import LearnedPilot

        pilots.append((args.name, LearnedPilot.load(args.name)))

    for name, pilot in pilots:
        for altitude in args.targets:
            mission = Mission(target_altitude=altitude)

            for seed in range(args.runs):
                if pilot is None:
                    result = fly_expert(vehicle, mission, seed=seed)
                else:
                    result = fly(vehicle, pilot, mission, seed=seed)

                tag = "chuyentay" if pilot is None else args.name
                flight_name = f"{tag}_{altitude:.0f}m_{seed + 1:02d}"

                save_flight(
                    result,
                    FLIGHTS_DIR / f"{flight_name}.json",
                    meta={
                        "policy": tag,
                        "target_altitude": altitude,
                        "seed": seed,
                        "landing_speed": round(result.landing_speed, 2),
                        "success": result.success(mission),
                    },
                )

                written.append(
                    (flight_name, result.peak_altitude, result.landing_speed,
                     result.success(mission))
                )

    print(f"Đã ghi {len(written)} chuyến bay vào {FLIGHTS_DIR}\n")

    for name, peak, speed, ok in written:
        print(f"  {name:22s} lên {peak:5.2f} m · chạm đất {speed:4.2f} m/s · "
              f"{'ĐẠT' if ok else 'HỎNG'}")

    print(f"\nMở trình xem:  python -m pneumatic_vector viewer")
    return 0


def cmd_viewer(args) -> int:
    if not (VIEWER_DIR / "node_modules").exists():
        print("Chưa cài thư viện cho trình xem. Chạy trước:\n"
              f"  cd {VIEWER_DIR}\n  bun install")
        return 1

    count = len(list(FLIGHTS_DIR.glob("*.json")))
    if count == 0:
        print("Chưa có chuyến bay nào. Chạy trước:\n"
              "  python -m pneumatic_vector fly")
        return 1

    print(f"Đang mở trình xem với {count} chuyến bay...")
    return subprocess.call(["bun", "run", "server.ts"], cwd=VIEWER_DIR)


def cmd_why(args) -> int:
    """Vì sao model hỏng? Nhìn vào lúc nó bắt đầu hãm."""
    from .explain import explain

    return explain(name=args.name, seed=args.seed)


def cmd_export(args) -> int:
    """Xuất model + dữ liệu đối chiếu cho trình duyệt chạy sống."""
    from .export import main as export_main

    return export_main()


def cmd_fins(args) -> int:
    """Cánh đuôi: có cần không, cần bao nhiêu, và nó không làm được gì."""
    from .fins import main as fins_main

    return fins_main()


def cmd_flights(args) -> int:
    files = sorted(FLIGHTS_DIR.glob("*.json"))

    if not files:
        print("Chưa có chuyến bay nào.")
        return 0

    print(f"{len(files)} chuyến bay trong {FLIGHTS_DIR}\n")

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("summary", {})

        print(
            f"  {path.stem:22s} lên {s.get('peak_altitude', 0):5.2f} m · "
            f"chạm đất {s.get('landing_speed', 0):4.2f} m/s · "
            f"{s.get('duration', 0):4.1f} s · "
            f"{'ĐẠT' if not s.get('crashed') else 'HỎNG'}"
        )

    return 0


# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pneumatic_vector",
        description="Động cơ đẩy vector khí nén — phóng lên 10-15 m rồi hạ xuống êm.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="kiểm tra mọi thứ có chạy đúng")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("expert", help="chuyên gia viết tay bay thử")
    p.add_argument("--targets", type=float, nargs="+", default=list(TARGETS))
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_expert)

    p = sub.add_parser("train", help="dạy model bắt chước chuyên gia")
    p.add_argument("--flights", type=int, default=120)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--runs", type=int, default=10, help="số chuyến mỗi đề bài khi đo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="pilot")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("dagger", help="thêm các vòng DAgger")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--flights", type=int, default=120, help="chuyến bay ở vòng 0")
    p.add_argument("--dagger-flights", type=int, default=70, help="chuyến mỗi vòng DAgger")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--runs", type=int, default=5, help="số chuyến mỗi đề bài giữa các vòng")
    p.add_argument("--final-runs", type=int, default=10,
                   help="số chuyến mỗi đề bài khi đo lần cuối")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--name", default="pilot")
    p.set_defaults(func=cmd_dagger)

    p = sub.add_parser("eval", help="đo tỉ lệ đạt của model đã lưu")
    p.add_argument("--name", default="pilot")
    p.add_argument("--runs", type=int, default=10)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("fly", help="ghi chuyến bay ra file cho trình xem 3D")
    p.add_argument("--targets", type=float, nargs="+", default=list(TARGETS))
    p.add_argument("--runs", type=int, default=3, help="số chuyến mỗi đề bài")
    p.add_argument("--with-model", action="store_true",
                   help="ghi cả chuyến do model lái, để so sánh")
    p.add_argument("--name", default="pilot")
    p.set_defaults(func=cmd_fly)

    p = sub.add_parser("flights", help="liệt kê các chuyến bay đã ghi")
    p.set_defaults(func=cmd_flights)

    p = sub.add_parser("why", help="vì sao model hỏng — nhìn lúc nó bắt đầu hãm")
    p.add_argument("--name", default="pilot")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_why)

    p = sub.add_parser("export", help="xuất model cho trình duyệt chạy sống")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("fins", help="cánh đuôi: có cần không, cần bao nhiêu")
    p.set_defaults(func=cmd_fins)
    p = sub.add_parser("viewer", help="mở trình xem 3D")
    p.set_defaults(func=cmd_viewer)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
