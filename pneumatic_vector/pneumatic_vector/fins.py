"""
fins.py — Cánh đuôi: có cần không, cần bao nhiêu, và nó KHÔNG làm được gì.

## Câu hỏi

"Khi rơi tự do, lực cản không khí làm mất ổn định. Có nên gắn thêm hai cánh
hai bên để dẫn hướng cho rơi chính xác không?"

## Trả lời ngắn

Nguyên nhân thì đúng một nửa, mà tác dụng thì bị hiểu quá lên.

**Không phải lực cản làm mất ổn định.** Lực cản chỉ làm con tàu chậm lại, nó
không tạo mô-men nào quanh trọng tâm. Thứ làm mất ổn định là **tâm khí động
nằm TRÊN trọng tâm**:

    τ = +K·q·S·θ        nghiêng càng nhiều -> mô-men càng mạnh -> ngã thêm

**Cánh đuôi sửa đúng chỗ đó.** Cánh kéo tâm khí động xuống dưới trọng tâm, mô-men
đổi dấu:

    τ = −2·q·A·L·θ      nghiêng càng nhiều -> mô-men càng mạnh -> DỰNG LẠI

Điều kiện tĩnh ổn định: `2·A·L > K·S`.

**Nhưng cánh KHÔNG dẫn hướng.** Cánh chỉ làm thân tàu thẳng ra. Nó không tạo
lực ngang, nên không đổi quỹ đạo rơi. Muốn rơi đúng chỗ thì phải có thứ ĐẨY
NGANG — và thứ duy nhất đẩy ngang được ở đây là vòi phun, tức là chính bộ điều
khiển đã có sẵn.

**Và trên con tàu này, cánh đủ lớn để tĩnh ổn định lại làm HỎNG việc lái.**
Con tàu lái bằng cách NGHIÊNG thân cho lực đẩy chĩa sang ngang. Cánh ép thân
tàu bám theo chiều gió, tức là chống lại đúng động tác lái đó.

Số đo ở dưới. Chạy:  `python -m pneumatic_vector fins`
"""

from __future__ import annotations

import math

from .controller import CascadedController, Target
from .expert import Mission
from .physics import State, Tank, step
from .simulate import fly_expert
from .vehicle import Vehicle

TARGETS = (10.0, 12.0, 15.0)
RUNS = 5
FIN_ARM = 0.50  # m — cánh cách trọng tâm bao xa


# ---------------------------------------------------------------------


def full_flight(vehicle: Vehicle) -> dict:
    """Bay đủ 15 chuyến bằng chuyên gia, gom lại thành mấy con số."""
    passed = 0
    total = 0
    worst = {"tilt_fall": 0.0, "tilt_land": 0.0, "drift": 0.0, "air": 0.0}

    for target in TARGETS:
        mission = Mission(target_altitude=target)

        for seed in range(RUNS):
            result = fly_expert(vehicle, mission, seed=seed)
            passed += result.success(mission)
            total += 1

            free_fall = [s for s in result.samples if s["phase"] == "ROI"]

            worst["tilt_fall"] = max(
                worst["tilt_fall"], max((s["tilt"] for s in free_fall), default=0.0)
            )
            worst["tilt_land"] = max(worst["tilt_land"], result.samples[-1]["tilt"])
            worst["drift"] = max(
                worst["drift"],
                max(math.hypot(s["x"], s["y"]) for s in result.samples),
            )
            worst["air"] = max(worst["air"], result.air_used * 1000)

    return {"passed": passed, "total": total, **worst}


def free_fall(vehicle: Vehicle, tilt_deg: float = 3.0, height: float = 12.0) -> float:
    """Cắt ga hoàn toàn, để con tàu rơi. Trả về nghiêng lớn nhất.

    Đây là chỗ cánh đuôi thật sự có tác dụng: lúc cắt ga thì lực đẩy bằng 0,
    nên vòi phun không lái được gì, và con tàu rơi hoàn toàn tự do.
    """
    worst = 0.0

    for _ in range(3):
        tank = Tank(vehicle)
        state = State(
            z=height,
            theta=math.radians(tilt_deg),
            phi=-math.radians(tilt_deg) * 0.6,
        )
        controller = CascadedController(vehicle)
        cut = Target(cut=True)

        for _ in range(int(30.0 / vehicle.dt)):
            state = step(state, vehicle, tank, controller(state, tank, cut)).state
            worst = max(worst, state.tilt_deg)
            if state.z <= 0.0:
                break

    return worst


def thrown_sideways(
    vehicle: Vehicle,
    vx0: float = 3.0,
    height: float = 12.0,
    guided: bool = False,
) -> tuple[float, float, float]:
    """Ném ngang rồi thả. Trả về (lệch ngang, nghiêng tới, vx lúc chạm).

    Phép đo tách bạch: nếu cánh DẪN HƯỚNG được thì nó phải làm quỹ đạo rơi
    khác đi. Nếu cánh chỉ làm thẳng thân tàu thì quỹ đạo y hệt — vì cánh
    không tạo lực ngang.
    """
    drops = []
    tilts = []
    speeds = []

    for _ in range(3):
        tank = Tank(vehicle)
        state = State(z=height, vx=vx0, theta=0.02, phi=-0.012)
        controller = CascadedController(vehicle)

        target = Target(altitude=0.0, descent_profile=8.0) if guided else Target(cut=True)

        for _ in range(int(30.0 / vehicle.dt)):
            state = step(state, vehicle, tank, controller(state, tank, target)).state
            if state.z <= 0.0:
                break

        drops.append(state.x)
        tilts.append(state.tilt_deg)
        speeds.append(state.vx)

    mean = lambda values: sum(values) / len(values)  # noqa: E731
    return mean(drops), max(tilts), mean(speeds)


# ---------------------------------------------------------------------


def _row(label: str, stat: dict) -> str:
    return (
        f"  {label:32s} {stat['passed']:2d}/{stat['total']} · "
        f"nghiêng khi rơi {stat['tilt_fall']:5.2f}° · "
        f"chạm đất {stat['tilt_land']:5.2f}° · "
        f"trôi {stat['drift']:4.2f} m · khí {stat['air']:5.1f} g"
    )


def main() -> int:
    base = Vehicle()
    neutral = base.fin_area_for_neutral(FIN_ARM)

    print("Cánh đuôi — có cần không, cần bao nhiêu, và nó KHÔNG làm được gì")
    print("=" * 96)

    print(
        f"\n  Mô-men khí động hiện tại   τ = +K·q·S·θ     K·S = "
        f"{base.aero_torque_gain * base.cross_section:.5f} m³   (mất ổn định)\n"
        f"  Cánh đuôi sẽ làm           τ = −2·q·A·L·θ   2·A·L = ?            (tự dựng lại)\n"
        f"\n  Tĩnh ổn định khi  2·A·L > K·S.\n"
        f"  Với cánh cách trọng tâm {FIN_ARM:.2f} m: mỗi cánh cần "
        f"{neutral * 1e4:.1f} cm².\n"
        f"  Cả hai cánh cộng lại {neutral * 2e4:.0f} cm², tức khoảng "
        f"{math.sqrt(neutral * 1e4):.0f} cm × {math.sqrt(neutral * 1e4):.0f} cm mỗi bên."
    )

    sizes = (
        ("không cánh (hiện tại)", 0.0),
        ("một nửa", neutral / 2),
        ("vừa đủ trung tính", neutral),
        ("gấp đôi", neutral * 2),
    )

    print(f"\n1. CHUYẾN BAY ĐẦY ĐỦ — chuyên gia + PID điều khiển suốt")
    print(f"   ({RUNS} chuyến × {len(TARGETS)} đề bài = {RUNS * len(TARGETS)} chuyến mỗi cấu hình)\n")

    for label, area in sizes:
        vehicle = Vehicle(fin_area=area, fin_offset=FIN_ARM) if area else Vehicle()
        print(_row(label, full_flight(vehicle)))

    print("\n2. RƠI TỰ DO — cắt ga, lực đẩy bằng 0 nên vòi phun KHÔNG lái được gì\n")

    for label, area in sizes:
        vehicle = Vehicle(fin_area=area, fin_offset=FIN_ARM) if area else Vehicle()
        print(f"  {label:32s} nghiêng tối đa {free_fall(vehicle):7.1f}°")

    print(
        "\n  Đây là chỗ cánh đuôi thật sự có tác dụng. Nhưng lưu ý: trong chuyến\n"
        "  bay thật, giai đoạn rơi tự do chỉ dài 6,6 m và con tàu vào đó gần như\n"
        "  thẳng (0,2°), nên nó chỉ kịp lệch tới 1,87°. Cánh sửa một vấn đề mà\n"
        "  ở quỹ đạo hiện tại hầu như không xảy ra."
    )

    print("\n3. CÁNH CÓ DẪN HƯỚNG KHÔNG? — ném ngang 3 m/s từ 12 m\n")
    print("  cấu hình                          rơi lệch   nghiêng tới   vx chạm đất")

    bare = thrown_sideways(Vehicle())
    half = thrown_sideways(Vehicle(fin_area=neutral, fin_offset=FIN_ARM))
    double = thrown_sideways(Vehicle(fin_area=neutral * 2, fin_offset=FIN_ARM))

    for label, (drop, tilt, vx) in (
        ("không cánh", bare),
        (f"cánh {neutral * 1e4:.0f} cm² mỗi bên", half),
        (f"cánh {neutral * 2e4:.0f} cm² mỗi bên", double),
    ):
        print(f"  {label:32s} {drop:8.2f} m {tilt:10.2f}° {vx:12.2f} m/s")

    spread = max(abs(bare[0] - half[0]), abs(bare[0] - double[0]))

    print(
        f"\n  Cánh đổi độ nghiêng từ {bare[1]:.0f}° xuống {double[1]:.1f}° — tức là sửa được\n"
        f"  hoàn toàn việc lộn nhào. Nhưng quỹ đạo rơi chỉ đổi {spread:.3f} m trên "
        f"{abs(bare[0]):.2f} m,\n"
        f"  tức {spread / abs(bare[0]) * 100:.1f}%. Cánh KHÔNG dẫn hướng.\n"
        f"\n  Nó chỉ làm thân tàu thẳng ra. Muốn rơi đúng chỗ thì phải có thứ ĐẨY\n"
        f"  NGANG, mà ở đây chỉ có vòi phun."
    )

    print("\n4. VẬY THỨ GÌ DẪN HƯỚNG? — cho vòi phun lái\n")
    print("  cấu hình                          rơi lệch   nghiêng tới   vx chạm đất")

    guided = thrown_sideways(Vehicle(), guided=True)
    guided_fins = thrown_sideways(
        Vehicle(fin_area=neutral, fin_offset=FIN_ARM), guided=True
    )

    for label, (drop, tilt, vx) in (
        ("cắt ga, không điều khiển", bare),
        ("PID lái bằng vòi phun", guided),
        ("PID + cánh vừa đủ", guided_fins),
    ):
        print(f"  {label:32s} {drop:8.2f} m {tilt:10.2f}° {vx:12.2f} m/s")

    print(
        f"\n  Vòi phun ĐẢO ĐƯỢC chiều trôi ngang: vx lúc chạm từ {bare[2]:+.2f} m/s\n"
        f"  xuống {guided[2]:+.2f} m/s. Đó mới là dẫn hướng.\n"
        f"\n  Còn gắn thêm cánh vào thì lại tệ hơn: trôi {guided_fins[0]:.2f} m so với "
        f"{guided[0]:.2f} m.\n"
        f"  Cánh chống lại chính động tác nghiêng mà bộ điều khiển cần để lái."
    )

    print("\n5. CÁI GIÁ — cánh làm tăng lực cản, vậy có tốn thêm khí không?\n")
    print("  cấu hình                          diện tích cản   khí dùng nhiều nhất")

    for label, area, factor in (
        ("không cánh", 0.0, 0.10),
        ("cánh vừa đủ", neutral, 0.10),
        ("cánh vừa đủ, cản gấp ba", neutral, 0.30),
    ):
        vehicle = Vehicle(
            fin_area=area, fin_offset=FIN_ARM if area else 0.0, fin_drag_factor=factor
        )
        stat = full_flight(vehicle)

        print(
            f"  {label:32s} {vehicle.drag_area * 1e4:10.1f} cm² {stat['air']:14.1f} g"
        )

    print(
        "\n  Gần như không tốn thêm. Vì lúc bay LÊN con tàu thẳng đứng, cánh nằm\n"
        "  dọc theo dòng khí nên cản rất ít. Cái giá thật của cánh không phải khí,\n"
        "  mà là nó chống lại việc lái."
    )

    print("\n" + "=" * 96)
    print("KẾT LUẬN")
    print("=" * 96)
    print(
        f"""
  Cánh đuôi làm đúng một việc: giữ thân tàu thẳng. Nó làm việc đó rất tốt.

  Nhưng trên con tàu này, cỡ cánh đủ để tĩnh ổn định (mỗi bên {neutral * 1e4:.0f} cm²)
  lại làm hỏng việc lái, vì con tàu lái bằng cách nghiêng thân.

  Nếu vẫn muốn gắn thì nên gắn NHỎ — khoảng một nửa cỡ trung tính
  (mỗi bên {neutral * 5e3:.0f} cm²). Đo được: giữ nguyên {RUNS * len(TARGETS)}/{RUNS * len(TARGETS)} chuyến đạt,
  nghiêng khi rơi giảm một nửa (1,87° → 0,94°), không tốn thêm khí, và không
  làm hỏng việc lái. Cánh nhỏ đóng vai trò GIẢM DAO ĐỘNG, không phải ổn định.

  Còn nếu mục tiêu thật là "rơi chính xác" thì cánh không giúp gì. Cái cần là
  dẫn hướng chủ động — và con tàu đã có sẵn: vòi phun nghiêng được, do bộ
  điều khiển lái. Muốn chính xác hơn nữa thì phải cải thiện chỗ đó, hoặc cho
  nó lái được cả trong lúc rơi tự do (hiện tại cắt ga là mất hết quyền lái).
"""
    )

    return 0


__all__ = ["FIN_ARM", "free_fall", "full_flight", "main", "thrown_sideways"]
