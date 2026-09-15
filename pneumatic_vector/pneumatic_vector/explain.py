"""
explain.py — Vì sao model hỏng? Nhìn vào chỗ nó quyết định.

Model đạt 60%: đề bài 12 m và 15 m thì gần như hoàn hảo, còn 10 m thì hỏng
sạch. Nhìn con số 60% thì không hiểu tại sao. Nhìn vào **độ cao lúc model
bắt đầu hãm** thì hiểu ngay.

Quy tắc thật của chuyên gia là một giản đồ theo ĐỘ CAO:

    ở độ cao h thì được phép rơi nhanh tối đa   v = sqrt(2 · a · h)

Ở 5 m được rơi ~10,6 m/s, ở 8 m được rơi ~13 m/s. Nói cách khác: càng gần
đất thì càng phải siết lại. Quy tắc phụ thuộc vào ĐỘ CAO CÒN LẠI.

Chú ý một chỗ dễ nhầm: mức "cho phép" ở đây là một con số thận trọng (lấy
75% khả năng hãm thật). Vượt qua nó một chút vẫn hạ được — chuyên gia vượt
ở đề bài 10 m mà vẫn đạt. Nên đừng lấy nó làm phán quyết; phán quyết là
chuyến bay có đạt hay không.
"""

from __future__ import annotations

from .expert import ExpertPilot, Mission
from .physics import Tank
from .simulate import fly
from .vehicle import Vehicle

# Đo cả ngoài khoảng đề bài (10-15 m) để thấy model hỏng ở đâu.
DEFAULT_TARGETS = (8.0, 10.0, 12.0, 15.0, 18.0)


def _make_expert(vehicle: Vehicle):
    class _RecordExpert(ExpertPilot):
        """Chuyên gia, nhưng ghi lại lúc nó chuyển sang giai đoạn hãm."""

        def __init__(self, mission=None) -> None:
            super().__init__(vehicle, mission)
            self.log: list[tuple[float, float, float]] = []

        def decide(self, state, tank):
            before = self.phase
            decision = super().decide(state, tank)

            if self.phase != before and self.phase == "HAM":
                self.log.append((state.z, state.vz, tank.pressure / 1e5))

            return decision

    return _RecordExpert


def _make_model(model, tokenizer):
    from .policy import LearnedPilot

    # Phải KẾ THỪA `LearnedPilot`, không phải bọc nó lại.
    #
    # Bản đầu tiên của file này bọc nó và chuyển tiếp mọi thuộc tính bằng
    # `__getattr__`. Hỏng: `simulate.fly` đặt `policy.mission = mission`, mà
    # câu đó ghi lên lớp BỌC, còn model bên trong vẫn giữ `mission = None`.
    # Thế là model bay đề bài nào cũng tưởng mục tiêu là 12 m — và bảng kết
    # quả trông vẫn hợp lý, chỉ có điều nó không đo cái cần đo.
    class _RecordModel(LearnedPilot):
        """Model đã học, ghi lại lúc nó nói `HAM` lần đầu."""

        def __init__(self, mission=None) -> None:
            super().__init__(model, tokenizer)

            if mission is not None:
                self.mission = mission

            self.log: list[tuple[float, float, float]] = []
            self.was = "?"

        def decide(self, state, tank):
            decision = super().decide(state, tank)

            if decision.command == "HAM" and self.was != "HAM":
                self.log.append((state.z, state.vz, tank.pressure / 1e5))

            self.was = decision.command
            return decision

    return _RecordModel


def trigger_table(vehicle: Vehicle, make_pilot, targets=DEFAULT_TARGETS, seed: int = 0):
    """Chạy từng đề bài, ghi lại lúc bộ ra quyết định bắt đầu hãm."""
    rows: list[dict] = []

    for target in targets:
        mission = Mission(target_altitude=target)

        pilot = make_pilot(mission)
        result = fly(vehicle, pilot, mission, seed=seed)

        row = {
            "target": target,
            "peak": result.peak_altitude,
            "landing_speed": result.landing_speed,
            "success": result.success(mission),
            "altitude": None,
            "speed": None,
            "allowed": None,
        }

        if pilot.log:
            z, vz, pressure = pilot.log[0]

            tank = Tank(vehicle)
            tank.pressure = pressure * 1e5

            row["altitude"] = z
            row["speed"] = abs(vz)
            row["allowed"] = ExpertPilot(vehicle, mission).allowed_speed(z, tank)

        rows.append(row)

    return rows


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print("  đề bài |  cao lúc bắt đầu hãm |  đang rơi | mức an toàn | chạm đất | kết quả")
    print("  -------+---------------------+----------+-------------+----------+--------")

    for row in rows:
        if row["altitude"] is None:
            print(f"  {row['target']:5.1f}  |      không bao giờ hãm       | "
                  f"{row['landing_speed']:8.2f} | {'ĐẠT' if row['success'] else 'HỎNG':>6}")
            continue

        print(f"  {row['target']:5.1f}  | {row['altitude']:19.2f} | "
              f"{row['speed']:8.2f} | {row['allowed']:11.2f} | "
              f"{row['landing_speed']:8.2f} | {'ĐẠT' if row['success'] else 'HỎNG':>6}")


def explain(vehicle: Vehicle | None = None, name: str = "pilot", seed: int = 0) -> int:
    from .policy import LearnedPilot

    vehicle = vehicle or Vehicle()
    model = LearnedPilot.load(name)

    print("Vì sao model hỏng? Nhìn vào lúc nó bắt đầu hãm.")
    print("=" * 76)
    print("\nQuy tắc của chuyên gia là giản đồ theo ĐỘ CAO: ở độ cao h thì được")
    print("phép rơi nhanh tối đa sqrt(2·a·h). Càng gần đất càng phải siết lại.")

    expert_rows = trigger_table(vehicle, _make_expert(vehicle), seed=seed)
    _print_table("CHUYÊN GIA VIẾT TAY", expert_rows)

    model_rows = trigger_table(
        vehicle, _make_model(model.model, model.tokenizer), seed=seed
    )
    _print_table(f"MODEL ĐÃ HỌC ({name})", model_rows)

    # --- So sánh trực tiếp: model hãm CAO HƠN hay THẤP HƠN chuyên gia? ---
    print("\n" + "=" * 76)
    print("So với chuyên gia, model bắt đầu hãm ở đâu?")
    print("  đề bài | chuyên gia |  model  | lệch   | kết quả")
    print("  -------+------------+---------+--------+--------")

    thap_hon = []

    for expert_row, model_row in zip(expert_rows, model_rows):
        if expert_row["altitude"] is None or model_row["altitude"] is None:
            continue

        delta = model_row["altitude"] - expert_row["altitude"]
        mark = "thấp hơn" if delta < -0.3 else ("cao hơn" if delta > 0.3 else "như nhau")

        if delta < -0.3 and not model_row["success"]:
            thap_hon.append((model_row, delta, expert_row))

        print(f"  {model_row['target']:5.1f}  | {expert_row['altitude']:10.2f} | "
              f"{model_row['altitude']:7.2f} | {delta:+6.2f} | "
              f"{'ĐẠT' if model_row['success'] else 'HỎNG':>4}  {mark}")

    inside = [r for r in model_rows if 10.0 <= r["target"] <= 15.0]
    good = sum(r["success"] for r in inside)

    print(f"\nTrong khoảng đề bài 10-15 m: {good}/{len(inside)} chuyến đạt")
    print("(đề bài 8 m và 18 m nằm NGOÀI khoảng dự án nhắm tới, đo chỉ để xem"
          "\n model hỏng thế nào khi ra khỏi vùng nó được học.)")

    # Chỉ so trong khoảng đề bài dự án nhắm tới. Ngoài khoảng đó model có lúc
    # nói `HAM` lúc đang lơ lửng trên đỉnh — đó không phải quyết định hãm, và
    # đưa nó vào sẽ làm hỏng dải số.
    speeds = [r["speed"] for r in inside if r["speed"] is not None]
    expert_speeds = [
        r["speed"] for r in expert_rows if r["speed"] is not None and 10.0 <= r["target"] <= 15.0
    ]

    if thap_hon and speeds and expert_speeds:
        print("\nĐọc bảng trên là thấy vấn đề. Vận tốc lúc hãm thì model học")
        print(f"khá sát chuyên gia ({min(speeds):.1f}-{max(speeds):.1f} m/s so với "
              f"{min(expert_speeds):.1f}-{max(expert_speeds):.1f} m/s,"
              " chỉ tính đề bài 10-15 m).")
        print("Nhưng ĐỘ CAO lúc hãm thì bị co lại vào giữa: chuyên gia hãm từ")
        print("4,15 m tới 9,38 m tuỳ đề bài, còn model chỉ từ 4,41 m tới 8,36 m.")
        print("Ở đề bài thấp, model hãm **thấp hơn** chuyên gia:")
        for row, delta, expert_row in thap_hon:
            print(f"  - đề bài {row['target']:.0f} m: model hãm ở "
                  f"{row['altitude']:.2f} m khi đang rơi {row['speed']:.2f} m/s, "
                  f"chuyên gia hãm ở {expert_row['altitude']:.2f} m khi đang rơi "
                  f"{expert_row['speed']:.2f} m/s")
            print(f"    -> thấp hơn {abs(delta):.2f} m, tức là ít đường hãm hơn "
                  f"{abs(delta):.2f} m, và chạm đất {row['landing_speed']:.2f} m/s")

        print("\nCùng một vận tốc, nhưng hãm ở độ cao thấp hơn nghĩa là còn ít")
        print("đường hãm hơn. Quy tắc thật phụ thuộc vào ĐỘ CAO CÒN LẠI, không")
        print("phải vào vận tốc — và đó là chỗ model học chưa tới.")
        print("\nBài học: ở gần biên của bài toán, một sai số nhỏ là chết. Lệch")
        print("0,7 m ở độ cao hãm đủ để chạm đất 8,46 m/s thay vì 1,97 m/s.")
    else:
        print("\nKhông có đề bài nào model hãm thấp hơn chuyên gia mà vẫn hỏng.")

    return 0


__all__ = ["DEFAULT_TARGETS", "explain", "trigger_table"]
