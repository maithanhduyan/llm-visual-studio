"""
expert.py — Chuyên gia viết tay: bộ ra quyết định cấp cao.

Đây là "người thầy". Nó biết lái con tàu theo đúng một trình tự, và ta sẽ
dạy cho model nhỏ bắt chước nó.

Nhiệm vụ: phóng lên 10-15 m, rồi hạ xuống **mềm** (chạm đất dưới 2 m/s).

---

## Vì sao không hạ từ từ cho chắc

Thử tính xem. Con tàu có 90 g khí. Lúc lên tới 12 m đã dùng hết ~41 g, còn 49 g.

Giữ cho con tàu lơ lửng tốn khoảng **10 g khí mỗi giây**. Hạ từ từ 12 m với
vận tốc 1,5 m/s mất 8 giây — tốn 80 g. Không đủ.

Cách các tàu thật làm: **rơi tự do rồi hãm ở cuối.**

    rơi tự do 12 m  ->  tốn 0 g khí
    hãm ở 6 m cuối  ->  tốn ~20 g

Tổng 20 g thay vì 80 g. Vì khí chỉ tốn khi có lực đẩy, mà rơi tự do thì
không có lực đẩy.

Đổi lại, phải **canh đúng lúc bắt đầu hãm**. Hãm sớm quá thì tốn khí, hãm
muộn quá thì đâm xuống đất. Đó chính là quyết định khó nhất trong cả bài
toán — và là việc của bộ ra quyết định cấp cao.

---

## Sáu giai đoạn

    LEN    lên tới ~12 m
    GIU    giữ ở đỉnh, giảm vận tốc lên về 0
    ROI    cắt ga, rơi tự do            <- quyết định: khi nào?
    HAM    hãm lại, xuống chậm dần
    HA     hạ xuống chạm đất
    XONG   cắt ga
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .controller import Target
from .physics import State, Tank
from .vehicle import G, Vehicle

# ---------------------------------------------------------------------
# Bốn lệnh mà bộ ra quyết định được phép ra
# ---------------------------------------------------------------------

COMMANDS = ("LEN", "GIU", "ROI", "HAM")


@dataclass
class Decision:
    """Một quyết định cấp cao: ra lệnh gì, kèm con số gì."""

    command: str
    value: float = 0.0

    def to_text(self) -> str:
        """Viết lệnh ra chữ.

        Chú ý: `LEN` và `GIU` KHÔNG kèm con số. Lúc đầu chúng có kèm độ cao
        mục tiêu (`LEN 12.0`), nhưng model nhỏ chép lại con số rất kém —
        đo ra thì nó viết đúng LỆNH 100% mà chỉ chép đúng con số 85%.

        Mà độ cao mục tiêu thì hệ thống đã biết rồi (nó nằm trong đề bài),
        nên bắt model chép lại là bắt nó làm việc vô ích mà lại hay sai.
        Bỏ con số đi thì nó chỉ còn phải chọn một trong bốn lệnh.
        """
        if self.command in ("LEN", "GIU", "ROI"):
            return self.command
        return f"{self.command} {self.value:.1f}"

    def to_target(self, target_altitude: float = 12.0) -> Target:
        """Đổi thành mục tiêu cho PID. Độ cao mục tiêu lấy từ ĐỀ BÀI."""
        if self.command == "LEN":
            return Target(altitude=target_altitude, climb_rate=1.8)
        if self.command == "GIU":
            return Target(altitude=target_altitude, climb_rate=0.0)
        if self.command == "ROI":
            return Target(cut=True)
        if self.command == "HAM":
            # Con số ở đây là GIA TỐC HÃM mong muốn, không phải vận tốc.
            # Bộ điều khiển tự lo việc rơi nhanh hay chậm theo giản đồ.
            return Target(descent_profile=self.value if self.value > 0 else 7.0)
        raise ValueError(f"Lệnh lạ: {self.command!r}")

    @staticmethod
    def parse(text: str) -> "Decision":
        """Đọc lại lệnh từ chữ. Dùng khi model tự viết ra lệnh."""
        parts = text.strip().split()

        if not parts:
            return Decision("ROI")

        name = parts[0].upper()
        if name not in COMMANDS:
            name = "ROI"  # không hiểu thì cắt ga cho an toàn

        value = 0.0
        if len(parts) > 1:
            try:
                value = float(parts[1])
            except ValueError:
                value = 0.0

        return Decision(name, value)


# ---------------------------------------------------------------------
# Chuyên gia
# ---------------------------------------------------------------------


@dataclass
class Mission:
    """Đề bài: lên bao cao, chạm đất chậm hơn bao nhiêu."""

    target_altitude: float = 12.0
    max_landing_speed: float = 2.0  # m/s
    brake_accel: float = 11.0  # m/s² — gia tốc hãm dự kiến
    margin: float = 1.15  # hãm sớm hơn tính toán một chút cho chắc


class ExpertPilot:
    """Bộ ra quyết định viết tay. Không học gì cả, chỉ là một cái máy trạng thái."""

    def __init__(self, vehicle: Vehicle, mission: Mission | None = None) -> None:
        self.vehicle = vehicle
        self.mission = mission or Mission()
        self.phase = "LEN"
        self.peak_altitude = 0.0

    # ------------------------------------------------------------------

    def available_deceleration(self, tank: Tank) -> float:
        """Còn hãm được mạnh nhất bao nhiêu m/s², với áp suất hiện tại.

        Đây là chỗ quan trọng nhất của cả bộ ra quyết định: **lực hãm yếu
        dần khi khí cạn**. Nếu cứ tưởng lúc nào cũng hãm được như lúc đầu
        thì sẽ bắt đầu hãm quá muộn, và đâm xuống đất.
        """
        from .controller import thrust_at

        mass = self.vehicle.dry_mass + tank.air_mass
        thrust = thrust_at(self.vehicle, tank.pressure, 1.0)

        return max((thrust - mass * G) / mass, 0.0)

    def allowed_speed(self, altitude: float, tank: Tank | None = None) -> float:
        """Ở độ cao này thì được phép rơi nhanh tối đa bao nhiêu.

        Giản đồ hạ cánh: `v = sqrt(2 · a · h)`. Rơi nhanh hơn mức đó thì
        không còn đủ quãng đường để hãm nữa.

        `a` lấy theo khả năng THẬT của con tàu lúc này, không phải một con
        số cố định — vì khí càng cạn thì hãm càng yếu.
        """
        decel = self.mission.brake_accel

        if tank is not None:
            decel = min(decel, self.available_deceleration(tank) * 0.75)

        return math.sqrt(2.0 * max(decel, 0.5) * max(altitude, 0.0))

    def must_brake(self, state: State, tank: Tank) -> bool:
        """Đã tới lúc phải hãm chưa?"""
        return abs(state.vz) >= self.allowed_speed(state.z, tank) * 0.95

    def brake_deceleration(self, tank: Tank) -> float:
        """Hãm mạnh hay nhẹ, dựa trên khí còn lại.

        Càng ít khí thì càng phải hãm nhẹ — hãm mạnh tốn khí hơn, mà hãm
        nhẹ thì phải bắt đầu sớm hơn. Đây là một trong những chỗ mà bộ ra
        quyết định cấp cao thật sự phải "suy nghĩ".
        """
        available = self.available_deceleration(tank)
        wanted = 8.0 if tank.fill_fraction > 0.4 else 6.5

        return max(min(wanted, available * 0.7), 2.0)

    # ------------------------------------------------------------------

    def decide(self, state: State, tank: Tank) -> Decision:
        """Nhìn trạng thái, quyết định làm gì tiếp."""
        m = self.mission
        self.peak_altitude = max(self.peak_altitude, state.z)

        # --- Lên ---
        if self.phase == "LEN":
            if state.z >= m.target_altitude * 0.98:
                self.phase = "GIU"
            elif state.vz < -0.5:
                # Đang tụt mà chưa tới nơi -> hết khí rồi, đành chuyển sang hãm.
                self.phase = "HAM"
            else:
                return Decision("LEN")

        # --- Giữ ở đỉnh ---
        if self.phase == "GIU":
            if state.vz <= 0.15 and state.z >= m.target_altitude * 0.9:
                self.phase = "ROI"
            else:
                return Decision("GIU")

        # --- Rơi tự do cho tới lúc phải hãm ---
        if self.phase == "ROI":
            if self.must_brake(state, tank) or state.z < 1.0:
                self.phase = "HAM"
            else:
                return Decision("ROI")

        # --- Hãm theo giản đồ ---
        if self.phase == "HAM":
            decel = self.brake_deceleration(tank)

            # Chạm đất rồi thì thôi.
            if state.z <= 0.05 and abs(state.vz) <= m.max_landing_speed:
                self.phase = "XONG"
                return Decision("ROI")

            return Decision("HAM", decel)

        # --- Xong ---
        return Decision("ROI")


def state_text(state: State, tank: Tank, target_altitude: float = 12.0) -> str:
    """Trạng thái viết thành chữ, để model đọc.

    Năm con số, và con số ĐẦU TIÊN là đề bài:

        muctieu  — bay lên cao bao nhiêu (đây là thứ model từng bị thiếu)
        cao      — đang ở độ cao nào
        vz       — đang lên hay xuống nhanh bao nhiêu
        nghieng  — có đang bị lệch không
        ap       — còn bao nhiêu khí

    Chỗ này từng viết thiếu `muctieu`, và hậu quả rất dễ thấy: model bay
    lên ~13,5 m với MỌI đề bài. Nó không ngu — nó chỉ không được cho biết
    đề bài là gì. Chuyên gia biết (vì `ExpertPilot` giữ `mission`), model
    thì không, nên nó học được mỗi hành vi trung bình của mọi đề bài.
    """
    return (
        f"{target_altitude:.1f} {state.z:.1f} {state.vz:.1f} "
        f"{state.tilt_deg:.1f} {tank.pressure / 1e5:.1f}"
    )


__all__ = ["COMMANDS", "Decision", "ExpertPilot", "Mission", "state_text", "math"]
