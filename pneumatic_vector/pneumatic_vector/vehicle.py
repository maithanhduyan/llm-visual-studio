"""
vehicle.py — Con tàu: một ống hình trụ chứa khí nén.

    ┌─────────────────────┐  ← đỉnh
    │                     │
    │   khí nén 8 bar     │  ống dài 1,2 m, đường kính 10 cm
    │                     │
    │                     │
    ├─────────┬───────────┤  ← trọng tâm ở giữa
    │         │           │
    │         ▼           │  ← vòi phun, nghiêng được ±12°
    └─────────┴───────────┘
              ↓ khí phụt ra

Ba thứ khiến việc điều khiển khó:

1. **Khí nén cạn dần.** Càng bay thì áp suất càng giảm, lực đẩy càng yếu.
   Không phải "ga to là xong" — ga to lúc đầu thì hết khí lúc cuối.

2. **Con tàu là con lắc ngược.** Trọng tâm ở trên, lực đẩy ở dưới. Chỉ cần
   nghiêng một chút là nó tự ngã thêm. Giống giữ cây chổi dựng trên tay.

3. **Lực đẩy có hai việc cùng lúc.** Vừa nâng con tàu lên, vừa lái nó. Cho
   nên nghiêng vòi phun vừa tạo lực ngang, vừa tạo mô-men xoay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- Hằng số vật lý ---------------------------------------------------
G = 9.81  # m/s²
R_AIR = 287.0  # J/(kg·K)
GAMMA = 1.4  # tỉ số nhiệt dung của không khí
P_ATM = 101325.0  # Pa
RHO_AIR = 1.225  # kg/m³ ở mặt đất
T_GAS = 293.0  # K


@dataclass
class Vehicle:
    """Thông số con tàu. Đổi mấy số này là đổi cả bài toán."""

    # --- Ống chứa khí ---
    length: float = 1.2  # m
    diameter: float = 0.10  # m
    pressure_bar: float = 10.0  # áp suất nạp ban đầu

    # --- Khối lượng ---
    dry_mass: float = 0.60  # kg (vỏ, van, đồ điện)

    # --- Vòi phun ---
    throat_diameter: float = 0.012  # m
    discharge_coefficient: float = 0.85  # van thật không mở hoàn hảo
    thrust_coefficient: float = 1.25  # hệ số lực đẩy của loa phun
    gimbal_max_deg: float = 12.0  # vòi nghiêng được bao nhiêu độ
    nozzle_offset: float = 0.62  # m — vòi nằm dưới trọng tâm bao xa

    # --- Hình dạng khí động ---
    drag_coefficient: float = 0.55
    cross_section: float = 0.0079  # m² (đường kính 10 cm)

    # Tâm khí động nằm TRÊN trọng tâm bao xa (theo "sải" khí động).
    # Số dương = mất ổn định = con tàu tự ngã thêm. Đây là thứ khiến việc
    # điều khiển trở nên cần thiết; để bằng 0 thì con tàu trung tính và
    # tắt điều khiển nó vẫn đứng yên.
    aero_torque_gain: float = 1.4

    # --- Cánh đuôi (tùy chọn) -------------------------------------------
    #
    # Hai cánh nhỏ hai bên, gắn ở PHÍA ĐUÔI. Chúng làm đúng một việc: kéo
    # tâm khí động xuống DƯỚI trọng tâm. Khi tâm khí động ở dưới trọng tâm
    # thì mô-men khí động đổi dấu — thành mô-men TỰ DỰNG LẠI, giống đuôi
    # mũi tên hay lông cầu lông.
    #
    # fin_area:   diện tích MỖI cánh, m² (không phải tổng hai cánh)
    # fin_offset: cánh nằm sau trọng tâm bao xa, m
    #
    # Mỗi cánh ở góc nghiêng θ tạo một lực ngang cỡ `q·A·θ`, cách trọng tâm L.
    # Hai cánh cùng chiều nên mô-men dựng lại là `2·q·A·L·θ`.
    #
    # Điều kiện để con tàu TĨNH ỔN ĐỊNH (không cần điều khiển vẫn đứng thẳng):
    #
    #     2 · fin_area · fin_offset  >  aero_torque_gain · cross_section
    #
    # Vế phải là 1,4 × 0,0079 = 0,01106 m³. Với cánh cách trọng tâm 0,5 m,
    # cần fin_area > 0,01106 / (2 × 0,5) = 0,0111 m² = 111 cm² MỖI cánh.
    #
    # Đổi lại phải trả giá: cánh làm tăng lực cản, và khi con tàu muốn LÁI
    # thì bộ điều khiển phải thắng cả cánh nữa.
    fin_area: float = 0.0  # m², diện tích MỖI cánh
    fin_offset: float = 0.0  # m, sau trọng tâm

    # Cánh mỏng đặt dọc theo dòng chảy nên cản ít hơn diện tích của nó nhiều.
    fin_drag_factor: float = 0.10

    # Cản quay: làm chậm tốc độ xoay, giúp bộ điều khiển đỡ vất vả.
    spin_damping: float = 0.006

    # --- Quán tính quay ---
    inertia: float = 0.035  # kg·m² — mô-men quán tính quanh trục ngang

    # --- Mô phỏng ---
    dt: float = 0.001  # giây mỗi bước vật lý

    @property
    def volume(self) -> float:
        """Thể tích ống, m³."""
        return math.pi * (self.diameter / 2) ** 2 * self.length

    @property
    def initial_pressure(self) -> float:
        return self.pressure_bar * 1e5

    @property
    def air_mass(self) -> float:
        """Khối lượng khí nén lúc nạp đầy, kg."""
        return self.initial_pressure * self.volume / (R_AIR * T_GAS)

    @property
    def throat_area(self) -> float:
        return math.pi * (self.throat_diameter / 2) ** 2

    @property
    def gimbal_max(self) -> float:
        """Góc nghiêng vòi tối đa, radian."""
        return math.radians(self.gimbal_max_deg)

    @property
    def total_mass(self) -> float:
        """Khối lượng lúc nạp đầy, kg."""
        return self.dry_mass + self.air_mass

    # --- Cánh đuôi ------------------------------------------------------

    @property
    def fin_moment(self) -> float:
        """Cánh kéo tâm khí động xuống được bao nhiêu, m³.

        Mô-men dựng lại của cánh: `2 · q · A · L · θ`. Hệ số đứng trước `q·θ`
        chính là `2 · A · L`, cùng đơn vị với `aero_torque_gain · cross_section`.
        """
        return 2.0 * self.fin_area * self.fin_offset

    @property
    def stability(self) -> float:
        """Số dương = MẤT ổn định, số âm = TỰ DỰNG LẠI. Đơn vị m³.

        Đây là hiệu giữa "tâm khí động đẩy ngã thêm" và "cánh kéo lại".
        """
        return self.aero_torque_gain * self.cross_section - self.fin_moment

    @property
    def is_stable(self) -> bool:
        """Không cần điều khiển thì con tàu có tự đứng thẳng lại không."""
        return self.stability < 0.0

    @property
    def drag_area(self) -> float:
        """Diện tích cản hiệu dụng, m². Cánh làm tăng lên."""
        return self.cross_section + self.fin_drag_factor * 2.0 * self.fin_area

    def fin_area_for_neutral(self, offset: float | None = None) -> float:
        """Diện tích mỗi cánh cần có để con tàu vừa đủ trung tính, m².

        `offset` để hỏi "nếu gắn cánh cách trọng tâm chừng này thì cần bao
        nhiêu" — hữu ích vì lúc đang thiết kế thì chưa gắn cánh, nên
        `self.fin_offset` còn bằng 0.
        """
        arm = self.fin_offset if offset is None else offset

        if arm <= 0:
            return float("inf")

        return self.aero_torque_gain * self.cross_section / (2.0 * arm)

    def check(self) -> list[str]:
        """Vài lời nhắc nếu thông số trông không hợp lý."""
        notes = []

        if self.nozzle_offset <= 0:
            notes.append("Vòi phun phải nằm DƯỚI trọng tâm, nếu không thì lái không được.")

        if self.air_mass / self.total_mass < 0.05:
            notes.append("Khí nén quá ít so với vỏ — bay không cao được.")

        thrust = maximum_thrust(self)
        if thrust < self.total_mass * G * 1.5:
            notes.append(
                f"Lực đẩy tối đa {thrust:.1f} N chỉ hơn trọng lượng "
                f"{self.total_mass * G:.1f} N có {thrust / (self.total_mass * G):.1f} lần. "
                "Cần ít nhất 1,5 lần mới bay lên được."
            )

        return notes


def mass_flow(vehicle: Vehicle, pressure: float, throttle: float) -> float:
    """Khí phụt ra bao nhiêu kg mỗi giây.

    Khi áp suất trong ống lớn hơn khoảng 1,9 lần áp suất ngoài thì dòng chảy
    đạt tốc độ âm thanh ở cổ vòi — gọi là "tắt nghẽn". Lúc đó lưu lượng chỉ
    phụ thuộc áp suất trong ống, không phụ thuộc áp suất ngoài nữa.
    """
    throat = vehicle.throat_area * max(0.0, min(1.0, throttle))

    if throat <= 0 or pressure <= P_ATM:
        return 0.0

    if pressure > 1.9 * P_ATM:
        choke = math.sqrt(GAMMA * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (GAMMA - 1)))
        return vehicle.discharge_coefficient * throat * pressure * choke / math.sqrt(R_AIR * T_GAS)

    # Áp suất đã yếu: dòng chảy dưới tốc độ âm thanh.
    #
    # Công thức ở đây dùng tỉ số `áp suất NGOÀI / áp suất TRONG`, tức là một
    # số NHỎ HƠN 1. Từng viết ngược lại (trong/ngoài), thành ra số hạng trong
    # căn luôn âm, lưu lượng luôn bằng 0 — động cơ tắt ngóm ngay khi áp suất
    # xuống dưới 1,9 bar. Mà 1,9 bar lại đúng là lúc con tàu cần hãm nhất.
    ratio = P_ATM / pressure
    term = (2 * GAMMA / (GAMMA - 1)) * (ratio ** (2 / GAMMA) - ratio ** ((GAMMA + 1) / GAMMA))

    if term <= 0:
        return 0.0

    return vehicle.discharge_coefficient * throat * pressure * math.sqrt(term) / math.sqrt(R_AIR * T_GAS)


def exhaust_velocity(vehicle: Vehicle, pressure: float) -> float:
    """Khí phụt ra nhanh bao nhiêu, m/s."""
    if pressure <= P_ATM:
        return 0.0

    ratio = P_ATM / pressure
    term = (2 * GAMMA / (GAMMA - 1)) * (1 - ratio ** ((GAMMA - 1) / GAMMA))

    return math.sqrt(max(term, 0.0) * R_AIR * T_GAS)


def maximum_thrust(vehicle: Vehicle) -> float:
    """Lực đẩy lúc mới nạp, van mở hết, N."""
    pressure = vehicle.initial_pressure
    flow = mass_flow(vehicle, pressure, 1.0)
    velocity = exhaust_velocity(vehicle, pressure)

    return flow * velocity * vehicle.thrust_coefficient


@dataclass
class Tank:
    """Trạng thái khí trong ống. Cạn dần khi phụt ra."""

    vehicle: Vehicle
    pressure: float = field(init=False)

    def __post_init__(self) -> None:
        self.pressure = self.vehicle.initial_pressure

    @property
    def air_mass(self) -> float:
        return self.pressure * self.vehicle.volume / (R_AIR * T_GAS)

    @property
    def fill_fraction(self) -> float:
        return self.pressure / self.vehicle.initial_pressure

    def burn(self, throttle: float, dt: float) -> float:
        """Phụt khí trong `dt` giây. Trả về lực đẩy sinh ra (N)."""
        mass = self.air_mass

        if mass <= 1e-6:
            self.pressure = P_ATM
            return 0.0

        flow = mass_flow(self.vehicle, self.pressure, throttle)

        # Van ĐÓNG thì không phụt gì cả — và áp suất phải giữ nguyên.
        #
        # Chỗ này từng viết sai: gộp chung `flow <= 0` vào nhánh xả hết khí,
        # nên mỗi lần cắt ga là áp suất tụt thẳng xuống 1 bar. Con tàu mất
        # sạch khí chỉ vì ngừng đốt.
        if flow <= 0:
            return 0.0

        # Lấy khối lượng khí tiêu tốn, nhưng không lấy quá số đang có.
        before = self.pressure
        used = min(flow * dt, mass)
        self.pressure = max(before - used * R_AIR * T_GAS / self.vehicle.volume, P_ATM)

        # Lực đẩy suy ra từ lưu lượng THẬT SỰ dùng, không phải lưu lượng mong muốn.
        real_flow = used / dt

        # Vận tốc phụt lấy ở áp suất TRUNG BÌNH trong bước, không phải áp suất
        # cuối bước. Với bước 1 ms thì hai số gần như bằng nhau. Nhưng nếu ai
        # đó gọi `burn` với bước dài — xả hết khí trong một lần — thì áp suất
        # cuối bước đúng bằng 1 bar, mà ở 1 bar thì không còn lực đẩy nào cả.
        # Kết quả là đốt sạch khí mà lực đẩy lại bằng 0, một cách vô lý.
        velocity = exhaust_velocity(self.vehicle, 0.5 * (before + self.pressure))

        return real_flow * velocity * self.vehicle.thrust_coefficient
