"""
physics.py — Mô phỏng chuyến bay.

Con tàu có 8 con số trạng thái:

    vị trí    x, y, z          (z là độ cao)
    vận tốc   vx, vy, vz
    nghiêng   theta (quanh trục y), phi (quanh trục x)
    quay      omega_theta, omega_phi

Vì sao chỉ có hai góc nghiêng mà không phải ba: con tàu đối xứng tròn, nên
góc quay quanh trục dọc (góc "vặn") không ảnh hưởng gì tới việc bay. Bỏ nó
đi cho dễ hiểu.

---

## Mô-men xoay đến từ đâu

Đây là chỗ dễ hiểu sai nhất. Lực đẩy tác dụng ở vòi phun, nằm **dưới** trọng
tâm một đoạn L. Mô-men xoay là tích có hướng của cánh tay đòn và lực:

    tau = r × F        với  r = -L · (trục thân tàu),  F = T · (hướng phụt)

Tính ra thì hai cái nghiêng **triệt tiêu nhau**, còn lại:

    tau = -L · T · δ        (δ là góc nghiêng của vòi phun)

Nghĩa là: **mô-men xoay chỉ phụ thuộc góc nghiêng VÒI PHUN**, không phụ thuộc
con tàu đang nghiêng bao nhiêu. Đó là toàn bộ nguyên lý lái vector: muốn xoay
thì nghiêng vòi, chứ không phải nghiêng cả con tàu.

Hệ quả quan trọng: lực đẩy càng yếu thì lái càng yếu. Lúc gần cạn khí, van mở
hết cũng không xoay được con tàu nữa.

---

## Vì sao con tàu tự ngã

Nếu chỉ có hai điều trên thì con tàu trung tính — nghiêng bao nhiêu thì giữ
nguyên bấy nhiêu. Nhưng thực tế nó **tự ngã thêm**, vì khí động học:

    tau_khi_dong = +K · q · θ        (q là áp suất động, θ là góc nghiêng)

Dấu **cộng** mới là chỗ chết người: nghiêng càng nhiều thì mô-men làm nó
nghiêng thêm càng mạnh. Đây là con lắc ngược. Tâm khí động nằm TRÊN trọng
tâm — giống như dựng cây chổi trên lòng bàn tay vậy.

Nên bộ điều khiển phải **liên tục** đẩy vòi phun ngược lại. Tắt điều khiển
một giây là con tàu lộn nhào.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .vehicle import G, P_ATM, RHO_AIR, Tank, Vehicle


@dataclass
class State:
    """Trạng thái con tàu tại một thời điểm."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    theta: float = 0.0  # nghiêng quanh trục y (rad)
    phi: float = 0.0  # nghiêng quanh trục x (rad)

    omega_theta: float = 0.0
    omega_phi: float = 0.0

    t: float = 0.0

    @property
    def tilt(self) -> float:
        """Độ nghiêng tổng hợp, radian. 0 = thẳng đứng."""
        return math.hypot(self.theta, self.phi)

    @property
    def tilt_deg(self) -> float:
        return math.degrees(self.tilt)

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def horizontal_speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def copy(self) -> "State":
        return State(**{name: getattr(self, name) for name in self.__dataclass_fields__})


@dataclass
class Command:
    """Lệnh từ bộ điều khiển xuống con tàu."""

    throttle: float = 0.0  # 0..1 — van mở bao nhiêu
    gimbal_theta: float = 0.0  # rad — vòi nghiêng quanh trục y
    gimbal_phi: float = 0.0  # rad — vòi nghiêng quanh trục x

    def clipped(self, vehicle: Vehicle) -> "Command":
        limit = vehicle.gimbal_max
        return Command(
            throttle=max(0.0, min(1.0, self.throttle)),
            gimbal_theta=max(-limit, min(limit, self.gimbal_theta)),
            gimbal_phi=max(-limit, min(limit, self.gimbal_phi)),
        )


@dataclass
class Step:
    """Mọi thứ xảy ra trong một bước, để ghi log và vẽ hình."""

    t: float
    state: State
    command: Command
    thrust: float
    mass: float
    pressure: float


def thrust_direction(theta: float, phi: float) -> tuple[float, float, float]:
    """Hướng lực đẩy trong hệ toạ độ thế giới, khi thân tàu nghiêng (θ, φ).

    Đây là phép quay vector "lên trên" (0,0,1) quanh trục y rồi quanh trục x.
    """
    return (
        math.sin(theta),
        -math.sin(phi) * math.cos(theta),
        math.cos(phi) * math.cos(theta),
    )


def step(
    state: State,
    vehicle: Vehicle,
    tank: Tank,
    command: Command,
    dt: float | None = None,
) -> Step:
    """Tiến con tàu thêm `dt` giây."""
    dt = dt or vehicle.dt
    command = command.clipped(vehicle)

    mass = vehicle.dry_mass + tank.air_mass

    # 1. Đốt khí: lực đẩy sinh ra, áp suất tụt xuống.
    thrust = tank.burn(command.throttle, dt)

    # 2. Hướng phụt = hướng thân tàu, xoay thêm một góc bằng góc nghiêng vòi.
    ux, uy, uz = thrust_direction(
        state.theta + command.gimbal_theta,
        state.phi + command.gimbal_phi,
    )

    # 3. Các lực tác dụng.
    fx = thrust * ux
    fy = thrust * uy
    fz = thrust * uz - mass * G

    # Lực cản không khí, ngược chiều chuyển động.
    if state.speed > 1e-6:
        drag = 0.5 * RHO_AIR * vehicle.drag_coefficient * vehicle.drag_area * state.speed
        fx -= drag * state.vx
        fy -= drag * state.vy
        fz -= drag * state.vz

    ax, ay, az = fx / mass, fy / mass, fz / mass

    # 4. Mô-men xoay.
    #    - Lái vector: chỉ phụ thuộc góc nghiêng vòi phun.
    #    - Khí động: `vehicle.stability` gộp hai thứ đối nghịch nhau:
    #         + tâm khí động trên trọng tâm  -> ĐỒNG DẤU với góc nghiêng
    #           (con tàu tự ngã thêm, như dựng chổi trên tay)
    #         - cánh đuôi                    -> NGƯỢC DẤU
    #           (tự dựng lại, như đuôi mũi tên)
    #      Số dương là mất ổn định, số âm là tự dựng lại.
    #    - Cản quay: làm chậm tốc độ xoay.
    dynamic_pressure = 0.5 * RHO_AIR * state.speed**2
    aero = vehicle.stability * dynamic_pressure

    tau_theta = -vehicle.nozzle_offset * thrust * command.gimbal_theta
    tau_theta += aero * state.theta
    tau_theta -= vehicle.spin_damping * state.omega_theta

    tau_phi = -vehicle.nozzle_offset * thrust * command.gimbal_phi
    tau_phi += aero * state.phi
    tau_phi -= vehicle.spin_damping * state.omega_phi

    alpha_theta = tau_theta / vehicle.inertia
    alpha_phi = tau_phi / vehicle.inertia

    # 5. Cộng dồn (Euler nửa ẩn — ổn định hơn Euler thường).
    nxt = state.copy()
    nxt.t = state.t + dt

    nxt.vx = state.vx + ax * dt
    nxt.vy = state.vy + ay * dt
    nxt.vz = state.vz + az * dt

    nxt.x = state.x + nxt.vx * dt
    nxt.y = state.y + nxt.vy * dt
    nxt.z = state.z + nxt.vz * dt

    nxt.omega_theta = state.omega_theta + alpha_theta * dt
    nxt.omega_phi = state.omega_phi + alpha_phi * dt

    nxt.theta = state.theta + nxt.omega_theta * dt
    nxt.phi = state.phi + nxt.omega_phi * dt

    # 6. Chạm đất thì dừng lại, không xuyên qua mặt đất.
    if nxt.z <= 0.0:
        nxt.z = 0.0
        nxt.vz = max(nxt.vz, 0.0)

    return Step(
        t=nxt.t,
        state=nxt,
        command=command,
        thrust=thrust,
        mass=vehicle.dry_mass + tank.air_mass,
        pressure=tank.pressure,
    )


def hovers_on_empty(vehicle: Vehicle) -> float:
    """Còn bao nhiêu giây giữ được con tàu lơ lửng, khi đã cạn khí.

    Con số này cho biết có đủ khí để hạ cánh có điều khiển hay không.
    """
    mass = vehicle.dry_mass + vehicle.air_mass
    thrust_needed = mass * G
    flow = mass_flow_at(vehicle, thrust_needed)

    if flow <= 0:
        return 0.0

    return vehicle.air_mass / flow


def mass_flow_at(vehicle: Vehicle, thrust: float) -> float:
    """Cần bao nhiêu kg khí mỗi giây để có lực đẩy này (lúc đầy khí)."""
    from .vehicle import exhaust_velocity

    velocity = exhaust_velocity(vehicle, vehicle.initial_pressure)

    if velocity <= 0:
        return 0.0

    return thrust / (velocity * vehicle.thrust_coefficient)


__all__ = [
    "Command",
    "G",
    "P_ATM",
    "State",
    "Step",
    "Tank",
    "Vehicle",
    "hovers_on_empty",
    "mass_flow_at",
    "step",
    "thrust_direction",
    "field",
]
