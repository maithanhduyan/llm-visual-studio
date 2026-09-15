"""
controller.py — Bộ điều khiển PID, chạy 200 lần mỗi giây.

Đây là vòng điều khiển **bên trong**. Nó không biết gì về "nhiệm vụ" cả —
nó chỉ nhận ba mục tiêu và cố đạt được:

    - nghiêng bao nhiêu độ
    - cao bao nhiêu mét
    - hạ xuống với vận tốc bao nhiêu

Ai đặt ra ba mục tiêu đó? Bộ ra quyết định cấp cao — chính là LLM. Đó là
toàn bộ sự phân công: LLM nghĩ chậm mà quyết định lớn, PID nghĩ nhanh mà
quyết định nhỏ.

---

## Vòng lái: khó hơn PID thường

Công thức PID quen thuộc:

    δ = Kp · sai số + Kd · đạo hàm sai số

Nhưng mô-men xoay ở đây là `tau = -L · T · δ` — **tỉ lệ với lực đẩy T**.
Lực đẩy lại thay đổi liên tục vì khí cạn dần.

Nên nếu dùng Kp cố định thì lúc đầu (T lớn) con tàu giật cục, lúc gần cạn
(T nhỏ) thì lái không nổi. Phải **chia cho T**:

    δ = (Kp · sai số + Kd · tốc độ xoay) / (L · T)

Đây gọi là "bù lực đẩy". Nhìn thì nhỏ, nhưng không có nó thì bộ điều khiển
chỉ chạy được trong khoảng nửa đầu chuyến bay.

---

## Ba tầng, không phải một

    vị trí ngang  ->  góc nghiêng mong muốn     (muốn đi sang phải thì nghiêng phải)
    độ cao        ->  ga                        (muốn lên thì mở van)
    góc nghiêng   ->  góc vòi phun              (muốn xoay thì nghiêng vòi)

Ba tầng này chạy **lồng vào nhau**: muốn sang phải thì nghiêng, muốn nghiêng
thì lái vòi. Mỗi tầng là một bài toán nhỏ, dễ hiểu. Gộp lại thành một bài
toán lớn thì rất khó.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .physics import Command, State, Tank, thrust_direction
from .vehicle import G, P_ATM, Vehicle, exhaust_velocity, mass_flow


def thrust_at(vehicle: Vehicle, pressure: float, throttle: float = 1.0) -> float:
    """Lực đẩy sẽ có nếu mở van ở mức `throttle`, với áp suất hiện tại."""
    flow = mass_flow(vehicle, pressure, throttle)
    return flow * exhaust_velocity(vehicle, pressure) * vehicle.thrust_coefficient


@dataclass
class Target:
    """Ba mục tiêu mà bộ điều khiển cấp cao đặt ra."""

    altitude: float = 12.0  # m — muốn ở độ cao này
    climb_rate: float = 0.0  # m/s — muốn lên/xuống với tốc độ này
    level: bool = True  # muốn thăng bằng (không bay ngang)
    cut: bool = False  # cắt ga hoàn toàn (rơi tự do)

    # Nếu > 0 thì không dùng `climb_rate` nữa, mà đi theo GIẢN ĐỒ HẠ CÁNH:
    # vận tốc cho phép ở độ cao h là `-sqrt(2 · a · h)`. Đây là số `a` đó.
    #
    # Vì sao cần giản đồ thay vì "hạ 1,5 m/s": nếu chỉ xin một vận tốc cố
    # định, bộ điều khiển sẽ hãm rất nhát khi đang rơi nhanh, và con tàu
    # đâm xuống đất. Giản đồ cho phép rơi nhanh khi còn cao, và tự động
    # siết lại khi gần đất — đúng cách các tàu thật hạ cánh.
    descent_profile: float = 0.0


@dataclass
class Gains:
    """Các hệ số của bộ điều khiển. Đổi mấy số này là đổi "tính cách" con tàu."""

    # Vòng lái (nghiêng -> vòi phun)
    att_kp: float = 9.0  # càng to càng dựng thẳng quyết liệt
    att_kd: float = 5.4  # càng to càng ít lắc

    # Vòng độ cao (cao -> ga)
    alt_kp: float = 0.55
    alt_kd: float = 1.35

    # Vòng hạ cánh: bám giản đồ vận tốc chặt hơn nhiều, vì rơi thì nhanh
    # mà đất thì gần.
    land_kd: float = 3.2

    # Vòng ngang (lệch -> nghiêng)
    lat_kp: float = 0.020
    lat_kd: float = 0.090

    # Giới hạn
    max_tilt: float = math.radians(9.0)  # không nghiêng quá mức này để bay ngang
    min_throttle_for_steering: float = 4.0  # N — dưới mức này thì lái vô hiệu


class CascadedController:
    """Bộ điều khiển ba tầng, chạy mỗi bước vật lý."""

    def __init__(self, vehicle: Vehicle, gains: Gains | None = None) -> None:
        self.vehicle = vehicle
        self.gains = gains or Gains()

    # ------------------------------------------------------------------

    def steering_tilt(self, state: State) -> tuple[float, float]:
        """Muốn đi về gốc toạ độ thì phải nghiêng bao nhiêu.

        Muốn sang phải thì nghiêng phải — vì lực đẩy nghiêng theo thân tàu.
        """
        g = self.gains

        theta_target = -g.lat_kp * state.x - g.lat_kd * state.vx
        phi_target = g.lat_kp * state.y + g.lat_kd * state.vy

        limit = g.max_tilt
        return (
            max(-limit, min(limit, theta_target)),
            max(-limit, min(limit, phi_target)),
        )

    def attitude_gimbal(
        self,
        state: State,
        thrust: float,
        theta_target: float,
        phi_target: float,
    ) -> tuple[float, float]:
        """Nghiêng vòi phun để đưa con tàu về góc mong muốn.

        Chia cho lực đẩy là chỗ quan trọng nhất — xem ghi chú đầu file.
        """
        g = self.gains
        v = self.vehicle

        # Lực đẩy quá yếu thì lái không có tác dụng gì; trả về 0 cho khỏi
        # sinh ra mô-men rác.
        authority = v.nozzle_offset * max(thrust, g.min_throttle_for_steering)

        # Khí động đang đẩy con tàu lệch đi, nên phải tính luôn vào đây.
        #
        # Dùng `v.stability`, KHÔNG dùng `aero_torque_gain · cross_section`.
        # Hai chỗ này từng tính khác nhau: `physics.step` biết về cánh đuôi
        # còn chỗ này thì không, nên khi gắn cánh vào thì bộ điều khiển vẫn
        # tưởng con tàu đang mất ổn định như cũ và đẩy vòi phun quá mạnh —
        # theo đúng chiều ngược lại. Phải dùng chung một con số.
        dynamic_pressure = 0.5 * 1.225 * state.speed**2
        aero = v.stability * dynamic_pressure

        error_theta = state.theta - theta_target
        error_phi = state.phi - phi_target

        delta_theta = (
            (aero + v.inertia * g.att_kp) * error_theta
            + (v.inertia * g.att_kd - v.spin_damping) * state.omega_theta
        ) / authority

        delta_phi = (
            (aero + v.inertia * g.att_kp) * error_phi
            + (v.inertia * g.att_kd - v.spin_damping) * state.omega_phi
        ) / authority

        return delta_theta, delta_phi

    def throttle_for(
        self,
        state: State,
        tank: Tank,
        mass: float,
        target: Target,
    ) -> float:
        """Mở van bao nhiêu để đạt độ cao và vận tốc mong muốn."""
        g = self.gains
        v = self.vehicle

        if target.descent_profile > 0:
            # --- Hạ cánh theo giản đồ ---
            # Ở độ cao h, cho phép rơi nhanh tối đa là sqrt(2·a·h). Rơi
            # nhanh hơn mức đó thì hãm; chậm hơn thì thả cho rơi.
            allowed = -math.sqrt(2.0 * target.descent_profile * max(state.z, 0.0))
            rate_wanted = max(allowed, -2.0)
            accel_wanted = g.land_kd * (rate_wanted - state.vz)
        else:
            # --- Lên hoặc giữ độ cao ---
            rate_wanted = target.climb_rate + g.alt_kp * (target.altitude - state.z)
            rate_wanted = max(-3.0, min(5.0, rate_wanted))
            accel_wanted = g.alt_kd * (rate_wanted - state.vz)

        # Cần lực đẩy bao nhiêu? (cộng trọng lực, chia cos nghiêng)
        tilt = state.tilt
        vertical_share = max(math.cos(tilt), 0.3)

        thrust_wanted = mass * (accel_wanted + G) / vertical_share

        if thrust_wanted <= 0:
            return 0.0

        # Van mở được tối đa là 1, nên chia cho lực đẩy lớn nhất có thể.
        ceiling = thrust_at(v, tank.pressure, 1.0)

        if ceiling <= 0:
            return 0.0

        return max(0.0, min(1.0, thrust_wanted / ceiling))

    # ------------------------------------------------------------------

    def __call__(self, state: State, tank: Tank, target: Target) -> Command:
        mass = self.vehicle.dry_mass + tank.air_mass

        if target.cut:
            # Rơi tự do. Vẫn phải lái vòi phun để giữ thăng bằng, nhưng
            # không có lực đẩy thì lái không có tác dụng — nên chỉ trả về
            # lệnh 0 và để con tàu rơi.
            return Command(throttle=0.0, gimbal_theta=0.0, gimbal_phi=0.0)

        throttle = self.throttle_for(state, tank, mass, target)
        thrust = thrust_at(self.vehicle, tank.pressure, throttle)

        theta_target, phi_target = self.steering_tilt(state)

        if not target.level:
            theta_target = phi_target = 0.0

        delta_theta, delta_phi = self.attitude_gimbal(
            state, thrust, theta_target, phi_target
        )

        return Command(
            throttle=throttle,
            gimbal_theta=delta_theta,
            gimbal_phi=delta_phi,
        ).clipped(self.vehicle)


def thrust_direction_of(command: Command) -> tuple[float, float, float]:
    """Hướng phụt thật sự, để vẽ mũi tên lực đẩy trong hình 3D."""
    return thrust_direction(command.gimbal_theta, command.gimbal_phi)


__all__ = [
    "CascadedController",
    "Command",
    "Gains",
    "P_ATM",
    "Target",
    "thrust_at",
    "thrust_direction_of",
]
