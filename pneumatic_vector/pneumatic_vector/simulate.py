"""
simulate.py — Chạy một chuyến bay và ghi lại mọi thứ.

Hai vòng lặp chạy với hai nhịp khác nhau:

    vật lý + PID       1000 Hz   mỗi 0,001 giây
    ra quyết định cấp cao  20 Hz   mỗi 0,05 giây

Chênh nhau 50 lần. Đó không phải chi tiết kỹ thuật vặt — đó chính là lý do
phải tách làm hai tầng. LLM không thể chạy ở 1000 Hz, và cũng không cần:
quyết định "lên 12 mét" thì 20 lần mỗi giây là quá đủ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .controller import CascadedController, Target
from .expert import Decision, ExpertPilot, Mission, state_text
from .physics import State, Tank, step
from .vehicle import G, Vehicle

DECISION_HZ = 20  # bao nhiêu lần mỗi giây bộ cấp cao được hỏi
LOG_HZ = 50  # bao nhiêu mẫu mỗi giây ghi vào file cho trình xem 3D


@dataclass
class FlightResult:
    """Kết quả một chuyến bay."""

    peak_altitude: float = 0.0
    landing_speed: float = 0.0
    duration: float = 0.0
    air_used: float = 0.0
    max_tilt_deg: float = 0.0
    phase_log: list[tuple[float, str]] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    crashed: bool = False
    timeout: bool = False

    def success(self, mission: Mission) -> bool:
        """Chuyến bay này có đạt yêu cầu không.

        Ba điều kiện:
            - lên tới nơi (ít nhất 10 m)
            - lên đúng tầm (không lệch quá 2 m so với đề bài)
            - chạm đất êm
        """
        if self.crashed or self.timeout:
            return False

        reached = self.peak_altitude >= 10.0
        on_target = abs(self.peak_altitude - mission.target_altitude) <= 2.0
        soft = self.landing_speed <= mission.max_landing_speed

        return reached and on_target and soft

    def summary(self) -> str:
        return (
            f"cao nhất {self.peak_altitude:5.2f} m · "
            f"chạm đất {self.landing_speed:4.2f} m/s · "
            f"nghiêng tối đa {self.max_tilt_deg:4.1f}° · "
            f"hết {self.air_used * 1000:4.0f} g khí · "
            f"{self.duration:4.1f} giây"
        )


def fly(
    vehicle: Vehicle,
    policy,
    mission: Mission | None = None,
    max_time: float = 25.0,
    keep_samples: bool = True,
    seed: int = 0,
) -> FlightResult:
    """Cho một bộ ra quyết định lái con tàu. Trả về kết quả chuyến bay."""
    mission = mission or Mission()
    tank = Tank(vehicle)
    controller = CascadedController(vehicle)

    # Cho bộ ra quyết định biết đề bài. Chuyên gia giữ sẵn `mission` từ lúc
    # tạo; model học thì cần được đưa vào, và nó cũng đọc đề bài từ đây.
    if hasattr(policy, "mission"):
        policy.mission = mission

    # Nghiêng sẵn một chút, khác nhau mỗi lần, để mỗi chuyến bay một khác.
    # Không có nhiễu thì mọi chuyến giống hệt nhau và model học được gì?
    generator = torch.Generator().manual_seed(seed)
    tilt = 0.02 + float(torch.rand(1, generator=generator).item()) * 0.04

    state = State(z=0.0, theta=tilt, phi=-tilt * 0.7)

    result = FlightResult()
    result.air_used = 0.0

    air_start = tank.air_mass
    decision_every = max(1, int(1.0 / (DECISION_HZ * vehicle.dt)))
    log_every = max(1, int(1.0 / (LOG_HZ * vehicle.dt)))

    target = Target(altitude=mission.target_altitude, climb_rate=1.8)
    decision = Decision("LEN", mission.target_altitude)
    last_phase = getattr(policy, "phase", "?")
    result.phase_log.append((0.0, str(last_phase)))

    landed = False
    previous_z = state.z

    steps = int(max_time / vehicle.dt)

    for index in range(steps):
        # --- Vòng chậm: hỏi bộ ra quyết định ---
        if index % decision_every == 0:
            decision = policy.decide(state, tank)
            target = decision.to_target(mission.target_altitude)

            phase = getattr(policy, "phase", decision.command)
            if phase != last_phase:
                result.phase_log.append((state.t, str(phase)))
                last_phase = phase

        # --- Vòng nhanh: PID rồi tới vật lý ---
        command = controller(state, tank, target)
        record = step(state, vehicle, tank, command)

        previous_z = state.z
        previous_vz = state.vz
        state = record.state

        result.peak_altitude = max(result.peak_altitude, state.z)
        result.max_tilt_deg = max(result.max_tilt_deg, state.tilt_deg)

        # --- Ghi log cho trình xem 3D ---
        if keep_samples and index % log_every == 0:
            result.samples.append(
                {
                    "t": round(state.t, 3),
                    "x": round(state.x, 3),
                    "y": round(state.y, 3),
                    "z": round(state.z, 3),
                    "vz": round(state.vz, 2),
                    "tilt": round(state.tilt_deg, 2),
                    "throttle": round(record.command.throttle, 3),
                    "gimbal": round(record.command.gimbal_theta, 4),
                    "pressure": round(record.pressure / 1e5, 2),
                    "thrust": round(record.thrust, 1),
                    "phase": last_phase,
                    "decision": decision.to_text(),
                }
            )

        # --- Chạm đất ---
        if state.z <= 0.0 and previous_z > 0.0:
            # Phải lấy vận tốc TRƯỚC bước va chạm. Trong `physics.step`, khi
            # chạm đất thì vz bị ghim về 0 để con tàu không xuyên qua mặt đất
            # — nên đọc sau đó thì chuyến bay nào cũng "hạ cánh 0,00 m/s".
            # Chỗ này từng viết sai và làm mọi cú đâm trông như hạ cánh êm.
            result.landing_speed = abs(previous_vz)
            landed = True
            break

    result.duration = state.t
    result.air_used = air_start - tank.air_mass
    result.timeout = not landed

    if not landed:
        result.landing_speed = abs(state.vz)
        result.crashed = True
    elif result.landing_speed > mission.max_landing_speed:
        result.crashed = True

    return result


def fly_expert(vehicle: Vehicle, mission: Mission | None = None, **kwargs) -> FlightResult:
    """Cho chuyên gia viết tay lái."""
    mission = mission or Mission()
    return fly(vehicle, ExpertPilot(vehicle, mission), mission, **kwargs)


def save_flight(result: FlightResult, path: str | Path, meta: dict | None = None) -> Path:
    """Ghi chuyến bay ra file để trình xem 3D phát lại."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": meta or {},
        "summary": {
            "peak_altitude": round(result.peak_altitude, 2),
            "landing_speed": round(result.landing_speed, 2),
            "duration": round(result.duration, 2),
            "max_tilt_deg": round(result.max_tilt_deg, 1),
            "air_used_g": round(result.air_used * 1000, 1),
            "crashed": result.crashed,
        },
        "phases": [{"t": round(t, 2), "name": name} for t, name in result.phase_log],
        "samples": result.samples,
    }

    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def decision_examples(vehicle: Vehicle, count: int = 4000, seed: int = 0) -> list[tuple[str, str]]:
    """Sinh dữ liệu huấn luyện: (trạng thái, quyết định của chuyên gia).

    Mỗi chuyến bay cho vài trăm ví dụ, nhưng chỉ những ví dụ ở giai đoạn
    khác nhau mới đáng học. Chạy nhiều chuyến với nhiễu khác nhau để phủ
    hết các tình huống.
    """
    examples: list[tuple[str, str]] = []

    per_flight = max(1, count // 40)

    for flight_index in range(40):
        mission = Mission(
            target_altitude=10.0 + 5.0 * ((flight_index % 5) / 4.0),
            max_landing_speed=2.0,
        )
        pilot = ExpertPilot(vehicle, mission)

        tank = Tank(vehicle)
        controller = CascadedController(vehicle)

        generator = torch.Generator().manual_seed(seed + flight_index)
        tilt = 0.02 + float(torch.rand(1, generator=generator).item()) * 0.05
        state = State(z=0.0, theta=tilt, phi=-tilt)

        target = Target(altitude=mission.target_altitude, climb_rate=1.8)
        every = max(1, int(1.0 / (DECISION_HZ * vehicle.dt)))
        taken = 0

        for index in range(int(22.0 / vehicle.dt)):
            if index % every == 0:
                decision = pilot.decide(state, tank)

                # Lấy mẫu thưa, và ưu tiên lấy đều các giai đoạn.
                if taken < per_flight and index % (every * 3) == 0:
                    examples.append((state_text(state, tank), decision.to_text()))
                    taken += 1

                target = decision.to_target(mission.target_altitude)

            command = controller(state, tank, target)
            state = step(state, vehicle, tank, command).state

            if state.z <= 0.0:
                break

    return examples


__all__ = [
    "DECISION_HZ",
    "LOG_HZ",
    "FlightResult",
    "Mission",
    "State",
    "Tank",
    "decision_examples",
    "field",
    "fly",
    "fly_expert",
    "json",
    "save_flight",
    "state_text",
]
