"""
dataset.py — Sinh dữ liệu để dạy model nhỏ bắt chước chuyên gia.

Một ví dụ trông như thế này:

    12.0 13.4 0.5 1.2 5.5 -> GIU
    └──┬──┘└─────┬─────┘    └┬┘
    đề bài   trạng thái     lệnh

Năm con số bên trái là năm thứ người lái cần biết:

    muctieu  bay lên cao bao nhiêu (m) — đề bài
    cao      đang ở độ cao nào (m)
    vz       đang lên hay xuống, nhanh bao nhiêu (m/s)
    nghieng  có đang bị lệch không (độ)
    áp       còn bao nhiêu khí (bar)

Năm con số đó là **tất cả** những gì model được biết. Nó không thấy vị trí
ngang, không thấy vận tốc quay, không thấy lực đẩy. Cố ý như vậy: một người
lái giỏi cũng chỉ cần ngần ấy thứ để quyết định.

Con số ĐẦU TIÊN từng bị thiếu, và hậu quả rất dễ thấy: model bay lên ~13,5 m
với MỌI đề bài. Nó không ngu — nó chỉ không được cho biết đề bài là gì.
"""

from __future__ import annotations

from pathlib import Path

from .expert import ExpertPilot, Mission, state_text
from .physics import State, Tank
from .vehicle import Vehicle

SEPARATOR = " -> "


def format_example(state_text_value: str, decision_text: str) -> str:
    return f"{state_text_value}{SEPARATOR}{decision_text}"


def generate(
    vehicle: Vehicle,
    flights: int = 60,
    seed: int = 0,
    decision_hz: int = 20,
    every: int = 6,
) -> list[str]:
    """Bay nhiều chuyến bằng chuyên gia, ghi lại từng quyết định.

    `every`: cứ bao nhiêu lần hỏi thì lấy một ví dụ. Lấy hết thì dữ liệu
    rất lệch — giai đoạn "LEN" kéo dài 3 giây còn "HAM" chưa đầy 1 giây,
    nên nếu lấy đều theo thời gian thì model sẽ học gần như chỉ có LEN.
    """
    from .controller import CascadedController, Target
    from .physics import step

    examples: list[str] = []

    for index in range(flights):
        # Mỗi chuyến một đề bài khác nhau, để model thấy nhiều tình huống.
        altitude = 10.0 + 5.0 * ((index % 6) / 5.0)
        mission = Mission(target_altitude=altitude)
        pilot = ExpertPilot(vehicle, mission)

        tank = Tank(vehicle)
        controller = CascadedController(vehicle)

        import torch

        generator = torch.Generator().manual_seed(seed + index)
        tilt = 0.02 + float(torch.rand(1, generator=generator).item()) * 0.05
        state = State(z=0.0, theta=tilt, phi=-tilt)

        target = Target(altitude=altitude, climb_rate=1.8)
        ask_every = max(1, int(1.0 / (decision_hz * vehicle.dt)))
        counter = 0

        for step_index in range(int(24.0 / vehicle.dt)):
            if step_index % ask_every == 0:
                decision = pilot.decide(state, tank)

                if counter % every == 0:
                    examples.append(
                        format_example(
                            state_text(state, tank, mission.target_altitude),
                            decision.to_text(),
                        )
                    )

                counter += 1
                target = decision.to_target()

            command = controller(state, tank, target)
            state = step(state, vehicle, tank, command).state

            if state.z <= 0.0:
                break

    return examples


def split(examples: list[str], val_fraction: float = 0.1) -> tuple[str, str]:
    """Chia dữ liệu thành phần học và phần thi."""
    cut = int(len(examples) * (1.0 - val_fraction))
    return "\n".join(examples[:cut]) + "\n", "\n".join(examples[cut:]) + "\n"


def save(examples: list[str], path: str | Path) -> Path:
    """Ghi dữ liệu ra file để mở mà đọc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# Dữ liệu huấn luyện bộ ra quyết định cấp cao\n"
        "# Mỗi dòng: <muctieu> <cao> <vz> <nghieng> <ap> -> <lenh> <so>\n"
        "# Lấy từ chuyên gia viết tay, bay nhiều chuyến với đề bài khác nhau.\n"
    )

    path.write_text(header + "\n".join(examples) + "\n", encoding="utf-8")
    return path


def load(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip() and not line.startswith("#")]


__all__ = [
    "SEPARATOR",
    "format_example",
    "generate",
    "load",
    "save",
    "split",
]
