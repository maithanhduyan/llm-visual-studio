"""
dagger.py — Chữa bệnh "lạc vào chỗ thầy chưa từng tới".

## Bệnh

Học bắt chước thường thất bại một cách khó hiểu: model viết đúng 99,8% các
ví dụ, mà bay thì vẫn rơi.

Lý do: lúc học, model chỉ thấy những trạng thái mà CHUYÊN GIA đi qua. Ra
bay thật, chỉ cần lệch một chút là nó rơi vào trạng thái chuyên gia CHƯA
TỪNG ở đó — và ở đó thì nó chưa được dạy gì cả. Lệch một chút thành lệch
nhiều, rồi hỏng.

    lúc học:   thầy đi đường A, trò học đường A
    ra bay:    trò lệch sang đường B -> chưa học gì về B -> lệch tiếp

## Thuốc: DAgger

Cho trò bay, nhưng **thầy ngồi cạnh sửa ngay trên đường trò đi**:

    1. Trò bay (bằng model hiện có)
    2. Ở mỗi chỗ trò phải quyết định, hỏi thầy: "chỗ này thầy làm gì?"
    3. Ghi lại (trạng thái trò đang ở, câu trả lời của thầy)
    4. Dạy lại model trên dữ liệu cũ + dữ liệu mới
    5. Lặp lại vài vòng

Điểm mấu chốt: dữ liệu mới ghi lại **những trạng thái mà TRÒ đi qua**, không
phải thầy. Nhờ vậy model học được cả những chỗ nó hay lạc vào.

Tên đầy đủ là DAgger — Dataset Aggregation.
"""

from __future__ import annotations

import torch

from .controller import CascadedController, Target
from .dataset import format_example
from .expert import ExpertPilot, Mission, state_text
from .physics import State, Tank, step
from .vehicle import Vehicle


def dagger_round(
    vehicle: Vehicle,
    model,
    tokenizer,
    existing: list[str],
    flights: int = 60,
    seed: int = 0,
    decision_hz: int = 20,
    every: int = 6,
) -> list[str]:
    """Một vòng DAgger: cho model bay, thầy ghi nhãn đúng, trả về dữ liệu mới.

    `existing` là dữ liệu của những vòng TRƯỚC, và nó được trả về nguyên vẹn
    cùng với dữ liệu mới. Chữ "Aggregation" trong tên DAgger — Dataset
    Aggregation — chính là chỗ này. Bỏ `existing` đi thì không còn là DAgger
    nữa, mà là "dạy lại model chỉ bằng những chỗ nó vừa bay hỏng": model mất
    luôn đường bay chuẩn mà nó đã học được, và **càng luyện càng dở**.

    `every`: cứ bao nhiêu lần hỏi thì ghi một ví dụ. Ghi hết thì dữ liệu rất
    lệch — giai đoạn `LEN` kéo dài mấy giây còn `HAM` chưa đầy một giây. Con
    số này phải giống `dataset.generate`, không thì trộn hai nguồn dữ liệu
    vào nhau sẽ làm lệch hẳn tỉ lệ giữa các lệnh.
    """
    from .policy import LearnedPilot

    examples = list(existing)
    added = 0

    for index in range(flights):
        altitude = 10.0 + 5.0 * ((index % 6) / 5.0)
        mission = Mission(target_altitude=altitude)

        student = LearnedPilot(model, tokenizer)
        student.mission = mission
        teacher = ExpertPilot(vehicle, mission)

        tank = Tank(vehicle)
        controller = CascadedController(vehicle)

        generator = torch.Generator().manual_seed(seed + 1000 + index)
        tilt = 0.02 + float(torch.rand(1, generator=generator).item()) * 0.05
        state = State(z=0.0, theta=tilt, phi=-tilt)

        target = Target(altitude=altitude, climb_rate=1.8)
        ask_every = max(1, int(1.0 / (decision_hz * vehicle.dt)))
        counter = 0

        for step_index in range(int(24.0 / vehicle.dt)):
            if step_index % ask_every == 0:
                # Trò quyết định để bay...
                student_decision = student.decide(state, tank)

                # ...còn thầy luôn được hỏi, để máy trạng thái của thầy bám
                # đúng đường bay. Hỏi thầy thưa hơn thì thầy phát hiện ra
                # "tới lúc phải hãm" muộn hơn, và nhãn dạy ra cũng muộn theo.
                teacher_decision = teacher.decide(state, tank)

                # Nhưng chỉ GHI 1 trong mỗi `every` lần, giống `dataset.generate`.
                # Ghi hết thì giai đoạn `LEN` (mấy giây) lấn át `HAM` (chưa đầy
                # một giây), và model học được đúng một điều: cứ bay lên.
                if counter % every == 0:
                    examples.append(
                        format_example(
                            state_text(state, tank, altitude),
                            teacher_decision.to_text(),
                        )
                    )
                    added += 1

                counter += 1
                target = student_decision.to_target(altitude)

            command = controller(state, tank, target)
            state = step(state, vehicle, tank, command).state

            if state.z <= 0.0:
                break

    return examples


__all__ = ["dagger_round"]
