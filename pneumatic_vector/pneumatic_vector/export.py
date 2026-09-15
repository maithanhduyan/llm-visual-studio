"""
export.py — Xuất model và dữ liệu đối chiếu cho trình duyệt.

Trình duyệt chạy lại vật lý và model bằng JavaScript. Muốn nói "đây là cùng
một mô hình" thì phải chứng minh, không phải nói suông. Nên file này xuất hai
thứ:

    viewer/public/model.json       trọng số + từ điển, để trình duyệt tự chạy model
    viewer/public/reference.json   dữ liệu để ĐỐI CHIẾU

`reference.json` chứa bốn tầng, mỗi tầng kiểm tra một phần:

    1. vật lý thuần      đưa lệnh cố định, ghi LẠI TỪNG BƯỚC  -> vehicle.py, physics.py
    2. chuyên gia        bay thật, ghi từng bước             -> controller.py, expert.py
    3. model             cùng câu, phải viết ra cùng lệnh    -> deepseek_lite, policy.py
    4. đầu-cuối          cả chuyến bay, so từng mẫu          -> tất cả ghép lại

Rồi `viewer/check-port.ts` chạy bản TypeScript trên đúng dữ liệu đó và so.

    python -m pneumatic_vector export
"""

from __future__ import annotations

import json
import sys

import torch

from .controller import CascadedController, Target
from .expert import Decision, ExpertPilot, Mission, state_text
from .paths import PROJECT_ROOT, VIEWER_DIR, ensure_dirs
from .physics import State, Tank, step
from .policy import PILOT_CONFIG, LearnedPilot, build_model
from .simulate import DECISION_HZ, LOG_HZ, fly, fly_expert
from .vehicle import Vehicle

PUBLIC_DIR = VIEWER_DIR / "public"
MODEL_PATH = PUBLIC_DIR / "model.json"
REFERENCE_PATH = PUBLIC_DIR / "reference.json"

# Bao nhiêu chữ số sau dấu phẩy khi ghi ra. float32 chỉ có ~7 chữ số có nghĩa,
# nên 9 là quá đủ, mà file nhỏ hơn nhiều so với ghi cả repr của Python.
DIGITS = 9

# Cánh đuôi dùng cho ca đối chiếu thứ hai. Chọn đúng cỡ làm con tàu TĨNH ỔN
# ĐỊNH, để nhánh mô-men đổi dấu được chạy thật chứ không chỉ nằm đó.
FIN_OFFSET = 0.50


def _round(value: float) -> float:
    return round(float(value), DIGITS)


# ---------------------------------------------------------------------
# 1. Trọng số
# ---------------------------------------------------------------------


def export_weights(name: str = "pilot") -> dict:
    """Ghi trọng số model ra JSON cho trình duyệt đọc."""
    from .paths import RUNS_DIR

    checkpoint = torch.load(RUNS_DIR / f"{name}.pt", map_location="cpu")

    tokenizer = checkpoint["chars"]
    config = dict(checkpoint["config"])
    state = checkpoint["model"]

    tensors: dict[str, dict] = {}
    seen: set[int] = set()

    for key, value in state.items():
        # `load_count` là bộ đếm lúc học, không dùng khi chạy.
        if key.endswith("load_count"):
            continue

        # `lm_head.weight` dùng CHUNG tensor với `embedding.weight` (weight
        # tying). Ghi hai lần là file phình thêm 1.664 số mà chẳng để làm gì —
        # bản TypeScript tự dùng lại bảng tra cho LM Head.
        #
        # Khử trùng theo STORAGE, không theo `id()`: `torch.load` dựng lại hai
        # object Tensor khác nhau nhưng dùng chung một vùng nhớ, nên `id()`
        # thấy chúng khác nhau.
        pointer = value.untyped_storage().data_ptr()
        if pointer in seen or key == "lm_head.weight":
            continue

        seen.add(pointer)

        tensors[key] = {
            "shape": list(value.shape),
            "data": [_round(x) for x in value.reshape(-1).tolist()],
        }

    return {
        "config": {
            "vocabSize": len(tokenizer),
            "dModel": config["d_model"],
            "nHeads": config["n_heads"],
            "nKvHeads": config["n_kv_heads"],
            "nLayers": config["n_layers"],
            "maxSeqLen": config["max_seq_len"],
            "window": config["window"],
            "nExperts": config["n_experts"],
            "topK": config["top_k"],
            "nShared": config["n_shared"],
            "expertHidden": config["expert_hidden"],
            "nStreams": config["n_streams"],
        },
        "chars": tokenizer,
        "tensors": tensors,
    }


# ---------------------------------------------------------------------
# 2. Đối chiếu vật lý thuần
# ---------------------------------------------------------------------

PHYSICS_STEPS = 4000


def _scripted_command(index: int, state):
    """Lệnh cố định, chỉ phụ thuộc số bước và trạng thái — không có gì ngẫu nhiên.

    Cố tình đi qua đủ các nhánh: van mở hết (dòng tắt nghẽn), van hé (dòng
    dưới tốc độ âm thanh), cắt ga hoàn toàn, và vòi nghiêng đổi dấu liên tục.

    Chỗ giữ thăng bằng ở dưới KHÔNG phải để bắt chước bộ điều khiển thật. Nó
    chỉ để con tàu khỏi lộn nhào, và đây là bài học đáng ghi lại:

    Bản đầu tiên của bài kiểm tra này lái vòi phun bằng một hàm sin cố định,
    không quan tâm con tàu đang nghiêng bao nhiêu. Con tàu là con lắc ngược,
    nên nó lộn nhào thật, và tới bước 4.000 thì `omega_theta` lên tới
    -2,4e11 rad/s — một con số vô nghĩa về mặt vật lý. Ở đó, sai số làm tròn
    1e-16 giữa float32 và float64 bị khuếch đại thành 61, và bài kiểm tra báo
    hỏng trong khi bản dịch không hề sai.

    Hệ bất ổn thì khuếch đại sai số. Muốn so hai bản dịch với nhau thì phải
    giữ hệ trong vùng còn lành mạnh.
    """
    import math

    from .physics import Command

    phase = index / PHYSICS_STEPS

    if phase < 0.35:
        throttle = 1.0
    elif phase < 0.6:
        throttle = 0.25
    elif phase < 0.75:
        throttle = 0.0
    else:
        throttle = 0.6

    limit = math.radians(12.0)
    gimbal = -2.0 * state.theta - 0.5 * state.omega_theta
    gimbal = max(-limit, min(limit, gimbal))
    gimbal += math.radians(1.5) * math.sin(index / 173.0)

    return Command(throttle=throttle, gimbal_theta=gimbal, gimbal_phi=-gimbal * 0.5)


def reference_physics(vehicle: Vehicle, fin_area: float = 0.0) -> dict:
    """Đưa lệnh cố định, ghi lại TỪNG BƯỚC vật lý."""
    if fin_area:
        vehicle = Vehicle(fin_area=fin_area, fin_offset=FIN_OFFSET)

    tank = Tank(vehicle)
    state = State(z=0.0, theta=0.03, phi=-0.02)

    rows = []

    for index in range(PHYSICS_STEPS):
        command = _scripted_command(index, state)
        record = step(state, vehicle, tank, command)
        state = record.state

        rows.append(
            [
                _round(state.t),
                _round(state.x),
                _round(state.y),
                _round(state.z),
                _round(state.vx),
                _round(state.vy),
                _round(state.vz),
                _round(state.theta),
                _round(state.phi),
                _round(state.omega_theta),
                _round(state.omega_phi),
                _round(record.thrust),
                _round(tank.pressure),
            ]
        )

    return {
        "steps": PHYSICS_STEPS,
        "fin_area": vehicle.fin_area,
        "fin_offset": vehicle.fin_offset,
        "stability": _round(vehicle.stability),
        "drag_area": _round(vehicle.drag_area),
        "rows": rows,
    }


# ---------------------------------------------------------------------
# 3. Đối chiếu chuyên gia (có bộ điều khiển)
# ---------------------------------------------------------------------


def _fly_recording(
    vehicle: Vehicle,
    policy,
    mission: Mission,
    seed: int,
    max_time: float = 25.0,
):
    """Bản chép của `simulate.fly`, nhưng ghi lại MỌI bước.

    Không dùng thẳng `simulate.fly` vì nó chỉ ghi 50 mẫu mỗi giây. Nhưng bản
    chép này phải cho ra đúng kết quả của `fly` — có kiểm tra ở dưới.
    """
    tank = Tank(vehicle)
    controller = CascadedController(vehicle)

    if hasattr(policy, "mission"):
        policy.mission = mission

    generator = torch.Generator().manual_seed(seed)
    tilt = 0.02 + float(torch.rand(1, generator=generator).item()) * 0.04

    state = State(z=0.0, theta=tilt, phi=-tilt * 0.7)

    decision_every = max(1, int(1.0 / (DECISION_HZ * vehicle.dt)))
    log_every = max(1, int(1.0 / (LOG_HZ * vehicle.dt)))

    target = Target(altitude=mission.target_altitude, climb_rate=1.8)
    decision = Decision("LEN", mission.target_altitude)
    last_phase = getattr(policy, "phase", "?")

    rows = []
    samples = []
    phases = [(0.0, str(last_phase))]

    peak = 0.0
    max_tilt = 0.0
    landing = None
    landed = False
    previous_z = state.z

    for index in range(int(max_time / vehicle.dt)):
        if index % decision_every == 0:
            decision = policy.decide(state, tank)
            target = decision.to_target(mission.target_altitude)

            phase = getattr(policy, "phase", decision.command)
            if phase != last_phase:
                phases.append((state.t, str(phase)))
                last_phase = phase

        command = controller(state, tank, target)
        record = step(state, vehicle, tank, command)

        previous_vz = state.vz
        state = record.state

        peak = max(peak, state.z)
        max_tilt = max(max_tilt, state.tilt_deg)

        rows.append(
            [
                _round(state.t),
                _round(state.z),
                _round(state.vz),
                _round(state.theta),
                _round(state.phi),
                _round(record.thrust),
                _round(record.command.throttle),
                _round(record.command.gimbal_theta),
                _round(tank.pressure),
            ]
        )

        if index % log_every == 0:
            samples.append(
                {
                    "t": round(state.t, 3),
                    "z": round(state.z, 3),
                    "vz": round(state.vz, 2),
                    "tilt": round(state.tilt_deg, 2),
                    "throttle": round(record.command.throttle, 3),
                    "pressure": round(record.pressure / 1e5, 2),
                    "thrust": round(record.thrust, 1),
                    "phase": last_phase,
                    "decision": decision.to_text(),
                }
            )

        if state.z <= 0.0 and previous_z > 0.0:
            landing = abs(previous_vz)
            landed = True
            break

        previous_z = state.z

    return {
        "initial_tilt": round(tilt, 12),
        "rows": rows,
        "samples": samples,
        "phases": [[round(t, 2), name] for t, name in phases],
        "peak": peak,
        "max_tilt": max_tilt,
        "landing": landing if landed else abs(state.vz),
        "duration": state.t,
        "landed": landed,
        "air_used": None,
    }


def reference_expert(vehicle: Vehicle) -> dict:
    """Bay bằng chuyên gia, ghi từng bước — kiểm tra cả bộ điều khiển."""
    mission = Mission(target_altitude=12.0)

    recorded = _fly_recording(vehicle, ExpertPilot(vehicle, mission), mission, seed=0)

    # Bản chép phải cho ra ĐÚNG kết quả của `simulate.fly`, nếu không thì nó
    # không còn là bản chép nữa và mọi so sánh ở dưới thành vô nghĩa.
    real = fly_expert(vehicle, mission, seed=0)

    for name, a, b in (
        ("độ cao lớn nhất", recorded["peak"], real.peak_altitude),
        ("vận tốc chạm đất", recorded["landing"], real.landing_speed),
        ("thời gian", recorded["duration"], real.duration),
    ):
        if abs(a - b) > 1e-9:
            raise RuntimeError(
                f"Bản chép lệch bản gốc ở {name}: {a} so với {b}. "
                "Sửa `_fly_recording` trước khi xuất."
            )

    return recorded


# ---------------------------------------------------------------------
# 4. Đối chiếu model
# ---------------------------------------------------------------------

PROMPTS = 24


def reference_model(name: str = "pilot") -> dict:
    """Cùng một câu đưa vào, model phải viết ra cùng một lệnh."""
    from .paths import RUNS_DIR

    checkpoint = torch.load(RUNS_DIR / f"{name}.pt", map_location="cpu")

    from .base import Tokenizer

    tokenizer = Tokenizer.load(checkpoint["chars"])
    model = build_model(tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pilot = LearnedPilot(model, tokenizer)

    # Lấy trạng thái thật từ một chuyến bay của chuyên gia, để câu đưa vào
    # model là những câu có thật chứ không phải bịa.
    vehicle = Vehicle()
    result = fly_expert(vehicle, Mission(target_altitude=12.0), seed=0)

    step_every = max(1, len(result.samples) // PROMPTS)
    prompts = []

    for sample in result.samples[::step_every][:PROMPTS]:
        state = State(
            z=sample["z"],
            vz=sample["vz"],
            theta=0.0,
            phi=0.0,
        )

        tank = Tank(vehicle)
        tank.pressure = sample["pressure"] * 1e5

        text = f"{state_text(state, tank, 12.0)} -> "
        prompts.append(
            {
                "text": text,
                "ids": tokenizer.encode(text),
                "answer": pilot.write(text),
            }
        )

    # Vài câu nữa, cố tình đưa vào cả trạng thái lạ, để kiểm tra cả nhánh
    # "model viết ra thứ vô nghĩa".
    for altitude, vz, pressure in (
        (0.0, 0.0, 10.0),
        (30.0, -20.0, 1.2),
        (5.0, 9.9, 3.3),
        (12.0, 0.0, 5.5),
    ):
        tank = Tank(vehicle)
        tank.pressure = pressure * 1e5

        state = State(z=altitude, vz=vz, theta=0.05, phi=-0.05)
        text = f"{state_text(state, tank, 12.0)} -> "

        prompts.append(
            {"text": text, "ids": tokenizer.encode(text), "answer": pilot.write(text)}
        )

    # Logits của một câu, để so cả con số chứ không chỉ so ký tự chọn ra.
    probe = prompts[0]
    with torch.no_grad():
        logits, _ = model(torch.tensor([probe["ids"]]))
        last = logits[0, -1, :].tolist()

    return {
        "chars": tokenizer.chars,
        "prompts": prompts,
        "logit_probe": {
            "text": probe["text"],
            "ids": probe["ids"],
            "logits": [_round(x) for x in last],
        },
    }


# ---------------------------------------------------------------------
# 5. Đầu-cuối: cả chuyến bay do model lái
# ---------------------------------------------------------------------


def reference_end_to_end(name: str = "pilot") -> dict:
    from .paths import RUNS_DIR

    checkpoint = torch.load(RUNS_DIR / f"{name}.pt", map_location="cpu")

    from .base import Tokenizer

    tokenizer = Tokenizer.load(checkpoint["chars"])
    model = build_model(tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    vehicle = Vehicle()
    out = []

    for target in (10.0, 12.0, 15.0):
        mission = Mission(target_altitude=target)

        pilot = LearnedPilot(model, tokenizer)
        recorded = _fly_recording(vehicle, pilot, mission, seed=0)

        real = fly(vehicle, LearnedPilot(model, tokenizer), mission, seed=0)

        if abs(recorded["peak"] - real.peak_altitude) > 1e-9:
            raise RuntimeError(
                f"Bản chép lệch bản gốc ở đề bài {target} m: "
                f"{recorded['peak']} so với {real.peak_altitude}."
            )

        out.append(
            {
                "target": target,
                "initial_tilt": recorded["initial_tilt"],
                "rows": recorded["rows"],
                "peak": recorded["peak"],
                "landing": recorded["landing"],
                "duration": recorded["duration"],
                "landed": recorded["landed"],
            }
        )

    return {"flights": out}


# ---------------------------------------------------------------------


def main() -> int:
    ensure_dirs()
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Xuất model và dữ liệu đối chiếu cho trình duyệt")
    print("=" * 60)

    weights = export_weights()
    params = sum(len(t["data"]) for t in weights["tensors"].values())

    MODEL_PATH.write_text(
        json.dumps(weights, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"  trọng số   : {params:,} thông số · {len(weights['tensors'])} tensor")
    print(f"               {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")

    vehicle = Vehicle()

    print("  đang dựng dữ liệu đối chiếu...")

    # Ca thứ hai bật cánh đuôi lên đúng cỡ tĩnh ổn định, để nhánh mô-men đổi
    # dấu trong `physics.step` được đối chiếu thật.
    fin_area = vehicle.fin_area_for_neutral(FIN_OFFSET) * 1.2

    reference = {
        "vehicle": {
            "pressureBar": vehicle.pressure_bar,
            "dryMass": vehicle.dry_mass,
            "aeroTorqueGain": vehicle.aero_torque_gain,
        },
        "physics": reference_physics(vehicle),
        "physics_with_fins": reference_physics(vehicle, fin_area=fin_area),
        "expert": reference_expert(vehicle),
        "model": reference_model(),
        "end_to_end": reference_end_to_end(),
    }

    REFERENCE_PATH.write_text(
        json.dumps(reference, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    print(f"  vật lý     : {reference['physics']['steps']:,} bước")
    print(
        f"  có cánh    : {reference['physics_with_fins']['steps']:,} bước · "
        f"mỗi cánh {fin_area * 1e4:.0f} cm² · ổn định {reference['physics_with_fins']['stability']:+.5f} m³"
    )
    print(f"  chuyên gia : {len(reference['expert']['rows']):,} bước")
    print(f"  model      : {len(reference['model']['prompts'])} câu")
    print(f"  đầu-cuối   : {len(reference['end_to_end']['flights'])} chuyến bay")
    print(f"               {REFERENCE_PATH}  ({REFERENCE_PATH.stat().st_size / 1e6:.2f} MB)")

    print("\nGiờ chạy:  cd viewer && bun run check-port.ts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
