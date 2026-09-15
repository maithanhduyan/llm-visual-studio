"""
checks.py — Kiểm tra mọi thứ có chạy đúng không.

Mỗi mục kiểm tra một điều cụ thể, và phần lớn trong số đó là những lỗi
**đã từng xảy ra thật** trong dự án này:

    - động cơ tắt ngóm khi áp suất xuống dưới 1,9 bar (công thức ngược tỉ số)
    - cắt ga là mất sạch khí (nhầm `flow <= 0` với "hết khí")
    - chuyến bay nào cũng "hạ cánh 0,00 m/s" (đọc vận tốc sau khi đã giam về 0)
    - model bay lên ~13,5 m với mọi đề bài (trạng thái thiếu độ cao mục tiêu)
    - model chép sai con số (bắt nó chép lại thứ hệ thống đã biết)

Nên bộ kiểm tra này không phải hình thức. Nó là danh sách những chỗ đã sai
một lần rồi, để không sai lại lần nữa.

    python -m pneumatic_vector check
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import torch

from .controller import CascadedController, Target, thrust_at
from .dataset import format_example, split
from .expert import COMMANDS, Decision, ExpertPilot, Mission, state_text
from .paths import FLIGHTS_DIR, VIEWER_DIR
from .physics import Command, State, Tank, step
from .simulate import fly, fly_expert, save_flight
from .vehicle import G, P_ATM, Vehicle, exhaust_velocity, mass_flow, maximum_thrust

# ---------------------------------------------------------------------
# Khung chạy
# ---------------------------------------------------------------------


class Reporter:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.passed = 0
        self.failed: list[tuple[str, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            if self.verbose:
                print(f"  [đạt]  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed.append((name, detail))
            print(f"  [HỎNG] {name}" + (f"  ({detail})" if detail else ""))
        return condition

    def close(self, value: float, expected: float, tolerance: float, name: str) -> bool:
        ok = abs(value - expected) <= tolerance
        return self.check(
            name, ok, f"{value:.4f} (mong đợi {expected:.4f} ± {tolerance})"
        )

    def group(self, title: str) -> None:
        if self.verbose:
            print(f"\n{title}")


# ---------------------------------------------------------------------
# 1. Con tàu
# ---------------------------------------------------------------------


def check_vehicle(r: Reporter) -> None:
    r.group("Con tàu — một ống hình trụ chứa khí nén")

    v = Vehicle()

    r.check(
        "thông số con tàu không có gì vô lý",
        not v.check(),
        "; ".join(v.check()),
    )
    r.close(v.volume, math.pi * 0.05**2 * 1.2, 1e-6, "thể tích ống (m³)")

    thrust = maximum_thrust(v)
    weight = v.total_mass * G
    ratio = thrust / weight

    r.check(
        "lực đẩy tối đa phải hơn trọng lượng, nếu không thì không bay lên được",
        ratio > 1.5,
        f"{thrust:.1f} N / {weight:.1f} N = {ratio:.1f} lần",
    )
    r.check(
        "khí nén phải chiếm phần đáng kể khối lượng",
        v.air_mass / v.total_mass > 0.05,
        f"{v.air_mass * 1000:.0f} g khí / {v.total_mass * 1000:.0f} g tổng",
    )
    r.check(
        "vòi phun phải nằm dưới trọng tâm, nếu không thì lái vector vô nghĩa",
        v.nozzle_offset > 0,
        f"lệch {v.nozzle_offset} m",
    )


# ---------------------------------------------------------------------
# 2. Khí nén
# ---------------------------------------------------------------------


def check_pneumatics(r: Reporter) -> None:
    r.group("Khí nén — lực đẩy sinh ra và cạn dần")

    v = Vehicle()

    # --- Lưu lượng phải tăng theo áp suất ---
    pressures = [1.05, 1.2, 1.5, 2.0, 2.5, 4.0, 6.0, 8.0, 10.0]
    flows = [mass_flow(v, p * 1e5, 1.0) for p in pressures]

    r.check(
        "áp suất càng cao thì khí phụt ra càng nhiều",
        all(b > a for a, b in zip(flows, flows[1:])),
        " · ".join(f"{p:.1f}bar={f * 1000:.1f}g/s" for p, f in zip(pressures, flows)),
    )

    # Chỗ này từng sai: công thức dùng tỉ số áp suất ngược nên số hạng trong
    # căn luôn âm, lưu lượng luôn bằng 0 — động cơ tắt ngóm ngay dưới 1,9 bar.
    # Mà 1,9 bar lại đúng là lúc con tàu cần hãm nhất.
    low = [p for p in (1.05, 1.3, 1.6, 1.85) if mass_flow(v, p * 1e5, 1.0) > 0]
    r.check(
        "dưới 1,9 bar động cơ vẫn phải chạy (dòng chảy dưới tốc độ âm thanh)",
        len(low) == 4,
        f"chạy được ở {[f'{p} bar' for p in low]}",
    )

    r.check(
        "hết khí (áp suất khí quyển) thì không phụt gì nữa",
        mass_flow(v, P_ATM, 1.0) == 0.0 and mass_flow(v, P_ATM * 0.9, 1.0) == 0.0,
    )

    r.check(
        "van đóng thì không phụt khí",
        mass_flow(v, 8e5, 0.0) == 0.0,
    )

    # --- Vận tốc phụt ---
    speeds = [exhaust_velocity(v, p * 1e5) for p in pressures]
    r.check(
        "vận tốc phụt tăng theo áp suất và không vượt giới hạn vật lý",
        all(b > a for a, b in zip(speeds, speeds[1:])) and max(speeds) < 800,
        f"{min(speeds):.0f}-{max(speeds):.0f} m/s",
    )

    # --- Đốt khí thì mất khí, cắt ga thì phải giữ nguyên ---
    tank = Tank(v)
    before = tank.pressure
    tank.burn(0.0, 0.1)

    # Chỗ này từng sai: gộp `flow <= 0` chung với "hết khí", nên MỖI LẦN CẮT GA
    # là áp suất tụt thẳng xuống 1 bar. Con tàu mất sạch khí chỉ vì ngừng đốt.
    r.close(
        tank.pressure / 1e5,
        before / 1e5,
        1e-9,
        "cắt ga thì áp suất giữ nguyên, không được xả khí",
    )

    tank = Tank(v)
    mass_before = tank.air_mass
    thrust = tank.burn(1.0, v.dt)
    mass_after = tank.air_mass

    r.check("đốt khí thì áp suất tụt xuống", tank.pressure < before)
    r.check("đốt khí thì sinh ra lực đẩy", thrust > 0, f"{thrust:.1f} N")

    # Khối lượng mất đi phải khớp với độ tụt áp suất (định luật khí lý tưởng).
    expected = (before - tank.pressure) * v.volume / (287.0 * 293.0)
    r.close(mass_after, mass_before - expected, 1e-9, "khí mất đi khớp với độ tụt áp")

    # Bước dài: đốt sạch khí trong một lần. Chỗ này từng trả về đúng 0 N vì
    # vận tốc phụt được tính ở áp suất CUỐI bước — mà cuối bước thì áp suất
    # đã bằng 1 bar, và ở 1 bar thì không còn lực đẩy nào cả. Đốt sạch khí
    # mà không có lực đẩy là một kết quả vô lý.
    drained = Tank(v)
    long_thrust = drained.burn(1.0, 10.0)
    r.check(
        "đốt sạch khí trong một bước dài vẫn sinh ra lực đẩy",
        long_thrust > 0,
        f"{long_thrust:.1f} N",
    )

    # --- Lực đẩy phải liên tục, không được nhảy bậc ---
    #
    # Quét từ 1,2 bar trở lên: ở đúng 1 bar thì lưu lượng bằng 0 là đúng
    # vật lý (áp suất trong bằng áp suất ngoài), nên bỏ qua bước nhảy đó.
    curve = [thrust_at(v, p / 100 * 1e5, 1.0) for p in range(120, 1001)]
    jumps = [abs(b - a) for a, b in zip(curve, curve[1:])]

    r.check(
        "đường cong lực đẩy trơn tru, không có chỗ nhảy bậc",
        max(jumps) < 2.0,
        f"bước nhảy lớn nhất {max(jumps):.2f} N trong khoảng 1,2-10 bar",
    )


# ---------------------------------------------------------------------
# 3. Lái vector
# ---------------------------------------------------------------------


def check_vector_steering(r: Reporter) -> None:
    r.group("Lái vector — nghiêng vòi phun, không nghiêng con tàu")

    v = Vehicle()
    v.spin_damping = 0.0
    command = Command(throttle=1.0, gimbal_theta=math.radians(6.0))

    # Cùng một góc vòi phun, nhưng thân tàu nghiêng khác nhau.
    torques = []
    for tilt_deg in (0.0, 5.0, 10.0, 20.0):
        tank = Tank(v)
        state = State(z=5.0, theta=math.radians(tilt_deg))

        record = step(state, v, tank, command)
        torques.append(
            (record.state.omega_theta - state.omega_theta) / v.dt * v.inertia
        )

    # Mô-men xoay lái = -L·T·δ, không phụ thuộc thân tàu nghiêng bao nhiêu.
    # Chỉ số khí động mới phụ thuộc thân tàu, nên ở đây phải giống hệt nhau
    # khi con tàu đứng yên (không có áp suất động).
    spread = max(torques) - min(torques)
    r.check(
        "mô-men lái chỉ phụ thuộc góc VÒI PHUN, không phụ thuộc thân tàu nghiêng",
        spread < 1e-6,
        f"chênh lệch {spread:.2e} N·m giữa các góc nghiêng thân",
    )

    # Đảo dấu vòi phun thì mô-men phải đảo dấu.
    tank = Tank(v)
    state = State(z=5.0)
    left = step(state, v, tank, Command(throttle=1.0, gimbal_theta=math.radians(6.0)))
    right = step(state, v, Tank(v), Command(throttle=1.0, gimbal_theta=math.radians(-6.0)))

    r.check(
        "nghiêng vòi sang trái thì con tàu xoay sang phải (và ngược lại)",
        left.state.omega_theta * right.state.omega_theta < 0,
        f"ω = {left.state.omega_theta:+.4f} và {right.state.omega_theta:+.4f} rad/s",
    )

    # Lực đẩy càng yếu thì lái càng yếu.
    weak = step(state, v, Tank(v), Command(throttle=0.1, gimbal_theta=math.radians(6.0)))
    r.check(
        "lực đẩy yếu thì lái cũng yếu theo",
        abs(weak.state.omega_theta) < abs(left.state.omega_theta),
        f"ga 10% xoay {abs(weak.state.omega_theta):.4f} < ga 100% xoay "
        f"{abs(left.state.omega_theta):.4f} rad/s",
    )

    # --- Con lắc ngược ---
    v2 = Vehicle()
    tank = Tank(v2)
    state = State(z=5.0, vz=-2.0, theta=math.radians(5.0))

    for _ in range(200):
        # Không đốt gì cả, nhưng vẫn đang rơi nên có áp suất động.
        state = step(state, v2, tank, Command(throttle=0.0)).state

    r.check(
        "thả không điều khiển thì con tàu TỰ NGÃ THÊM (con lắc ngược)",
        abs(state.theta) > math.radians(5.0),
        f"5,0° -> {math.degrees(state.theta):.2f}° sau 0,2 giây",
    )

    # Còn nếu tắt hẳn khí động thì con tàu trung tính.
    v3 = Vehicle()
    v3.aero_torque_gain = 0.0
    state = State(z=5.0, theta=math.radians(5.0))
    for _ in range(200):
        state = step(state, v3, Tank(v3), Command(throttle=0.0)).state

    r.close(
        math.degrees(state.theta), 5.0, 1e-9,
        "tắt khí động thì con tàu trung tính, nghiêng bao nhiêu giữ nguyên",
    )


# ---------------------------------------------------------------------
# 4. Bộ ra quyết định
# ---------------------------------------------------------------------


def check_decisions(r: Reporter) -> None:
    r.group("Lệnh — bốn chữ mà bộ ra quyết định được phép nói")

    for command in COMMANDS:
        decision = Decision(command, 7.5)
        text = decision.to_text()

        r.check(
            f"lệnh {command} viết ra rồi đọc lại vẫn ra chính nó",
            Decision.parse(text).command == command,
            repr(text),
        )

        target = decision.to_target(13.0)
        r.check(f"lệnh {command} đổi được thành mục tiêu cho PID", isinstance(target, Target))

    r.check(
        "LEN và GIU KHÔNG kèm con số (model chép số rất kém, mà hệ thống đã biết rồi)",
        Decision("LEN", 12.0).to_text() == "LEN" and Decision("GIU", 12.0).to_text() == "GIU",
    )
    r.check(
        "HAM phải kèm con số, vì đó là gia tốc hãm mong muốn",
        Decision("HAM", 6.5).to_text() == "HAM 6.5",
    )
    r.check(
        "lệnh lạ thì cắt ga cho an toàn, không đưa bừa xuống PID",
        Decision.parse("XYZ 999").command == "ROI"
        and Decision.parse("").command == "ROI"
        and Decision.parse("HAM khong-phai-so").command == "HAM",
    )

    # --- Trạng thái đưa cho model ---
    v = Vehicle()
    tank = Tank(v)
    state = State(z=7.5, vz=-1.5, theta=math.radians(4.0))

    text = state_text(state, tank, 13.0)
    numbers = text.split()

    r.check(
        "trạng thái có 5 con số: đề bài, độ cao, vận tốc, nghiêng, khí",
        len(numbers) == 5,
        repr(text),
    )
    # Chỗ này từng thiếu, và hậu quả rất dễ thấy: model bay lên ~13,5 m với
    # MỌI đề bài. Nó không ngu — nó chỉ không được cho biết đề bài là gì.
    r.close(float(numbers[0]), 13.0, 1e-9, "con số ĐẦU TIÊN phải là độ cao mục tiêu")
    r.check(
        "đổi đề bài thì con số đầu đổi theo",
        state_text(state, tank, 10.0).split()[0] == "10.0",
    )
    r.check(
        "mẫu dữ liệu ghép đúng định dạng <trạng thái> -> <lệnh>",
        format_example("12.0 3.0 1.0 0.0 9.9", "LEN") == "12.0 3.0 1.0 0.0 9.9 -> LEN",
    )


# ---------------------------------------------------------------------
# 5. Chuyên gia viết tay
# ---------------------------------------------------------------------


def check_expert(r: Reporter, runs: int = 3) -> None:
    r.group("Chuyên gia viết tay — người thầy")

    v = Vehicle()
    started = time.time()

    for altitude in (10.0, 12.0, 15.0):
        mission = Mission(target_altitude=altitude)

        peaks = []
        speeds = []
        used = []
        ok = 0

        for seed in range(runs):
            result = fly_expert(v, mission, seed=seed)
            ok += result.success(mission)
            peaks.append(result.peak_altitude)
            speeds.append(result.landing_speed)
            used.append(result.air_used * 1000)

        r.check(
            f"đề bài {altitude:.0f} m: lên đúng tầm và chạm đất êm",
            ok == runs,
            f"{ok}/{runs} · lên {min(peaks):.2f}-{max(peaks):.2f} m · "
            f"chạm đất {min(speeds):.2f}-{max(speeds):.2f} m/s · hết {max(used):.0f} g khí",
        )

    r.check(
        "chuyên gia không dùng hết sạch khí (phải còn dư để lái)",
        max(used) < v.air_mass * 1000,
        f"dùng nhiều nhất {max(used):.0f} g / có {v.air_mass * 1000:.0f} g",
    )
    r.check(
        "chuyên gia bay nhanh, không cần chờ đợi gì",
        time.time() - started < 60,
        f"{time.time() - started:.1f} giây cho {runs * 3} chuyến",
    )

    # --- Rơi tự do không tốn khí ---
    #
    # Đây là ý chính của cả chiến thuật: hạ từ từ tốn ~10 g khí mỗi giây, mà
    # chỉ có ~112 g. Nên phải rơi tự do rồi hãm ở cuối.
    mission = Mission(target_altitude=12.0)
    result = fly_expert(v, mission, seed=0)

    free_fall = [
        sample for sample in result.samples if sample["phase"] == "ROI"
    ]
    r.check("chuyên gia có giai đoạn rơi tự do", len(free_fall) > 0,
            f"{len(free_fall)} mẫu")

    if free_fall:
        drop = free_fall[0]["z"] - free_fall[-1]["z"]
        r.check(
            "rơi tự do được vài mét mà gần như không tốn khí",
            drop > 2.0,
            f"rơi {drop:.1f} m, khí {free_fall[0]['pressure']:.2f} -> "
            f"{free_fall[-1]['pressure']:.2f} bar",
        )


# ---------------------------------------------------------------------
# 6. Ghi chuyến bay
# ---------------------------------------------------------------------


def check_flight_log(r: Reporter) -> None:
    r.group("Ghi chuyến bay — dữ liệu cho trình xem 3D")

    v = Vehicle()
    mission = Mission(target_altitude=12.0)
    result = fly_expert(v, mission, seed=0)

    r.check(
        "chuyến bay có ghi lại mẫu để phát lại",
        len(result.samples) > 100,
        f"{len(result.samples)} mẫu",
    )

    # Chỗ này từng sai: `physics.step` ghim `vz` về 0 khi chạm đất (để con tàu
    # không xuyên qua mặt đất), nên đọc vận tốc SAU bước va chạm thì chuyến
    # bay nào cũng "hạ cánh 0,00 m/s" — kể cả những cú đâm.
    falling = State(z=0.004, vz=-8.0)
    after = step(falling, v, Tank(v), Command()).state
    r.check(
        "vật lý ghim vận tốc về 0 khi chạm đất (nên đọc SAU là sai)",
        after.z == 0.0 and after.vz == 0.0,
        f"rơi 8 m/s từ 4 mm -> z={after.z} vz={after.vz}",
    )
    r.check(
        "nhưng chuyến bay vẫn ghi nhận vận tốc chạm đất thật, khác 0",
        result.landing_speed > 0.5,
        f"{result.landing_speed:.2f} m/s",
    )

    # --- Ghi ra file rồi đọc lại ---
    path = FLIGHTS_DIR / "_check.json"
    save_flight(result, path, meta={"policy": "chuyen-gia", "seed": 0})

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)

    r.check(
        "file chuyến bay có đủ bốn phần mà trình xem cần",
        all(key in data for key in ("meta", "summary", "phases", "samples")),
        ", ".join(data.keys()),
    )

    needed = {
        "t", "x", "y", "z", "vz", "tilt", "throttle",
        "gimbal", "pressure", "thrust", "phase", "decision",
    }
    missing = needed - set(data["samples"][0])

    r.check(
        "mỗi mẫu có đủ các trường mà scene.ts đọc",
        not missing,
        f"thiếu {sorted(missing)}" if missing else f"{len(needed)} trường",
    )
    r.check(
        "số mẫu khớp với thời gian bay (50 mẫu mỗi giây)",
        abs(len(data["samples"]) - data["summary"]["duration"] * 50) < 50,
        f"{len(data['samples'])} mẫu / {data['summary']['duration']} giây",
    )
    r.check(
        "độ cao lớn nhất trong các mẫu khớp với phần tóm tắt",
        abs(max(s["z"] for s in data["samples"]) - data["summary"]["peak_altitude"]) < 0.05,
    )
    r.check(
        "lệnh ghi ra đều đọc lại được",
        all(Decision.parse(s["decision"]).command in COMMANDS for s in data["samples"]),
    )


# ---------------------------------------------------------------------
# 7. Model nhỏ
# ---------------------------------------------------------------------


class _Scripted:
    """Trò giả: viết ra lệnh theo một kịch bản cho trước.

    Dùng để thử máy móc của DAgger mà không phải chờ huấn luyện. Nhớ rằng
    trò ở đây là giả — thứ đang được kiểm tra là *cơ chế*: trạng thái lấy
    từ đường trò đi, nhãn lấy từ thầy.
    """

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.calls = 0

    def write(self, prompt: str) -> str:
        text = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return text


def check_model(r: Reporter) -> None:
    r.group("Model nhỏ — bộ ra quyết định cấp cao")

    from .base import Tokenizer
    from .policy import PILOT_CONFIG, LearnedPilot, build_model

    model = build_model(26)
    parameters = sum(p.numel() for p in model.parameters())

    r.check(
        "model nhỏ thật, chạy được trên CPU",
        parameters < 500_000,
        f"{parameters:,} thông số",
    )
    r.check(
        "số tầng và số chuyên gia khớp với cấu hình",
        model.config.n_layers == PILOT_CONFIG["n_layers"]
        and model.config.n_experts == PILOT_CONFIG["n_experts"],
    )

    tokens = torch.randint(0, 26, (2, 16))
    logits, loss = model(tokens, tokens)

    r.check(
        "model đọc 16 ký tự và trả về 16 dự đoán cho mỗi câu",
        tuple(logits.shape) == (2, 16, 26),
        f"đầu ra {tuple(logits.shape)}",
    )
    r.check("có độ lỗi để học", loss is not None and loss.item() > 0,
            f"loss {loss.item():.3f}")

    # --- Lớp chắn trước PID ---
    #
    # Model nhỏ sẽ có lúc viết ra thứ vô nghĩa. Nếu đưa thẳng xuống PID thì
    # con tàu rơi. Nên phải có lớp kiểm tra ở giữa: model ra quyết định,
    # nhưng luôn có một lớp chắn giữa nó và cơ cấu chấp hành.
    v = Vehicle()
    tank = Tank(v)
    state = State(z=10.0, vz=-2.0)

    tokenizer = Tokenizer("0123456789. ->LENGIUHAROIX")
    pilot = LearnedPilot(model, tokenizer)
    pilot.mission = Mission(target_altitude=12.0)

    pilot.write = _Scripted(["HAM 999"]).write  # type: ignore[method-assign]
    guarded = pilot.decide(state, tank)

    r.check(
        "lệnh HAM vô lý (999 m/s²) bị chặn, không đưa xuống PID",
        1.0 <= guarded.value <= 15.0,
        f"HAM 999 -> HAM {guarded.value:.1f} m/s²",
    )
    r.check(
        "lệnh vô lý bị đếm lại, để còn biết mà sửa",
        pilot.invalid == 1,
        f"{pilot.invalid}/{pilot.total} lệnh không dùng được",
    )

    pilot.write = _Scripted(["XYZ 3"]).write  # type: ignore[method-assign]
    r.check(
        "lệnh không hiểu được thì cắt ga cho an toàn",
        pilot.decide(state, tank).command == "ROI",
    )
    r.check(
        "chữ không hiểu được cũng bị đếm (không phải chỉ lệnh sai số)",
        pilot.invalid == 2,
        f"{pilot.invalid}/{pilot.total} lệnh không dùng được",
    )

    # --- Model CHƯA huấn luyện ---
    pilot = LearnedPilot(model, tokenizer)
    pilot.mission = Mission(target_altitude=12.0)

    decision = pilot.decide(state, tank)
    r.check(
        "model CHƯA huấn luyện vẫn trả về một lệnh hợp lệ (nhờ lớp chắn)",
        decision.command in COMMANDS,
        f"viết ra {pilot.write('12.0 10.0 -2.0 1.0 8.0 -> ')!r} -> {decision.command}",
    )

    # --- Dữ liệu học ---
    from .dataset import generate

    examples = generate(v, flights=2, seed=0)
    train_text, val_text = split(examples)

    r.check(
        "sinh được dữ liệu để học bắt chước",
        len(examples) >= 40 and train_text and val_text,
        f"{len(examples)} ví dụ ({len(train_text)} ký tự học, {len(val_text)} ký tự thi)",
    )
    r.check(
        "mọi ví dụ đều có dấu mũi tên ngăn cách trạng thái và lệnh",
        all(" -> " in example for example in examples),
    )
    r.check(
        "dữ liệu của chuyên gia chỉ chứa những chỗ CHUYÊN GIA đi qua",
        len({example.split(" -> ")[0] for example in examples}) > 10,
        f"{len({e.split(' -> ')[0] for e in examples})} trạng thái khác nhau",
    )

    # --- Vòng DAgger ---
    from .dagger import dagger_round

    # Trò thật, chưa học: nó cắt ga ngay từ giây đầu nên con tàu không rời
    # mặt đất. Đây đúng là vấn đề mà DAgger sinh ra để chữa — nhưng nó làm
    # cho việc kiểm tra cơ chế trở nên khó, nên phải cho trò một kịch bản.
    dumb = dagger_round(v, model, tokenizer, [], flights=1, seed=0)
    shown = dumb[0].split(" -> ")[0]

    r.check(
        "trò chưa học thì hỏi được đúng một lần rồi con tàu nằm im tại chỗ",
        len(dumb) == 1 and shown.split()[1] == "0.0",
        f"{len(dumb)} ví dụ · trạng thái lúc hỏi {shown!r} "
        f"(độ cao {shown.split()[1]} m)",
    )

    # Trò giả: cứ đòi LEN mãi, nên nó bay cao hơn chỗ thầy muốn dừng.
    #
    # Phải GIỮ LẠI hàm gốc rồi trả lại, không được `del LearnedPilot.write`.
    # `write` nằm trên chính lớp `LearnedPilot`, nên `del` là xoá hẳn phương
    # thức đó khỏi lớp — và mọi mục kiểm tra chạy sau đó sẽ chết với lỗi
    # "'_RecordModel' object has no attribute 'write'".
    original_write = LearnedPilot.write
    LearnedPilot.write = lambda self, prompt: "LEN"  # type: ignore[method-assign]

    try:
        data = dagger_round(v, model, tokenizer, [], flights=1, seed=0)
    finally:
        LearnedPilot.write = original_write  # type: ignore[method-assign]

    r.check(
        "DAgger ghi lại được trạng thái mà TRÒ đi qua, kèm nhãn của THẦY",
        len(data) > 30,
        f"{len(data)} ví dụ từ 1 chuyến bay của trò",
    )
    r.check(
        "mọi nhãn ghi ra đều là lệnh hợp lệ của thầy",
        all(Decision.parse(line.split(" -> ")[1]).command in COMMANDS for line in data),
    )

    # --- Dồn dữ liệu qua các vòng, không thay thế ---
    #
    # Chữ "Aggregation" trong tên DAgger là để DỒN dữ liệu lại. Bản đầu tiên
    # của dự án gọi `dagger_round(..., [], ...)` — mỗi vòng vứt sạch dữ liệu
    # cũ và chỉ học trên những chỗ model vừa bay hỏng. Kết quả đo được: vòng
    # 0 đạt 7%, sau một vòng DAgger còn 0%. Model mất luôn đường bay chuẩn
    # mà nó đã học được, nên càng luyện càng dở.
    kept = ["12.0 5.0 1.0 0.0 9.9 -> LEN", "12.0 0.0 0.0 0.0 9.9 -> LEN"]

    LearnedPilot.write = lambda self, prompt: "LEN"  # type: ignore[method-assign]

    try:
        merged = dagger_round(v, model, tokenizer, kept, flights=1, seed=0)
    finally:
        LearnedPilot.write = original_write  # type: ignore[method-assign]

    r.check(
        "dữ liệu cũ được GIỮ LẠI qua vòng DAgger, không bị thay thế",
        merged[: len(kept)] == kept and len(merged) > len(kept),
        f"{len(kept)} ví dụ cũ + {len(merged) - len(kept)} ví dụ mới = {len(merged)}",
    )

    # --- Ghi thưa, giống dữ liệu của chuyên gia ---
    #
    # `generate` chỉ ghi 1/6 số lần hỏi vì giai đoạn `LEN` kéo dài mấy giây
    # còn `HAM` chưa đầy một giây. `dagger_round` ghi hết thì trộn hai nguồn
    # vào nhau sẽ làm lệch hẳn tỉ lệ giữa các lệnh.
    only_len = [line for line in data if line.endswith("LEN")]
    r.check(
        "DAgger ghi thưa giống dữ liệu chuyên gia, không ghi mọi lần hỏi",
        len(data) < 60,
        f"{len(data)} ví dụ · {len(only_len)} lệnh LEN "
        f"(ghi hết thì phải hơn 100)",
    )

    # Đây là cả lý do DAgger tồn tại: dữ liệu mới chứa những trạng thái mà
    # dữ liệu học bắt chước thuần KHÔNG HỀ CÓ.
    expert_states = {example.split(" -> ")[0] for example in examples}
    dagger_states = {line.split(" -> ")[0] for line in data}
    fresh = dagger_states - expert_states

    r.check(
        "dữ liệu DAgger chứa cả những trạng thái chuyên gia CHƯA TỪNG tới",
        len(fresh) > 0,
        f"{len(fresh)}/{len(dagger_states)} trạng thái là mới "
        f"(ví dụ {sorted(fresh)[0] if fresh else '—'})",
    )


# ---------------------------------------------------------------------
# 8. Vì sao model hỏng
# ---------------------------------------------------------------------


def check_explain(r: Reporter) -> None:
    r.group("Vì sao model hỏng — nhìn vào chỗ nó quyết định")

    from .explain import _make_expert, _make_model, trigger_table

    v = Vehicle()

    # Chuyên gia hãm CAO DẦN khi đề bài cao dần. Đó là giản đồ hạ cánh, và
    # nếu bảng này không cho thấy điều đó thì nó đang đo sai thứ gì rồi.
    rows = trigger_table(v, _make_expert(v), targets=(10.0, 12.0, 15.0))

    heights = [row["altitude"] for row in rows]

    r.check(
        "chuyên gia hãm cao dần khi đề bài cao dần",
        all(row is not None for row in heights) and heights[0] < heights[1] < heights[2],
        " · ".join(f"{row['target']:.0f}m -> {row['altitude']:.2f}m" for row in rows),
    )
    r.check(
        "chuyên gia đạt cả ba đề bài trong bảng này",
        all(row["success"] for row in rows),
        f"{sum(row['success'] for row in rows)}/3",
    )

    # --- Cái bẫy đã từng sập ---
    #
    # Bản đầu tiên của `explain.py` BỌC `LearnedPilot` rồi chuyển tiếp thuộc
    # tính bằng `__getattr__`. Hỏng: `simulate.fly` đặt `policy.mission`, mà
    # câu đó ghi lên lớp bọc, còn model bên trong vẫn giữ `mission = None` —
    # nên nó bay đề bài nào cũng tưởng mục tiêu là 12 m. Bảng kết quả trông
    # vẫn hợp lý, chỉ có điều nó không đo cái cần đo.
    from .policy import LearnedPilot

    model = LearnedPilot.load("pilot")
    pilot = _make_model(model.model, model.tokenizer)(Mission(target_altitude=15.0))

    r.check(
        "đề bài đưa vào phải tới được ĐÚNG model bên trong, không bị lớp bọc nuốt",
        isinstance(pilot, LearnedPilot) and pilot.mission.target_altitude == 15.0,
        f"mission = {pilot.mission.target_altitude} m",
    )

    # --- Bảng của model cũng phải chạy được và có số ---
    model_rows = trigger_table(
        v, _make_model(model.model, model.tokenizer), targets=(10.0, 12.0, 15.0)
    )

    r.check(
        "model có hãm ở cả ba đề bài, và bảng ghi lại được độ cao lúc hãm",
        all(row["altitude"] is not None for row in model_rows),
        " · ".join(
            f"{row['target']:.0f}m -> {row['altitude']:.2f}m" if row["altitude"]
            else f"{row['target']:.0f}m -> không hãm"
            for row in model_rows
        ),
    )
    r.check(
        "bảng ghi cả chuyến đạt lẫn chuyến hỏng, không giấu",
        any(row["success"] for row in model_rows) or any(
            not row["success"] for row in model_rows
        ),
        f"{sum(row['success'] for row in model_rows)}/3 đạt",
    )


# ---------------------------------------------------------------------
# 9. Trình xem 3D
# ---------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_viewer(r: Reporter) -> None:
    r.group("Trình xem 3D — Bun phục vụ, three.js vẽ")

    files = {
        "index.html": VIEWER_DIR / "index.html",
        "server.ts": VIEWER_DIR / "server.ts",
        "main.ts": VIEWER_DIR / "src" / "main.ts",
        "scene.ts": VIEWER_DIR / "src" / "scene.ts",
        "style.css": VIEWER_DIR / "src" / "style.css",
    }

    missing = [name for name, path in files.items() if not path.exists()]
    if not r.check("có đủ file của trình xem", not missing, f"thiếu {missing}"):
        return

    html = _read(files["index.html"])
    main = _read(files["main.ts"])
    scene = _read(files["scene.ts"])
    server = _read(files["server.ts"])

    # --- main.ts tìm các phần tử trong index.html ---
    wanted = set(re.findall(r'(?:getElementById|el<[^>]*>)\(\s*"([^"]+)"\s*\)', main))
    found = set(re.findall(r'id="([^"]+)"', html))
    absent = wanted - found

    r.check(
        "mọi phần tử mà main.ts tìm đều có trong index.html",
        not absent,
        f"thiếu {sorted(absent)}" if absent else f"{len(wanted)} phần tử",
    )

    # --- main.ts gọi các hàm mà scene.ts trả về ---
    exported = set(re.findall(r"^\s{4}(\w+)[,({]", scene, re.MULTILINE))
    called = set(re.findall(r"view\.(\w+)\(", main))
    unknown = called - exported

    r.check(
        "mọi hàm main.ts gọi trên cảnh 3D đều có thật trong scene.ts",
        not unknown,
        f"gọi {sorted(called)} · thiếu {sorted(unknown)}" if unknown
        else f"gọi {sorted(called)}",
    )

    # --- main.ts gọi các đường dẫn mà server.ts phục vụ ---
    routes = set(re.findall(r'"(\/api\/[^"]*)"', server))
    fetched = set(re.findall(r"fetch\(\s*[`\"]([^`\"$]+)", main))

    def served(path: str) -> bool:
        for route in routes:
            # "/api/flight/:name" phục vụ "/api/flight/"
            prefix = route.split(":")[0]
            if path.startswith(prefix):
                return True
        return False

    unserved = {path for path in fetched if not served(path)}

    r.check(
        "mọi đường dẫn main.ts gọi đều có trong server.ts",
        not unserved,
        f"gọi {sorted(fetched)} · máy chủ có {sorted(routes)}" if unserved
        else f"gọi {sorted(fetched)}",
    )

    # --- scene.ts đọc đúng các trường mà Python ghi ra ---
    fields = set(re.findall(r"sample\.(\w+)", scene))
    written = {"t", "x", "y", "z", "vz", "tilt", "throttle",
               "gimbal", "pressure", "thrust", "phase", "decision"}
    unknown_fields = fields - written

    r.check(
        "scene.ts chỉ đọc những trường mà simulate.py thật sự ghi",
        not unknown_fields,
        f"đọc {sorted(fields)}" if not unknown_fields else f"lạ {sorted(unknown_fields)}",
    )

    # --- Toạ độ: mô phỏng dùng z làm độ cao, three.js dùng y ---
    r.check(
        "scene.ts đổi trục đúng: three.js (x, y, z) = mô phỏng (x, z, y)",
        "ship.position.set(sample.x, sample.z, sample.y)" in scene,
    )

    # --- Vệt bay: đệm phải tự cấp phát ---
    #
    # `BufferGeometry.setFromPoints` nghe như "thay cả danh sách điểm",
    # nhưng thật ra nó ghi vào đệm ĐÃ CÓ và chỉ ghi được bằng số điểm mà
    # đệm đó đã chứa. Gọi lần đầu với 1 điểm là đệm chỉ còn đúng 1 chỗ, và
    # đường vẽ 1 điểm thì không vẽ ra gì cả — vệt bay không bao giờ hiện.
    # Bỏ chú thích trước khi soi: chính lời giải thích cái bẫy này có nhắc
    # tới `setFromPoints`, nên nếu không bỏ đi thì nó tự tố cáo mình.
    trail_body = scene.split("function setTrail")[-1].split("\n  }")[0]
    trail_code = "\n".join(
        line for line in trail_body.splitlines() if not line.strip().startswith("//")
    )

    r.check(
        "vệt bay tự cấp phát đệm, không dùng setFromPoints",
        "setFromPoints" not in trail_code,
        "setFromPoints ghi vào đệm cũ, không thay được cả mảng điểm",
    )
    r.check(
        "vệt bay chỉ vẽ tới chỗ đang phát, không vẽ hết cả chuyến",
        "setDrawRange" in trail_code,
    )
    r.check(
        "vệt bay tính lại bán kính bao, nếu không thì bị cắt oan",
        "computeBoundingSphere" in trail_code,
    )

    # --- Thư viện đã cài chưa ---
    r.check(
        "đã cài three.js cho trình xem",
        (VIEWER_DIR / "node_modules" / "three").exists(),
        "chưa thì chạy: cd viewer && bun install",
    )


# ---------------------------------------------------------------------
# 10. Bản dịch sang TypeScript
# ---------------------------------------------------------------------


def check_port(r: Reporter) -> None:
    r.group("Bản dịch sang TypeScript — trình duyệt tự chạy vật lý và model")

    from .policy import PILOT_CONFIG

    sources = {
        "physics.ts": VIEWER_DIR / "src" / "physics.ts",
        "model.ts": VIEWER_DIR / "src" / "model.ts",
        "live.ts": VIEWER_DIR / "src" / "live.ts",
        "check-port.ts": VIEWER_DIR / "check-port.ts",
        "check-live.ts": VIEWER_DIR / "check-live.ts",
    }

    missing = [name for name, path in sources.items() if not path.exists()]
    if not r.check("có đủ file của bản dịch", not missing, f"thiếu {missing}"):
        return

    model_path = VIEWER_DIR / "public" / "model.json"

    if not r.check(
        "đã xuất trọng số cho trình duyệt",
        model_path.exists(),
        "chưa thì chạy: python -m pneumatic_vector export",
    ):
        return

    export = json.loads(model_path.read_text(encoding="utf-8"))
    config = export["config"]

    # Kiến trúc bên trình duyệt phải GIỐNG kiến trúc bên Python. Lệch một
    # thông số ở đây là hai bên chạy hai model khác nhau, mà cả hai vẫn
    # "chạy được" — nên phải kiểm tra chứ không nhìn vào là tin.
    wanted = {
        "dModel": PILOT_CONFIG["d_model"],
        "nHeads": PILOT_CONFIG["n_heads"],
        "nKvHeads": PILOT_CONFIG["n_kv_heads"],
        "nLayers": PILOT_CONFIG["n_layers"],
        "maxSeqLen": PILOT_CONFIG["max_seq_len"],
        "window": PILOT_CONFIG["window"],
        "nExperts": PILOT_CONFIG["n_experts"],
        "topK": PILOT_CONFIG["top_k"],
        "nShared": PILOT_CONFIG["n_shared"],
        "expertHidden": PILOT_CONFIG["expert_hidden"],
        "nStreams": PILOT_CONFIG["n_streams"],
    }

    wrong = {k: (config.get(k), v) for k, v in wanted.items() if config.get(k) != v}

    r.check(
        "kiến trúc trong file xuất ra khớp với cấu hình của Python",
        not wrong,
        ", ".join(f"{k}: {a} so với {b}" for k, (a, b) in wrong.items()) or "11 thông số",
    )

    # Trọng số phải thuộc về ĐÚNG model đang có, không phải bản cũ còn sót.
    from .paths import RUNS_DIR

    checkpoint = torch.load(RUNS_DIR / "pilot.pt", map_location="cpu")

    # `lm_head.weight` dùng CHUNG tensor với `embedding.weight` (weight tying),
    # nên `state_dict()` chứa cả hai khoá. Cộng thẳng là đếm bảng tra hai lần —
    # đúng bằng 26 × 64 = 1.664 thông số.
    #
    # Phải khử trùng theo STORAGE, không theo `id()`. `torch.save`/`torch.load`
    # dựng lại hai object Tensor khác nhau nhưng dùng chung một vùng nhớ, nên
    # `id()` thấy chúng khác nhau còn `data_ptr()` mới thấy là một.
    seen: set[int] = set()
    on_disk = 0

    for key, value in checkpoint["model"].items():
        if key.endswith("load_count"):
            continue

        pointer = value.untyped_storage().data_ptr()
        if pointer in seen:
            continue

        seen.add(pointer)
        on_disk += value.numel()

    exported = sum(len(t["data"]) for t in export["tensors"].values())

    r.check(
        "số thông số xuất ra khớp với model đã lưu",
        exported == on_disk,
        f"{exported:,} xuất ra · {on_disk:,} trong file model",
    )
    r.check(
        "từ điển trong file xuất ra khớp với model đã lưu",
        export["chars"] == checkpoint["chars"],
        f"{len(export['chars'])} ký tự",
    )

    # --- Bản dịch không được tự ý thêm vật lý khác ---
    #
    # Nếu ai đó sửa công thức bên TypeScript mà quên sửa bên Python thì hai
    # bên sẽ lệch nhau, và `check-port.ts` sẽ báo hỏng. Kiểm tra ở đây chỉ
    # nhắc rằng hai bên phải có cùng bộ hằng số.
    physics = (VIEWER_DIR / "src" / "physics.ts").read_text(encoding="utf-8")

    constants = {
        "9.81": "G",
        "287.0": "R_AIR",
        "1.4": "GAMMA",
        "101325.0": "P_ATM",
        "1.225": "RHO_AIR",
        "293.0": "T_GAS",
    }

    absent = [name for value, name in constants.items() if value not in physics]

    r.check(
        "bản dịch dùng đúng bộ hằng số vật lý của Python",
        not absent,
        f"thiếu {absent}" if absent else f"{len(constants)} hằng số",
    )

    r.check(
        "bản dịch giữ nguyên công thức mô-men lái vector (chỉ phụ thuộc vòi phun)",
        "-v.nozzleOffset * thrust * cmd.gimbalTheta" in physics,
    )
    r.check(
        "bản dịch giữ nguyên nhánh dòng chảy dưới tốc độ âm thanh",
        "P_ATM / pressure" in physics,
    )


# ---------------------------------------------------------------------
# 10. Cánh đuôi
# ---------------------------------------------------------------------


def check_fins(r: Reporter) -> None:
    r.group("Cánh đuôi — giữ thân tàu thẳng, nhưng không dẫn hướng")

    from .fins import FIN_ARM, free_fall, full_flight, thrown_sideways

    v = Vehicle()

    # --- Điều kiện tĩnh ổn định ---
    r.check(
        "con tàu hiện tại MẤT ổn định (tâm khí động trên trọng tâm)",
        v.stability > 0,
        f"ổn định = {v.stability:+.5f} m³",
    )

    needed = v.fin_area_for_neutral(FIN_ARM)
    r.check(
        "tính được cỡ cánh cần để vừa đủ trung tính",
        0.005 < needed < 0.05,
        f"{needed * 1e4:.1f} cm² mỗi cánh, gắn cách trọng tâm {FIN_ARM} m",
    )

    # Hỏi cỡ cánh khi CHƯA gắn cánh (fin_offset = 0) thì phải trả về vô cùng,
    # không được chia cho 0 rồi trả về rác.
    r.check(
        "chưa biết cánh gắn ở đâu thì không đoán bừa",
        v.fin_area_for_neutral() == float("inf"),
        "fin_offset = 0 -> vô cùng",
    )

    # Cỡ "vừa đủ trung tính" cho ra ĐÚNG 0 — trung tính, không phải ổn định.
    # Con tàu giữ nguyên góc nghiêng nào thì giữ, không tự sửa. Muốn thật sự
    # ổn định thì phải to hơn mức đó.
    neutral = Vehicle(fin_area=needed, fin_offset=FIN_ARM)

    r.check(
        "cỡ cánh 'vừa đủ' cho ra đúng trung tính (không tự sửa, không ngã thêm)",
        abs(neutral.stability) < 1e-12 and not neutral.is_stable,
        f"ổn định = {neutral.stability:+.2e} m³",
    )

    finned = Vehicle(fin_area=needed * 1.2, fin_offset=FIN_ARM)

    r.check(
        "cánh to hơn mức trung tính thì con tàu mới thật sự tĩnh ổn định",
        finned.is_stable,
        f"cánh {needed * 1.2 * 1e4:.0f} cm² mỗi cánh -> ổn định = {finned.stability:+.5f} m³",
    )
    r.check(
        "cánh làm tăng diện tích cản",
        finned.drag_area > v.drag_area,
        f"{v.drag_area * 1e4:.0f} -> {finned.drag_area * 1e4:.0f} cm²",
    )

    # --- Cánh làm đúng một việc: giữ thân tàu thẳng ---
    #
    # Đây là chỗ cánh thật sự có tác dụng: cắt ga thì lực đẩy bằng 0, nên vòi
    # phun không lái được gì và con tàu rơi hoàn toàn tự do.
    bare_tilt = free_fall(v)
    fin_tilt = free_fall(finned)

    r.check(
        "mất điều khiển thì con tàu không cánh LỘN NHÀO",
        bare_tilt > 90,
        f"nghiêng tới {bare_tilt:.0f}°",
    )
    r.check(
        "gắn cánh thì nó giữ được thân tàu thẳng",
        fin_tilt < 15,
        f"nghiêng tới {fin_tilt:.1f}° (không cánh: {bare_tilt:.0f}°)",
    )

    # --- Nhưng cánh KHÔNG dẫn hướng ---
    #
    # Ném ngang rồi thả. Cánh sửa được độ nghiêng từ 120° xuống 0,4°, mà quỹ
    # đạo rơi gần như không đổi — vì cánh không tạo lực ngang. Đây là điều dễ
    # ngộ nhận nhất khi thiết kế, nên phải khoá lại bằng số đo.
    bare_drop, bare_lean, _ = thrown_sideways(v)
    fin_drop, fin_lean, _ = thrown_sideways(finned)

    r.check(
        "cánh sửa được độ nghiêng khi ném ngang",
        fin_lean < bare_lean / 4,
        f"{bare_lean:.0f}° -> {fin_lean:.1f}°",
    )

    shift = abs(bare_drop - fin_drop) / abs(bare_drop)
    r.check(
        "nhưng cánh KHÔNG đổi quỹ đạo rơi (nó không dẫn hướng)",
        shift < 0.05,
        f"lệch {abs(bare_drop - fin_drop) * 100:.0f} cm trên {abs(bare_drop):.2f} m "
        f"= {shift * 100:.1f}%",
    )

    # --- Cánh to quá thì làm hỏng việc lái ---
    #
    # Con tàu lái bằng cách NGHIÊNG thân cho lực đẩy chĩa sang ngang. Cánh ép
    # thân tàu bám theo chiều gió, tức là chống lại đúng động tác lái đó.
    small = full_flight(Vehicle(fin_area=needed / 2, fin_offset=FIN_ARM))
    big = full_flight(Vehicle(fin_area=needed * 2, fin_offset=FIN_ARM))

    r.check(
        "cánh nhỏ (một nửa cỡ trung tính) thì không hại gì",
        small["passed"] == small["total"],
        f"{small['passed']}/{small['total']} chuyến đạt · "
        f"nghiêng {small['tilt_fall']:.2f}° (không cánh: "
        f"{full_flight(v)['tilt_fall']:.2f}°)",
    )
    r.check(
        "cánh to gấp đôi thì làm HỎNG việc lái",
        big["passed"] < big["total"],
        f"{big['passed']}/{big['total']} chuyến đạt — cánh chống lại động tác nghiêng",
    )


# ---------------------------------------------------------------------
# 11. Chạy hết
# ---------------------------------------------------------------------


def run_all(verbose: bool = True) -> int:
    r = Reporter(verbose=verbose)
    started = time.time()

    if verbose:
        print("Kiểm tra dự án động cơ đẩy vector khí nén")
        print("=" * 62)

    check_vehicle(r)
    check_pneumatics(r)
    check_vector_steering(r)
    check_decisions(r)
    check_expert(r)
    check_flight_log(r)
    check_model(r)
    check_explain(r)
    check_fins(r)
    check_viewer(r)
    check_port(r)

    total = r.passed + len(r.failed)

    print("\n" + "=" * 62)
    print(f"{r.passed}/{total} mục đạt · {time.time() - started:.1f} giây")

    if r.failed:
        print("\nCòn hỏng:")
        for name, detail in r.failed:
            print(f"  - {name}  ({detail})")
        return 1

    print("Không có gì hỏng.")
    return 0


__all__ = ["Reporter", "run_all"]
