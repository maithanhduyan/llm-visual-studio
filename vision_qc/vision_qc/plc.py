"""
plc.py — Nói với PLC: gửi quyết định, và giữ cho PLC biết PC còn sống.

GIAO THỨC BẮT TAY
------------------
Đây là phần dễ làm hỏng nhất của cả hệ thống, vì máy tính và PLC chạy song
song và không ai chờ ai. Trình tự:

    PLC -> PC   M100 = 1        "sản phẩm đã tới điểm chụp"
    PC  -> PLC  D200 = điểm     điểm lỗi, 0..10000
    PC  -> PLC  D201 = số thứ tự sản phẩm
    PC  -> PLC  M201 = 0 hoặc 1 (OK / NG)
    PC  -> PLC  M200 = 1        "kết quả đã sẵn sàng"
    PLC         đọc, thổi nếu M201 = 1 (giữ xung theo `reject_pulse_ms`)
    PLC -> PC   M200 = 0        "đã nhận" — chính việc PLC xoá M200 là ack

Dùng chính việc PLC xoá M200 làm tín hiệu xác nhận, không thêm bit riêng: ít
bit hơn thì ít chỗ để lập trình sai hơn. PC đặt M200 = 1 rồi chờ nó về 0.
Chờ quá lâu -> M202 (fault) = 1.

BA THỨ BẢO VỆ, KHÔNG PHẢI CHO ĐẸP
----------------------------------
1. **Nhịp tim (D210).** PC tăng đều một bộ đếm. Chương trình PLC canh bộ đếm
   đó; đứng yên 5 giây là PLC biết PC đã chết. Không có nhịp tim thì PC treo
   giữa ca, PLC vẫn chạy, và mọi sản phẩm lỗi đi thẳng ra thị trường.

2. **Bit lỗi (M202).** PC tự nói "tôi đang hỏng". PLC quyết định làm gì —
   dừng băng tải, hay thổi hết. Việc quyết định thuộc về PLC, không thuộc PC.

3. **Hướng hỏng an toàn.** Khi không suy luận được, mặc định là **thổi**.
   Thà loại oan một sản phẩm tốt còn hơn để lọt một sản phẩm lỗi. Đổi được
   bằng `fail_safe_reject`.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import LineConfig, PlcConfig


class PlcError(RuntimeError):
    """Mất liên lạc với PLC, hoặc PLC không xác nhận."""


# ---------------------------------------------------------------------
# Một quyết định
# ---------------------------------------------------------------------


@dataclass
class Decision:
    """Kết quả kiểm tra của MỘT sản phẩm, đúng thứ sẽ ghi xuống PLC."""

    part_id: int
    score: float          # P(NG), 0..1
    is_ng: bool
    frames: int = 1       # số khung hình đã bỏ phiếu
    fault: bool = False   # True khi PC không suy luận được

    @property
    def scaled_score(self) -> int:
        """Điểm ghi xuống PLC: số nguyên 0..10000 (PLC không có dấu phẩy động)."""
        return int(round(min(max(self.score, 0.0), 1.0) * 10000))

    def describe(self) -> str:
        tag = "LỖI " if self.is_ng else "tốt "
        note = "  [KHÔNG SUY LUẬN ĐƯỢC]" if self.fault else ""
        return (f"sản phẩm #{self.part_id:6d}  {tag} "
                f"điểm {self.score:.4f} ({self.scaled_score:5d}/10000)  "
                f"{self.frames} khung{note}")


# ---------------------------------------------------------------------
# Giao diện chung
# ---------------------------------------------------------------------


class RejectPlc(ABC):
    """PLC điều khiển van thổi."""

    #: Với backend thật thì True; giả lập thì False.
    is_simulated: bool = False

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def is_ready(self) -> bool:
        """PLC có đang chạy và cho phép gửi kết quả không?"""

    @abstractmethod
    def wait_trigger(self, timeout_s: float) -> bool:
        """Chờ tới khi có sản phẩm ở điểm chụp."""

    @abstractmethod
    def send(self, decision: Decision) -> float:
        """Ghi kết quả xuống PLC. Trả về số mili-giây đã mất."""

    @abstractmethod
    def heartbeat(self) -> None:
        """Báo cho PLC biết PC còn sống."""

    def stats(self) -> dict:
        return {}


# ---------------------------------------------------------------------
# PLC Mitsubishi thật
# ---------------------------------------------------------------------


class McRejectPlc(RejectPlc):
    """PLC Mitsubishi FX5U / Q series, qua giao thức MC trên TCP."""

    def __init__(self, cfg: PlcConfig | None = None, line: LineConfig | None = None):
        self.cfg = cfg or PlcConfig()
        self.line = line or LineConfig()
        self.client = None
        self.heartbeat_count = 0
        self.sent = 0
        self.ack_timeouts = 0
        self.last_error: str | None = None

    # -- vòng đời -----------------------------------------------------

    def open(self) -> None:
        from .mc import McClient

        try:
            self.client = McClient(
                self.cfg.host, self.cfg.port, self.cfg.timeout_s, self.cfg.retries
            )
            self.client.connect()

            # Dọn trạng thái cũ. PLC có thể còn giữ kết quả của lần chạy trước,
            # và ghi đè lên một kết quả chưa được xử lý là cách để bỏ sót lỗi.
            self.client.write_bit(self.cfg.result_valid, False)
            self.client.write_bit(self.cfg.ng_flag, False)
            self.client.write_bit(self.cfg.fault, False)

            if not self.is_ready():
                raise PlcError(
                    f"PLC {self.cfg.host}:{self.cfg.port} kết nối được nhưng "
                    f"{self.cfg.plc_ready} = 0 (PLC chưa cho phép chạy)."
                )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self) -> McRejectPlc:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- tín hiệu -----------------------------------------------------

    def is_ready(self) -> bool:
        assert self.client is not None
        return self.client.read_bit(self.cfg.plc_ready)

    def wait_trigger(self, timeout_s: float) -> bool:
        """Chờ M100 = 1. Hỏi liên tục chứ không ngủ lâu: mỗi mili-giây chờ thừa
        là một mili-giây lấy từ ngân sách thời gian."""
        assert self.client is not None
        deadline = time.perf_counter() + timeout_s

        while time.perf_counter() < deadline:
            try:
                if self.client.read_bit(self.cfg.trigger):
                    return True
            except Exception as exc:  # noqa: BLE001
                # Mất một nhịp đọc giữa ca là chuyện bình thường (nhiễu mạng
                # công nghiệp). Thử lại, chỉ báo lỗi khi hết thời gian chờ.
                self.last_error = str(exc)
            time.sleep(0.002)

        return False

    def send(self, decision: Decision) -> float:
        """Ghi kết quả và chờ PLC xác nhận. Trả về số mili-giây đã mất."""
        assert self.client is not None
        started = time.perf_counter()

        self.client.write_word(self.cfg.score, decision.scaled_score)
        self.client.write_word(self.cfg.part_id, decision.part_id & 0xFFFF)

        # Ghi cờ lỗi TRƯỚC khi bật result_valid — nếu bật trước, PLC có thể
        # đọc được kết quả của sản phẩm trước đó trong lúc cờ chưa kịp ghi.
        self.client.write_bit(self.cfg.ng_flag, decision.is_ng)
        self.client.write_bit(self.cfg.fault, decision.fault)
        self.client.write_bit(self.cfg.result_valid, True)

        deadline = started + (self.line.plc_scan_ms + self.cfg.timeout_s * 1000) / 1000.0
        while time.perf_counter() < deadline:
            if not self.client.read_bit(self.cfg.result_valid):
                self.sent += 1
                return (time.perf_counter() - started) * 1000.0
            time.sleep(0.001)

        self.ack_timeouts += 1
        self.client.write_bit(self.cfg.result_valid, False)
        raise PlcError(
            f"PLC không xác nhận sau {self.cfg.timeout_s:.1f} s "
            f"(M200 vẫn = 1). Kiểm tra chương trình PLC có xoá M200 sau khi đọc."
        )

    def heartbeat(self) -> None:
        assert self.client is not None
        self.heartbeat_count = (self.heartbeat_count + 1) & 0xFFFF
        self.client.write_word(self.cfg.heartbeat, self.heartbeat_count)

    def stats(self) -> dict:
        return {
            "gửi được": self.sent,
            "hết thời gian chờ xác nhận": self.ack_timeouts,
            "nhịp tim": self.heartbeat_count,
        }


# ---------------------------------------------------------------------
# PLC giả lập
# ---------------------------------------------------------------------


class SimRejectPlc(RejectPlc):
    """PLC giả, chạy trong tiến trình. Để thử cả trạm khi chưa có phần cứng.

    Nó bắt chước đúng trình tự bắt tay của PLC thật, kể cả việc tự xoá M200
    sau khi nhận. Nhờ vậy `station.py` chạy y hệt trên hai backend, và lỗi
    trong trình tự bắt tay lộ ra ngay trên máy tính chứ không phải trong xưởng.
    """

    is_simulated = True

    def __init__(
        self,
        line: LineConfig | None = None,
        simulate_ack_delay_ms: float = 2.0,
        simulate_plc_scan_ms: float | None = None,
    ):
        self.line = line or LineConfig()
        self.ack_delay_ms = simulate_ack_delay_ms
        self.scan_ms = simulate_plc_scan_ms if simulate_plc_scan_ms is not None else \
            self.line.plc_scan_ms

        self.ready = True
        self.trigger = False
        self.result_valid = False
        self.ng_flag = False
        self.fault = False
        self.score = 0
        self.part_id = 0
        self.heartbeat_count = 0
        self.last_heartbeat_at = time.perf_counter()

        self.decisions: list[Decision] = []
        self.rejected = 0
        self._lock = threading.Lock()
        self._valid_set_at = 0.0

    # -- điều khiển từ bên ngoài (dùng trong test và mô phỏng) ---------

    def fire_trigger(self, present: bool = True) -> None:
        self.trigger = present

    # -- giao diện RejectPlc ------------------------------------------

    def open(self) -> None:
        self.ready = True
        self.trigger = False
        self.result_valid = False
        self.last_heartbeat_at = time.perf_counter()

    def close(self) -> None:
        self.ready = False

    def is_ready(self) -> bool:
        return self.ready

    def wait_trigger(self, timeout_s: float) -> bool:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self.trigger:
                return True
            time.sleep(0.001)
        return False

    def send(self, decision: Decision) -> float:
        started = time.perf_counter()

        with self._lock:
            if self.result_valid:
                raise PlcError(
                    "Kết quả trước chưa được PLC nhận (M200 vẫn = 1). "
                    "PC đang ghi đè lên một quyết định chưa xử lý."
                )

            self.score = decision.scaled_score
            self.part_id = decision.part_id
            self.ng_flag = decision.is_ng
            self.fault = decision.fault
            self.result_valid = True
            self._valid_set_at = started
            self.decisions.append(decision)
            if decision.is_ng:
                self.rejected += 1

        # PLC quét theo chu kỳ, rồi mới xoá M200. Bắt chước đúng độ trễ đó.
        time.sleep((self.scan_ms + self.ack_delay_ms) / 1000.0)

        with self._lock:
            self.result_valid = False

        return (time.perf_counter() - started) * 1000.0

    def heartbeat(self) -> None:
        self.heartbeat_count = (self.heartbeat_count + 1) & 0xFFFF
        self.last_heartbeat_at = time.perf_counter()

    def heartbeat_age_s(self) -> float:
        """Bao lâu rồi chưa có nhịp tim. PLC thật canh con số này."""
        return time.perf_counter() - self.last_heartbeat_at

    def stats(self) -> dict:
        return {
            "kết quả đã gửi": len(self.decisions),
            "đã thổi": self.rejected,
            "nhịp tim": self.heartbeat_count,
        }


# ---------------------------------------------------------------------
# Nhà máy
# ---------------------------------------------------------------------


def make_plc(cfg: PlcConfig | None = None, line: LineConfig | None = None) -> RejectPlc:
    """Tạo backend theo cấu hình. `mc` = PLC thật, `sim` = giả lập."""
    cfg = cfg or PlcConfig()

    if cfg.backend == "sim":
        return SimRejectPlc(line)
    if cfg.backend == "mc":
        return McRejectPlc(cfg, line)

    raise PlcError(
        f"Backend PLC không biết: {cfg.backend!r}. Chọn 'sim' hoặc 'mc'."
    )
