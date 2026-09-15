"""
station.py — Trạm kiểm tra: một sản phẩm đi qua, từ lúc chụp tới lúc thổi.

    chờ trigger  ->  chụp  ->  tiền xử lý  ->  model  ->  ghi PLC  ->  thổi

MỘT QUYẾT ĐỊNH PHẢI XONG TRƯỚC KHI SẢN PHẨM TỚI VAN
-----------------------------------------------------
Đây là ràng buộc thật của cả hệ thống, và nó không thương lượng:

    ngân sách = khoảng cách camera->van / tốc độ băng tải - chu kỳ quét PLC
                                                      - độ trễ van

Với thông số mặc định: 300 mm / 200 mm/s = 1500 ms, trừ 10 ms PLC và 15 ms
van -> **1475 ms**. Nghe rộng rãi, nhưng đổi tốc độ băng tải lên 1200 mm/s
(rất thật với dây chuyền nhanh) thì còn **235 ms**, và lúc đó thời gian suy
luận trên CPU trở thành vấn đề sống còn.

Model đúng 99,9% mà trả lời sau khi sản phẩm đã đi qua van thì vô nghĩa. Vì
vậy trạm đo từng công đoạn và in ra lúc kết thúc.

CÁCH CHẠY KHÔNG CẦN PHẦN CỨNG
-------------------------------
Trạm lấy ảnh từ một `FrameSource` và nói chuyện với một `RejectPlc`. Cả hai
đều có bản giả lập, nên toàn bộ vòng lặp — kể cả trình tự bắt tay và ngân sách
thời gian — chạy và đo được ngay trên máy tính, chưa cần camera lẫn PLC.

`SynthSource` còn mang theo **sự thật** (sản phẩm này có lỗi hay không), nên
trạm đếm được cả số lỗi bị bỏ sót thật sự chứ không chỉ số lần thổi.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .camera import Camera
from .config import CameraConfig, Config, LineConfig, ModelConfig
from .imageops import crop_roi, preprocess
from .paths import LOGS_DIR, ensure_dirs
from .plc import Decision, PlcError, RejectPlc

# ---------------------------------------------------------------------
# Nguồn ảnh
# ---------------------------------------------------------------------


class FrameSource(ABC):
    """Nơi ảnh đi vào trạm."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def frame(self) -> np.ndarray:
        """Ảnh BGR của sản phẩm đang ở điểm chụp."""

    def truth(self) -> bool | None:
        """Sản phẩm này có lỗi thật không? `None` = không biết (camera thật)."""
        return None

    def describe(self) -> str:
        return self.__class__.__name__


class CameraSource(FrameSource):
    """Camera thật. Không biết sự thật — đó là việc của model."""

    def __init__(self, camera_cfg: CameraConfig | None = None):
        self.camera = Camera(camera_cfg)

    def open(self) -> None:
        self.camera.open()

    def close(self) -> None:
        self.camera.close()

    def frame(self) -> np.ndarray:
        return self.camera.read_roi()

    def describe(self) -> str:
        return f"camera thật · {self.camera.report.describe()}"


class SynthSource(FrameSource):
    """Sinh ảnh giả ngay lúc chạy, kèm sự thật. Không cần camera."""

    def __init__(self, defect_rate: float = 0.3, seed: int = 0, config=None):
        from .synth import SynthConfig, render

        self.defect_rate = defect_rate
        self.rng = np.random.default_rng(seed)
        self.config = config or SynthConfig()
        self._render = render
        self._truth: bool | None = None

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def frame(self) -> np.ndarray:
        is_defect = bool(self.rng.random() < self.defect_rate)
        self._truth = is_defect
        image, _ = self._render(not is_defect, self.rng, self.config)
        return image

    def truth(self) -> bool | None:
        return self._truth

    def describe(self) -> str:
        return (f"ảnh giả {self.config.width}x{self.config.height} · "
                f"tỉ lệ lỗi {self.defect_rate:.0%}")


class FolderSource(FrameSource):
    """Phát lại ảnh có sẵn trên đĩa, kèm nhãn. Để thử lại trên dữ liệu thật."""

    def __init__(self, directory, camera_cfg: CameraConfig | None = None, seed: int = 0):
        from .dataset import scan

        self.directory = Path(directory)
        self.camera_cfg = camera_cfg
        self.samples = scan(self.directory, compute_stats=False)
        self.index = 0
        self.rng = np.random.default_rng(seed)
        self._truth: bool | None = None

        if not self.samples:
            raise SystemExit(f"Không có ảnh nào trong {self.directory}")

    def open(self) -> None:
        # Trộn lên để không phải chạy hết 200 ảnh OK rồi mới tới ảnh NG.
        order = self.rng.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in order]

    def close(self) -> None:
        pass

    def frame(self) -> np.ndarray:
        import cv2

        sample = self.samples[self.index % len(self.samples)]
        self.index += 1
        self._truth = sample.label == 1

        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Không đọc được {sample.path}")
        return crop_roi(image, self.camera_cfg.roi if self.camera_cfg else None)

    def truth(self) -> bool | None:
        return self._truth

    def describe(self) -> str:
        return f"phát lại {len(self.samples)} ảnh từ {self.directory}"


# ---------------------------------------------------------------------
# Kết quả một sản phẩm
# ---------------------------------------------------------------------


@dataclass
class PartResult:
    part_id: int
    score: float
    is_ng: bool
    frames: int

    truth: bool | None = None
    fault: bool = False

    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    plc_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def correct(self) -> bool | None:
        if self.truth is None:
            return None
        return self.is_ng == self.truth

    @property
    def escaped(self) -> bool:
        """Lỗi đi lọt — kiểu sai tốn tiền nhất."""
        return self.truth is True and not self.is_ng

    @property
    def false_reject(self) -> bool:
        """Sản phẩm tốt bị thổi oan."""
        return self.truth is False and self.is_ng


@dataclass
class StationReport:
    parts: int = 0
    rejected: int = 0

    truth_known: int = 0
    escaped: int = 0
    false_rejects: int = 0
    correct: int = 0

    budget_ms: float = 0.0
    over_budget: int = 0

    latency: dict[str, float] = field(default_factory=dict)
    results: list[PartResult] = field(default_factory=list)
    log_path: str = ""

    @property
    def reject_rate(self) -> float:
        return self.rejected / self.parts if self.parts else 0.0

    @property
    def escape_rate(self) -> float:
        return self.escaped / self.truth_known if self.truth_known else 0.0

    @property
    def false_reject_rate(self) -> float:
        return self.false_rejects / self.truth_known if self.truth_known else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("results", None)  # danh sách dài, đã có trong file log
        return data


# ---------------------------------------------------------------------
# Trạm
# ---------------------------------------------------------------------


class InspectionStation:
    """Vòng lặp kiểm tra. Một sản phẩm, một quyết định, một lần thổi."""

    def __init__(
        self,
        model,
        source: FrameSource,
        plc: RejectPlc,
        config: Config | None = None,
        threshold: float = 0.5,
        frames_per_part: int = 1,
        fail_safe_reject: bool = True,
        verbose: bool = False,
    ):
        self.model = model
        self.source = source
        self.plc = plc
        self.config = config or Config()
        self.threshold = threshold
        self.frames_per_part = max(1, frames_per_part)
        self.fail_safe_reject = fail_safe_reject
        self.verbose = verbose

        # Cỡ ảnh lấy từ CHÍNH MODEL, không lấy từ cấu hình đang mở.
        # Model biết nó cần ảnh bao nhiêu; file cấu hình thì có thể đã bị sửa
        # từ sau lần huấn luyện. Lấy sai thì model nhận ảnh sai cỡ và hoặc là
        # báo lỗi, hoặc tệ hơn là chạy sai một cách âm thầm.
        self.model_cfg: ModelConfig = getattr(model, "cfg", None) or self.config.model
        self.line: LineConfig = self.config.line

        self.part_counter = 0
        self._warmed = False

    # -- suy luận -----------------------------------------------------

    def _warmup(self) -> None:
        """Chạy thử một lần trước khi đo.

        Lần suy luận đầu tiên chậm hơn hẳn (PyTorch khởi tạo bộ nhớ, BatchNorm
        chuyển chế độ). Tính nó vào số đo thì con số p95 sai hẳn.
        """
        if self._warmed:
            return
        size = self.model_cfg.image_size
        channels = 1 if self.model_cfg.grayscale else 3
        with torch.no_grad():
            self.model(torch.zeros(1, channels, size, size))
        self._warmed = True

    def score_frame(self, image: np.ndarray) -> tuple[float, float]:
        """Một ảnh -> (P(NG), số mili-giây tiền xử lý)."""
        started = time.perf_counter()
        tensor = preprocess(image, self.model_cfg.image_size, self.model_cfg.grayscale)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        batch = torch.from_numpy(tensor).unsqueeze(0)
        with torch.no_grad():
            probability = float(torch.sigmoid(self.model(batch)).item())

        return probability, preprocess_ms

    # -- một sản phẩm -------------------------------------------------

    def inspect_one(self) -> PartResult:
        """Chụp, suy luận, gửi PLC. Trả về kết quả kèm thời gian từng công đoạn."""
        self._warmup()
        started = time.perf_counter()

        capture_ms = preprocess_ms = inference_ms = 0.0

        # --- chụp, có thể nhiều khung để bỏ phiếu ---
        scores: list[float] = []
        for _ in range(self.frames_per_part):
            shot = time.perf_counter()
            image = self.source.frame()
            capture_ms += (time.perf_counter() - shot) * 1000.0

            inference_started = time.perf_counter()
            probability, one_preprocess = self.score_frame(image)
            inference_ms += (time.perf_counter() - inference_started) * 1000.0
            preprocess_ms += one_preprocess
            scores.append(probability)

        self.part_counter += 1

        # Trung bình nhiều khung. Bỏ phiếu theo đa số cũng được, nhưng trung
        # bình giữ lại được thông tin "gần ngưỡng" — hữu ích khi cần soi lại.
        score = float(np.mean(scores))
        is_ng = score >= self.threshold

        result = PartResult(
            part_id=self.part_counter,
            score=score,
            is_ng=is_ng,
            frames=len(scores),
            truth=self.source.truth(),
            capture_ms=capture_ms,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
        )

        # --- ghi xuống PLC ---
        decision = Decision(
            part_id=result.part_id, score=score, is_ng=is_ng, frames=result.frames
        )

        plc_started = time.perf_counter()
        try:
            self.plc.send(decision)
        except PlcError:
            # Không nói được với PLC thì phải nói to lên. Bit lỗi để PLC tự
            # quyết định (dừng băng tải hay thổi hết) — quyết định đó thuộc
            # về PLC, không thuộc về PC.
            result.fault = True
            decision.fault = True
            raise

        result.plc_ms = (time.perf_counter() - plc_started) * 1000.0
        result.total_ms = (time.perf_counter() - started) * 1000.0

        return result

    # -- cả ca --------------------------------------------------------

    def run(
        self,
        max_parts: int = 100,
        trigger: str = "interval",
        name: str = "station",
    ) -> StationReport:
        """Chạy cả ca. `trigger`: "plc" = chờ PLC, "interval" = theo nhịp bảng tải."""
        ensure_dirs()

        report = StationReport(budget_ms=self.line.time_budget_ms)
        self.source.open()
        self.plc.open()

        log_path = LOGS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        report.log_path = str(log_path)

        interval = self.line.part_interval_s
        next_part_at = time.perf_counter()

        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write(json.dumps({
                    "kiểu": "bắt đầu",
                    "nguồn ảnh": self.source.describe(),
                    "plc": "giả lập" if self.plc.is_simulated else "thật",
                    "ngưỡng": self.threshold,
                    "ngân sách ms": round(report.budget_ms, 1),
                    "khung mỗi sản phẩm": self.frames_per_part,
                }, ensure_ascii=False) + "\n")

                for _ in range(max_parts):
                    if trigger == "interval":
                        # Nhịp của băng tải, không phải nhịp của CPU.
                        now = time.perf_counter()
                        if next_part_at > now:
                            time.sleep(next_part_at - now)
                        next_part_at += interval
                    else:
                        if not self.plc.wait_trigger(timeout_s=interval * 2):
                            break  # băng tải dừng

                    try:
                        result = self.inspect_one()
                    except PlcError as exc:
                        if self.verbose:
                            print(f"  LỖI PLC: {exc}")
                        log.write(json.dumps(
                            {"kiểu": "lỗi plc", "chi tiết": str(exc)}, ensure_ascii=False
                        ) + "\n")
                        break

                    report.parts += 1
                    report.rejected += int(result.is_ng)
                    report.results.append(result)

                    if result.total_ms > report.budget_ms > 0:
                        report.over_budget += 1

                    if result.truth is not None:
                        report.truth_known += 1
                        report.escaped += int(result.escaped)
                        report.false_rejects += int(result.false_reject)
                        report.correct += int(bool(result.correct))

                    # Nhịp tim mỗi vòng — PLC canh bộ đếm này để biết PC còn sống.
                    try:
                        self.plc.heartbeat()
                    except Exception:  # noqa: BLE001
                        pass

                    log.write(json.dumps({
                        "kiểu": "sản phẩm",
                        **{k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in asdict(result).items()},
                    }, ensure_ascii=False) + "\n")
                    log.flush()

                    if self.verbose:
                        note = ""
                        if result.truth is not None:
                            note = "  ĐÚNG" if result.correct else (
                                "  BỎ SÓT" if result.escaped else "  THỔI OAN")
                        print(f"  {result.part_id:4d}  điểm {result.score:.4f}  "
                              f"{'NG ' if result.is_ng else 'OK '} "
                              f"{result.total_ms:6.1f} ms{note}")

                log.write(json.dumps({
                    "kiểu": "kết thúc",
                    "số sản phẩm": report.parts,
                    "đã thổi": report.rejected,
                }, ensure_ascii=False) + "\n")

        finally:
            self.source.close()
            self.plc.close()

        if report.results:
            totals = np.array([r.total_ms for r in report.results])
            report.latency = {
                "chụp": float(np.mean([r.capture_ms for r in report.results])),
                "tiền xử lý": float(np.mean([r.preprocess_ms for r in report.results])),
                "suy luận": float(np.mean([r.inference_ms for r in report.results])),
                "ghi plc": float(np.mean([r.plc_ms for r in report.results])),
                "tổng trung bình": float(totals.mean()),
                "tổng p95": float(np.percentile(totals, 95)),
                "tổng lớn nhất": float(totals.max()),
            }

        return report


# ---------------------------------------------------------------------
# In báo cáo
# ---------------------------------------------------------------------


def print_report(report: StationReport, line: LineConfig) -> None:
    """In kết quả một ca."""
    print(f"\n  Đã kiểm tra {report.parts} sản phẩm · thổi {report.rejected} "
          f"({report.reject_rate:.1%})")

    if report.truth_known:
        print(f"\n  Trong số {report.truth_known} sản phẩm biết trước kết quả:")
        print(f"    đúng        {report.correct:5d}  {report.correct / report.truth_known:6.1%}")
        print(f"    BỎ SÓT      {report.escaped:5d}  {report.escape_rate:6.1%}"
              f"   <- lỗi đi ra thị trường")
        print(f"    thổi oan    {report.false_rejects:5d}  "
              f"{report.false_reject_rate:6.1%}   <- sản phẩm tốt bị bỏ")

    if report.latency:
        print("\n  Thời gian mỗi sản phẩm:")
        for key in ("chụp", "tiền xử lý", "suy luận", "ghi plc"):
            print(f"    {key:14s} {report.latency[key]:7.1f} ms")
        print(f"    {'TỔNG p95':14s} {report.latency['tổng p95']:7.1f} ms   "
              f"(ngân sách {report.budget_ms:.0f} ms)")

        if report.over_budget:
            print(f"    VƯỢT NGÂN SÁCH {report.over_budget}/{report.parts} sản phẩm "
                  f"-> những sản phẩm đó đã đi qua van trước khi có quyết định")
        else:
            print("    không sản phẩm nào vượt ngân sách")

    if report.log_path:
        print(f"\n  log -> {report.log_path}")
