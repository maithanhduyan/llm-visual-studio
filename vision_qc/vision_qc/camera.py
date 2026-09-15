"""
camera.py — Lấy ảnh từ camera 2MP, và khoá mọi thứ có thể tự đổi.

Camera Logitech C922 Pro Stream: 1920x1080 = 2,07 MP, đúng 2MP.

Bốn chuyện dưới đây là bốn lần thực tế đã làm hỏng hệ thống thị giác công
nghiệp, nên chúng được xử lý ngay trong file này chứ không phải ở đâu khác:

1. **Phơi sáng tự động.** Camera tự chỉnh sáng theo khung hình. Lúc dạy model
   phòng sáng kiểu này, lúc chạy thật đèn huỳnh quang kiểu khác -> model nhận
   ảnh khác hẳn lúc học. Phải khoá.

2. **Cân trắng tự động.** Cùng lý do, và tệ hơn: cân trắng trôi chậm theo nhiệt
   độ cảm biến, nên buổi sáng và buổi chiều cho ra hai màu khác nhau.

3. **Lấy nét tự động.** Nếu camera có, nó sẽ lấy nét vào băng tải thay vì vào
   sản phẩm, và mỗi lần lấy nét là một lần ảnh mờ.

4. **Băng thông USB.** 1920x1080 ở 30 fps chỉ chạy được với MJPG. Ở chế độ thô
   YUY2, USB 2.0 chỉ đủ ~5 fps tại cỡ này.

Vì vậy `open()` trả về một **bản báo cáo** nói rõ cái gì đặt được và cái gì
không, thay vì im lặng tin rằng driver đã nghe lời. Driver im lặng bỏ qua là
chuyện bình thường.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CameraConfig


class CameraError(RuntimeError):
    """Không mở được camera, hoặc camera không chịu trả khung hình."""


# ---------------------------------------------------------------------
# Bản báo cáo sau khi mở
# ---------------------------------------------------------------------


@dataclass
class CameraReport:
    """Camera thực sự đang chạy ở đâu — không phải chỗ ta yêu cầu."""

    index: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    requested: tuple[int, int] = (0, 0)

    locks: dict[str, str] = field(default_factory=dict)
    """Tên thông số -> "đặt được" hoặc lý do không đặt được."""

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def resolution_ok(self) -> bool:
        return (self.width, self.height) == tuple(self.requested)

    @property
    def all_locked(self) -> bool:
        return bool(self.locks) and not self.refused

    @property
    def refused(self) -> list[str]:
        """Những thông số driver từ chối thẳng — đây mới là vấn đề thật."""
        return [name for name, status in self.locks.items() if status.startswith("BỊ TỪ CHỐI")]

    @property
    def unreadable(self) -> list[str]:
        """Những thông số đặt được nhưng không đọc lại được. Thường vô hại."""
        return [n for n, s in self.locks.items() if s.startswith("đặt được")]

    def describe(self) -> str:
        size = f"{self.width}x{self.height} ({self.megapixels:.2f} MP)"
        if not self.resolution_ok:
            size += f"  [yêu cầu {self.requested[0]}x{self.requested[1]} — driver trả cỡ khác]"
        return f"camera #{self.index}: {size} @ {self.fps:.1f} fps · {self.fourcc}"


# ---------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------


class Camera:
    """Camera có khoá. Dùng như context manager:

        with Camera(cfg) as cam:
            report = cam.report
            frame = cam.read()
    """

    def __init__(self, cfg: CameraConfig | None = None) -> None:
        self.cfg = cfg or CameraConfig()
        self.cap: cv2.VideoCapture | None = None
        self.report = CameraReport()

    # -- vòng đời -----------------------------------------------------

    def open(self) -> CameraReport:
        """Mở camera, đặt độ phân giải, rồi khoá phơi sáng / cân trắng / nét."""
        cfg = self.cfg

        # CAP_DSHOW là backend đúng trên Windows: MSMF hay trả khung hình đầu
        # rất chậm và đôi khi bỏ qua yêu cầu FOURCC.
        cap = cv2.VideoCapture(cfg.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Không mở được camera #{cfg.index}.\n"
                f"  - Camera đã cắm chưa? Thử `python -m vision_qc camera` để dò.\n"
                f"  - Có phần mềm khác đang giữ camera không (Zoom, Teams, OBS)?\n"
                f"    Windows chỉ cho MỘT chương trình mở camera tại một thời điểm."
            )

        self.cap = cap

        # THỨ TỰ QUAN TRỌNG — và nó ngược với trực giác.
        #
        # Đo trên C922 thật (1920x1080, backend DSHOW):
        #     đặt FOURCC trước  ->  driver chốt YUY2 thô,  4,7 fps
        #     đặt cỡ trước      ->  driver nhận MJPG,     26,2 fps
        #
        # Nhanh hơn 5,6 lần, chỉ khác thứ tự hai dòng. Ở chế độ YUY2, USB 2.0
        # không đủ băng thông cho 1080p nên camera tự hạ xuống ~5 fps — mà
        # không báo lỗi gì cả. Băng tải chạy 200 mm/s thì 5 fps là quá chậm.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        # Giữ hàng đợi ngắn: lấy ảnh cũ trong đệm còn tệ hơn là không lấy được.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.report = self._describe()
        self._discard_warmup()

        if cfg.lock:
            self.report.locks = self.lock()

        return self.report

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- đọc ảnh ------------------------------------------------------

    def read(self) -> np.ndarray:
        """Trả về một khung hình BGR. Ném lỗi nếu camera không trả gì."""
        if self.cap is None:
            raise CameraError("Camera chưa mở. Gọi `open()` trước.")

        # Đọc hai lần và lấy khung sau: với CAP_DSHOW, khung đầu tiên trong
        # đệm có thể đã nằm đó từ trước, tức là ảnh của sản phẩm trước đó.
        self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("Camera không trả về khung hình nào.")
        return frame

    def read_roi(self) -> np.ndarray:
        """Đọc ảnh rồi cắt vùng quan tâm theo cấu hình."""
        from .imageops import crop_roi

        return crop_roi(self.read(), self.cfg.roi)

    @property
    def frame_size(self) -> tuple[int, int]:
        """Cỡ ảnh SAU khi cắt ROI — đây mới là cỡ model nhìn thấy."""
        w, h = self.report.width, self.report.height
        roi = self.cfg.roi
        if roi is None:
            return w, h
        return int(round((roi[2] - roi[0]) * w)), int(round((roi[3] - roi[1]) * h))

    # -- nội bộ -------------------------------------------------------

    def _describe(self) -> CameraReport:
        cap = self.cap
        assert cap is not None

        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")

        return CameraReport(
            index=self.cfg.index,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            fourcc=fourcc,
            requested=(self.cfg.width, self.cfg.height),
        )

    def _discard_warmup(self) -> None:
        """Vứt những khung hình đầu, lúc camera còn đang tự chỉnh."""
        for _ in range(max(0, self.cfg.warmup_frames)):
            if self.cap is not None:
                self.cap.grab()

    def _set(self, prop: int, value: float, name: str, tol: float = 0.06) -> None:
        """Đặt một thông số rồi ĐỌC LẠI. Driver bỏ qua im lặng là bình thường.

        Ba kết quả, và chúng KHÁC NHAU về mức độ nghiêm trọng:

        - `OK` — đọc lại khớp.
        - `ĐẶT ĐƯỢC, KHÔNG ĐỌC LẠI ĐƯỢC` — lệnh đặt thành công (driver trả
          true) nhưng giá trị đọc về khác. DirectShow rất hay như vậy với
          `AUTO_EXPOSURE` và `WB_TEMPERATURE`: nó nhận lệnh rồi trả về -1 vì
          không có thang đo để đọc. Thường là vô hại.
        - `BỊ TỪ CHỐI` — driver trả false. Cái này mới đáng lo.

        Vì không phân biệt được chắc chắn từ xa, phép thử thật nằm ở
        `measure_stability()`: ảnh có ổn định không. Xem `verify_lock()`.
        """
        assert self.cap is not None

        accepted = self.cap.set(prop, value)
        back = self.cap.get(prop)

        # DirectShow hay trả lại giá trị hơi khác (làm tròn theo nấc phần cứng),
        # nên so bằng sai số tương đối chứ không so bằng dấu bằng.
        scale = max(abs(value), 1.0)
        close = abs(back - value) <= tol * scale

        if close:
            self.report.locks[name] = f"OK ({back:g})"
        elif accepted:
            self.report.locks[name] = (
                f"đặt được, không đọc lại được (driver trả {back:g}, xin {value:g})"
            )
        else:
            self.report.locks[name] = f"BỊ TỪ CHỐI (xin {value:g}, driver trả {back:g})"

    def lock(self) -> dict[str, str]:
        """Tắt mọi thứ tự động. Trả về báo cáo cái gì khoá được, cái gì không."""
        cfg = self.cfg
        locks: dict[str, str] = {}
        self.report.locks = locks

        # Lấy nét
        if cfg.autofocus:
            self._set(cv2.CAP_PROP_AUTOFOCUS, 1, "lấy nét tự động")
        else:
            self._set(cv2.CAP_PROP_AUTOFOCUS, 0, "tắt lấy nét tự động")
            self._set(cv2.CAP_PROP_FOCUS, cfg.focus, "lấy nét", tol=0.5)

        # Cân trắng. Đặt AUTO_WB = 0 trước, nhiệt độ màu sau, nếu không camera
        # sẽ ghi đè ngay giá trị vừa đặt.
        self._set(cv2.CAP_PROP_AUTO_WB, 0, "tắt cân trắng tự động", tol=0.5)
        self._set(cv2.CAP_PROP_WB_TEMPERATURE, cfg.wb_temperature, "nhiệt độ màu", tol=0.2)

        # Phơi sáng. Với DirectShow: 0,25 = thủ công, 0,75 = tự động.
        self._set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "tắt phơi sáng tự động", tol=0.5)
        self._set(cv2.CAP_PROP_EXPOSURE, cfg.exposure, "phơi sáng", tol=0.15)
        self._set(cv2.CAP_PROP_GAIN, cfg.gain, "độ lợi", tol=0.5)

        return locks

    # -- đo đạc -------------------------------------------------------

    def measure_fps(self, frames: int = 30) -> float:
        """Đo tốc độ thật, không tin con số driver khai."""
        for _ in range(3):
            self.read()  # bỏ vài khung cho nóng máy

        start = time.perf_counter()
        for _ in range(frames):
            self.read()
        elapsed = time.perf_counter() - start
        return frames / elapsed if elapsed > 0 else 0.0

    def measure_stability(self, frames: int = 20) -> tuple[float, float]:
        """Độ lệch độ sáng giữa các khung. Trả về (trung bình, độ lệch chuẩn).

        Đứng yên mà độ sáng còn nhảy thì có hai thủ phạm thường gặp:

        - **Đèn nhấp nháy 50 Hz.** Ở 30 fps, mỗi khung rơi vào một pha khác
          nhau của lưới điện, nên độ sáng dao động theo chu kỳ.
        - **Phơi sáng chưa khoá được.** Xem `report.locks`.

        Con số này phải nhỏ. Lớn hơn ~1% độ sáng trung bình là dấu hiệu xấu:
        model sẽ học ánh sáng thay vì học lỗi.
        """
        means: list[float] = []
        for _ in range(frames):
            frame = self.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append(float(gray.mean()))

        array = np.array(means)
        return float(array.mean()), float(array.std())

    def verify_lock(self, frames: int = 25) -> tuple[bool, str]:
        """Phép thử THẬT về việc khoá có tác dụng hay không.

        Không hỏi driver (nó hay trả lời sai), mà đo hậu quả: chĩa vào một
        cảnh đứng yên và xem độ sáng có đứng yên không. Phơi sáng còn tự động
        thì độ sáng sẽ trôi; đã khoá thì đứng im.

        Đây là con số đáng tin, còn cột `locks` chỉ là thông tin tham khảo.
        """
        mean, std = self.measure_stability(frames)
        percent = std / mean * 100 if mean > 0 else 0.0

        if percent < 0.5:
            verdict = f"ổn định ({percent:.3f}%) — khoá có tác dụng"
            return True, verdict
        if percent < 2.0:
            verdict = (f"dao động {percent:.2f}% — có thể do nhiễu cảm biến, "
                       f"hoặc đèn nhấp nháy 50 Hz")
            return True, verdict

        verdict = (
            f"dao động {percent:.1f}% — QUÁ LỚN. Phơi sáng gần như chắc chắn "
            f"chưa khoá được. Ảnh sẽ trôi theo thời gian và model học buổi "
            f"sáng sẽ sai vào buổi chiều."
        )
        return False, verdict


# ---------------------------------------------------------------------
# Dò camera
# ---------------------------------------------------------------------


def probe(indices: range = range(4)) -> list[CameraReport]:
    """Thử mở từng chỉ số camera, trả về những cái mở được.

    Dùng cho lệnh `python -m vision_qc camera`.
    """
    found: list[CameraReport] = []

    for index in indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue

        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        report = CameraReport(
            index=index,
            width=frame.shape[1],
            height=frame.shape[0],
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            fourcc="".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00"),
            requested=(frame.shape[1], frame.shape[0]),
        )
        found.append(report)
        cap.release()

    return found
