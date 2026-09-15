"""
capture.py — Thu ảnh từ camera 2MP. Đây là bước quyết định cả dự án.

Một model không thể tốt hơn dữ liệu dạy nó. Và trong thị giác công nghiệp, ba
lỗi dữ liệu dưới đây phổ biến tới mức đáng gọi là quy luật:

1. **Ảnh OK và ảnh NG chụp trong hai điều kiện sáng khác nhau.** Chụp 200 ảnh
   tốt buổi sáng, 40 ảnh lỗi buổi chiều. Model đạt 99% trên tập thi và bằng 0
   ngoài xưởng, vì nó học "sáng = tốt". Cách chữa: **trộn lẫn hai lớp trong
   cùng một lượt chụp**, đổi qua đổi lại liên tục.

2. **Nhiều khung hình của CÙNG MỘT sản phẩm.** Bấm 20 lần cho một chi tiết đang
   đứng yên. Được 20 ảnh nhưng chỉ có một sản phẩm — model học thuộc lòng chi
   tiết đó. Cách chữa: mỗi lần bấm là một **sản phẩm mới** (`part_id` tăng),
   và tập thi chia theo sản phẩm chứ không theo ảnh.

3. **Bấm nhầm, ảnh trùng.** Cách chữa: băm ảnh, chặn ảnh gần giống nhau.

Công cụ này xử lý cả ba, và nhắc liên tục trong lúc chụp.

    SPACE / 1   chụp vào lớp OK
    2 / X       chụp vào lớp NG
    u           xoá lần chụp vừa rồi
    d           xoá ảnh trùng (giữ ảnh đầu)
    b           đổi tên sản phẩm — bắt đầu một chi tiết mới
    r           bật/tắt vùng cắt ROI
    [ ]         giảm/tăng phơi sáng
    - =         giảm/tăng độ lợi
    q / ESC     thoát
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, CameraError
from .config import CameraConfig
from .imageops import dhash, hamming
from .paths import CLASSES, ensure_dirs

WINDOW = "vision_qc - thu anh"


# ---------------------------------------------------------------------
# Trạng thái một lượt chụp
# ---------------------------------------------------------------------


@dataclass
class CaptureSession:
    """Đếm và ghi nhớ mọi thứ về lượt chụp hiện tại."""

    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in CLASSES})
    part_id: int = 0
    hashes: dict[str, list[tuple[int, str]]] = field(
        default_factory=lambda: {name: [] for name in CLASSES}
    )
    recent: list[Path] = field(default_factory=list)
    skipped_duplicates: int = 0
    last_label: str = ""

    def total(self) -> int:
        return sum(self.counts.values())

    def next_part(self) -> int:
        self.part_id += 1
        return self.part_id

    def find_duplicate(self, digest: int, label: str, tolerance: int = 2) -> str | None:
        """Tên ảnh gần trùng đã có trong lớp này, hoặc None."""
        for other_digest, other_name in self.hashes[label]:
            if hamming(digest, other_digest) <= tolerance:
                return other_name
        return None

    def remember(self, label: str, digest: int, name: str) -> None:
        self.hashes[label].append((digest, name))
        self.counts[label] += 1

    def undo(self) -> int:
        """Xoá những ảnh của lần chụp vừa rồi (một sản phẩm = một lần bấm)."""
        if not self.recent:
            return 0

        last_part = self.recent[-1].stem.split("_")[1]  # "p00007"
        removed = 0

        while self.recent and self.recent[-1].stem.split("_")[1] == last_part:
            path = self.recent.pop()
            label = "OK" if path.parent.name == "OK" else "NG"

            if path.exists():
                path.unlink()
                removed += 1
                self.counts[label] = max(0, self.counts[label] - 1)

                for index in range(len(self.hashes[label]) - 1, -1, -1):
                    if self.hashes[label][index][1] == path.name:
                        self.hashes[label].pop(index)
                        break

        return removed


# ---------------------------------------------------------------------
# Ghi ảnh
# ---------------------------------------------------------------------


def save_frame(
    image: np.ndarray,
    label: str,
    part_id: int,
    sequence: int,
    raw_dir: Path,
) -> Path:
    """Ghi một ảnh theo đúng quy ước tên mà `dataset.py` đọc được."""
    folder = raw_dir / label
    folder.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = folder / f"{label}_p{part_id:05d}_{stamp}_{sequence:02d}.jpg"

    # Chất lượng 95: nén mạnh tay xoá đúng những chi tiết nhỏ mà ta cần tìm.
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


# ---------------------------------------------------------------------
# Vẽ lớp phủ lên ảnh xem trước
# ---------------------------------------------------------------------


def draw_overlay(
    frame: np.ndarray,
    session: CaptureSession,
    fps: float,
    brightness: float,
    camera_cfg: CameraConfig,
    roi: tuple[float, float, float, float] | None,
    message: str = "",
) -> np.ndarray:
    """Vẽ thông tin lên ảnh xem trước."""
    view = frame.copy()

    if roi is not None:
        height, width = view.shape[:2]
        x0, y0 = int(roi[0] * width), int(roi[1] * height)
        x1, y1 = int(roi[2] * width), int(roi[3] * height)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 255), 2)

    height, width = view.shape[:2]
    scale = width / 1280.0

    # Nền mờ cho chữ dễ đọc
    panel = view.copy()
    cv2.rectangle(panel, (0, 0), (int(430 * scale), int(210 * scale)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.45, view, 0.55, 0, view)

    def text(line: str, row: int, color=(255, 255, 255), size=0.62) -> None:
        cv2.putText(view, line, (int(12 * scale), int(row * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, size * scale, color,
                    max(1, int(2 * scale)), cv2.LINE_AA)

    text(f"OK {session.counts['OK']:4d}     NG {session.counts['NG']:4d}", 32)
    text(f"san pham ke tiep: #{session.part_id + 1}", 60, (200, 255, 200))
    text(f"sang {brightness:5.1f}   {fps:4.1f} fps", 88, (200, 200, 200), 0.55)
    text(f"phoi sang {camera_cfg.exposure:g}  do loi {camera_cfg.gain:g}", 112,
         (200, 200, 200), 0.55)

    if session.skipped_duplicates:
        text(f"bo qua {session.skipped_duplicates} anh trung", 136, (120, 200, 255), 0.55)

    if brightness < 60:
        text("ANH QUA TOI - tang phoi sang [ hoac do loi =", 164, (0, 180, 255), 0.55)
    elif brightness > 200:
        text("ANH QUA SANG - giam phoi sang ] hoac do loi -", 164, (0, 180, 255), 0.55)
    elif abs(brightness - 128) < 60:
        text("do sang tot", 164, (120, 255, 120), 0.55)

    if message:
        text(message, height - 20 if height > 300 else 190, (0, 255, 255), 0.65)

    return view


# ---------------------------------------------------------------------
# Vòng lặp chính
# ---------------------------------------------------------------------


def run_capture(
    camera_cfg: CameraConfig | None = None,
    raw_dir: Path | None = None,
    burst: int = 1,
    display: bool = True,
    min_ok: int = 150,
    min_ng: int = 50,
) -> CaptureSession:
    """Mở camera, cho người dùng chụp ảnh vào hai lớp.

    `burst` khung hình mỗi lần bấm — hữu ích khi sản phẩm đang chạy trên băng
    tải và ta muốn bắt nhiều góc. Vẫn tính là MỘT sản phẩm.
    """
    ensure_dirs()
    from .paths import RAW_DIR

    raw_dir = raw_dir or RAW_DIR
    camera_cfg = camera_cfg or CameraConfig()

    session = CaptureSession()
    camera = Camera(camera_cfg)
    report = camera.open()

    roi = camera_cfg.roi
    message = ""
    message_until = 0.0
    show_window = display

    print(f"\n  {report.describe()}")

    if not report.resolution_ok:
        print(f"  CẢNH BÁO: xin {report.requested[0]}x{report.requested[1]} "
              f"nhưng driver trả {report.width}x{report.height}")

    print("  Khoá thông số:")
    for name, status in report.locks.items():
        mark = "  " if status.startswith("OK") else "!!"
        print(f"   {mark} {name:26s} {status}")

    if not report.all_locked:
        print("\n  CẢNH BÁO: có thông số không khoá được. Ảnh sẽ trôi theo thời")
        print("  gian, và model học buổi sáng sẽ sai vào buổi chiều.")

    exposure_warning = camera_cfg.exposure_warning()
    if exposure_warning:
        print(f"\n  CẢNH BÁO: {exposure_warning}")

    print(f"\n  Lưu vào: {raw_dir}")
    _print_help()

    last_time = time.perf_counter()
    fps = 0.0
    brightness = 128.0

    try:
        while True:
            try:
                frame = camera.read()
            except CameraError as exc:
                print(f"\n  LỖI CAMERA: {exc}")
                break

            now = time.perf_counter()
            delta = now - last_time
            last_time = now
            if delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())

            if show_window:
                view = draw_overlay(frame, session, fps, brightness, camera_cfg, roi,
                                    message if now < message_until else "")
                try:
                    cv2.imshow(WINDOW, view)
                except cv2.error as exc:
                    print(f"\n  Không mở được cửa sổ xem trước ({exc}).")
                    print("  Chuyển sang chế độ chỉ dùng bàn phím.")
                    show_window = False

            key = _wait_key(show_window)
            if key is None:
                continue

            label = None
            if key in (ord(" "), ord("1")):
                label = "OK"
            elif key in (ord("2"), ord("x"), ord("X")):
                label = "NG"

            # --- chụp ---
            if label is not None:
                region = frame
                if roi is not None:
                    from .imageops import crop_roi

                    region = crop_roi(frame, roi)

                # Băm một lần, dùng cho cả kiểm trùng lẫn ghi nhớ.
                digest = dhash(region)
                duplicate = session.find_duplicate(digest, label)
                if duplicate:
                    session.skipped_duplicates += 1
                    message = f"TRUNG voi {duplicate} - khong luu"
                    message_until = time.perf_counter() + 1.5
                    continue

                part_id = session.next_part()
                saved = 0

                for sequence in range(burst):
                    shot = region if sequence == 0 else camera.read_roi()
                    path = save_frame(shot, label, part_id, sequence, raw_dir)
                    session.remember(label, dhash(shot), path.name)
                    session.recent.append(path)
                    session.last_label = label
                    saved += 1

                message = f"{label} #{part_id} - luu {saved} anh"
                message_until = time.perf_counter() + 1.0

            # --- xoá lần vừa rồi ---
            elif key in (ord("u"), ord("U")):
                removed = session.undo()
                message = f"da xoa {removed} anh"
                message_until = time.perf_counter() + 1.5

            # --- sản phẩm mới ---
            elif key in (ord("b"), ord("B")):
                session.next_part()
                message = f"sang san pham #{session.part_id + 1}"
                message_until = time.perf_counter() + 1.5

            # --- bật tắt ROI ---
            elif key in (ord("r"), ord("R")):
                if roi is None:
                    roi = (0.25, 0.15, 0.75, 0.85)
                    message = "bat ROI"
                else:
                    roi = None
                    message = "tat ROI"
                camera_cfg.roi = roi
                message_until = time.perf_counter() + 1.5

            # --- phơi sáng / độ lợi ---
            elif key in (ord("["), ord("]"), ord("-"), ord("=")):
                if key == ord("["):
                    camera_cfg.exposure -= 1.0
                elif key == ord("]"):
                    camera_cfg.exposure += 1.0
                elif key == ord("-"):
                    camera_cfg.gain = max(0.0, camera_cfg.gain - 16.0)
                else:
                    camera_cfg.gain = min(255.0, camera_cfg.gain + 16.0)

                camera.lock()
                message = f"phoi sang {camera_cfg.exposure:g} · do loi {camera_cfg.gain:g}"
                message_until = time.perf_counter() + 1.5

                if not camera_cfg.exposure_fits:
                    message = "phoi sang dai hon 1 khung hinh!"

            # --- thoát ---
            elif key in (ord("q"), 27):
                break

    finally:
        camera.close()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    _print_summary(session, min_ok, min_ng)
    return session


def _wait_key(show_window: bool) -> int | None:
    """Đọc phím. Có cửa sổ thì lấy từ OpenCV, không thì lấy từ bàn phím."""
    if show_window:
        key = cv2.waitKey(1) & 0xFF
        return key if key != 255 else None

    try:
        line = input("  [Enter]=OK  [n]=NG  [u]=xoá  [b]=sản phẩm mới  [q]=thoát > ")
    except (EOFError, KeyboardInterrupt):
        return ord("q")

    line = line.strip().lower()
    if line == "n":
        return ord("2")
    if line == "u":
        return ord("u")
    if line == "b":
        return ord("b")
    if line == "q":
        return ord("q")
    return ord(" ")


def _print_help() -> None:
    print("""
  Phím:
    SPACE hoặc 1   chụp vào lớp OK (tốt)
    2 hoặc X       chụp vào lớp NG (lỗi)
    u              xoá lần chụp vừa rồi
    b              sang sản phẩm mới
    r              bật/tắt vùng cắt ROI
    [ ]            giảm/tăng phơi sáng
    - =            giảm/tăng độ lợi
    q hoặc ESC     thoát

  CÁCH CHỤP CHO ĐÚNG — ba điều quan trọng nhất:
    1. Mỗi sản phẩm bấm MỘT lần. Đừng bấm 20 lần cho một chi tiết đang đứng yên.
    2. ĐỔI QUA ĐỔI LẠI giữa OK và NG trong cùng một lượt. Đừng chụp hết OK rồi
       mới chụp NG — làm vậy là dạy model nhận biết ánh sáng, không phải lỗi.
    3. Đa dạng hơn là nhiều hơn. 200 sản phẩm khác nhau tốt hơn 2000 khung hình
       của 20 sản phẩm.
""")


def _print_summary(session: CaptureSession, min_ok: int, min_ng: int) -> None:
    print(f"\n  Đã chụp: OK {session.counts['OK']} · NG {session.counts['NG']}")

    if session.skipped_duplicates:
        print(f"  Bỏ qua {session.skipped_duplicates} ảnh trùng")

    print()
    if session.counts["OK"] < min_ok:
        print(f"  Còn thiếu ảnh OK: {session.counts['OK']}/{min_ok}")
    if session.counts["NG"] < min_ng:
        print(f"  Còn thiếu ảnh NG: {session.counts['NG']}/{min_ng}")

    if session.counts["OK"] >= min_ok and session.counts["NG"] >= min_ng:
        print("  Đủ ảnh để huấn luyện. Bước tiếp:")
        print("    python -m vision_qc dataset      # xem lại dữ liệu")
        print("    python -m vision_qc train        # dạy model")


# ---------------------------------------------------------------------
# Xoá ảnh trùng trên cả thư mục
# ---------------------------------------------------------------------


def deduplicate(raw_dir: Path, tolerance: int = 2) -> tuple[int, int]:
    """Quét cả thư mục, xoá ảnh gần trùng. Trả về (đã xoá, còn lại)."""
    from .dataset import scan

    samples = scan(raw_dir, compute_stats=False)
    kept: dict[int, str] = {}
    removed = 0

    for sample in samples:
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        digest = dhash(image)
        duplicate_of = None
        for other_digest, other_name in kept.items():
            if hamming(digest, other_digest) <= tolerance:
                duplicate_of = other_name
                break

        if duplicate_of:
            sample.path.unlink()
            removed += 1
            print(f"  xoá {sample.path.name}  (trùng {duplicate_of})")
        else:
            kept[digest] = sample.path.name

    return removed, len(kept)
