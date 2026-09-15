"""
config.py — Các con số của cả dây chuyền nằm gọn trong một file.

Có ba nhóm, và chúng thuộc về ba người khác nhau:

    CameraConfig   — thợ chỉnh máy đổi (độ phân giải, phơi sáng, vùng cắt)
    LineConfig     — kỹ sư dây chuyền đổi (tốc độ băng tải, khoảng cách, van)
    PlcConfig      — lập trình viên PLC đổi (địa chỉ thiết bị, cổng mạng)

Sửa ở đây rồi chạy lại, không phải sửa trong code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------
# 1. CAMERA
# ---------------------------------------------------------------------


@dataclass
class CameraConfig:
    """Camera 2MP (Logitech C922 Pro Stream: 1920x1080 = 2,07 MP)."""

    index: int = 0

    # 1920x1080 ở 30 fps CHỈ chạy được với MJPG. Ở chế độ YUY2 thô, băng thông
    # USB 2.0 chỉ đủ cho ~5 fps tại độ phân giải này. Đây là lý do `fourcc`
    # tồn tại chứ không phải một tuỳ chọn cho vui.
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"

    # Số khung hình vứt đi lúc mở camera. Camera tự chỉnh phơi sáng trong
    # khoảng 0,5-1 giây đầu; giữ lại thì 10 ảnh đầu khác hẳn 10 ảnh sau.
    warmup_frames: int = 15

    # Khoá phơi sáng / cân trắng / lấy nét. Bật thì ảnh mới tái lập được —
    # xem `camera.lock()` để biết cái gì khoá được, cái gì không.
    lock: bool = True

    # Phơi sáng, đơn vị DirectShow: luỹ thừa cơ số 2 của giây.
    #     -5 = 1/32 s   -6 = 1/64 s   -7 = 1/128 s
    #
    # RÀNG BUỘC THẬT: phơi sáng không thể dài hơn một khung hình.
    # Ở 30 fps, khung hình là 1/30 s = 2^-4,9, nên **-5 là mức dài nhất dùng
    # được**. Đặt -4 hay -3 thì camera buộc phải hạ fps xuống, và băng tải sẽ
    # đi qua trước khi kịp chụp.
    exposure: float = -5.0

    # Độ lợi. Ưu tiên phơi sáng dài + độ lợi thấp hơn là phơi sáng ngắn + độ
    # lợi cao: độ lợi khuếch đại cả nhiễu cảm biến, mà nhiễu thì model cũng
    # học luôn. Đo trên C922: expo -5 gain 32 cho độ sáng 92; expo -6 gain 64
    # cho 108 nhưng nhiễu gấp đôi.
    gain: float = 32.0

    wb_temperature: float = 4600.0
    focus: float = 0.0
    autofocus: bool = False

    # Vùng cắt quanh sản phẩm, toạ độ tương đối 0..1. Sản phẩm luôn nằm đúng
    # một chỗ vì có cảm biến trigger, nên cắt được — và cắt là cách duy nhất
    # giữ lại chi tiết vết xước nhỏ. Cắt từ 1920x1080 còn 40% rồi đưa vào
    # model 192x192 thì vết xước to gấp 4 lần so với thu cả khung.
    # None = dùng cả khung.
    roi: tuple[float, float, float, float] | None = None

    # -- kiểm tra ràng buộc phơi sáng / fps ---------------------------

    @property
    def exposure_seconds(self) -> float:
        return 2.0 ** self.exposure

    @property
    def max_exposure(self) -> float:
        """Phơi sáng dài nhất mà tốc độ khung hình cho phép (cùng đơn vị)."""
        import math

        return math.log2(1.0 / max(self.fps, 1))

    @property
    def exposure_fits(self) -> bool:
        """Phơi sáng có nằm trong một khung hình không?

        Không nằm thì camera buộc phải hạ fps, và mọi con số thời gian trong
        `LineConfig` thành sai.
        """
        return self.exposure <= self.max_exposure + 1e-9

    def exposure_warning(self) -> str | None:
        if self.exposure_fits:
            return None
        return (
            f"phơi sáng {self.exposure:g} = 1/{1 / self.exposure_seconds:.0f} s "
            f"dài hơn một khung hình ở {self.fps} fps "
            f"(tối đa {self.max_exposure:.1f} = 1/{self.fps} s). "
            f"Camera sẽ tự hạ fps — giảm phơi sáng xuống {self.max_exposure:.0f} "
            f"hoặc tăng độ lợi."
        )


# ---------------------------------------------------------------------
# 2. DÂY CHUYỀN
# ---------------------------------------------------------------------


@dataclass
class LineConfig:
    """Thông số cơ khí — quyết định ngân sách thời gian cho một quyết định."""

    conveyor_speed_mm_s: float = 200.0
    """Tốc độ băng tải. 200 mm/s = 12 m/phút, chậm rãi và thực tế."""

    part_pitch_mm: float = 400.0
    """Khoảng cách giữa hai sản phẩm liên tiếp trên băng tải."""

    camera_to_reject_mm: float = 300.0
    """Từ điểm chụp tới điểm van thổi. Đây là quãng đường model có để suy nghĩ."""

    plc_scan_ms: float = 10.0
    """Chu kỳ quét PLC. Kết quả ghi xuống muộn nhất là sau một chu kỳ."""

    valve_delay_ms: float = 15.0
    """Van khí nén nhận tín hiệu tới lúc thực sự thổi."""

    reject_pulse_ms: int = 100
    """Độ dài xung van. PLC giữ, không phải PC."""

    @property
    def time_budget_ms(self) -> float:
        """Bao nhiêu mili-giây để chụp, suy luận và ghi kết quả xuống PLC.

        Quãng đường chia tốc độ, trừ đi phần PLC và van tiêu tốn. Vượt con số
        này thì sản phẩm lỗi đã đi qua van rồi — model đúng cũng vô nghĩa.
        """
        travel_ms = self.camera_to_reject_mm / self.conveyor_speed_mm_s * 1000.0
        return travel_ms - self.plc_scan_ms - self.valve_delay_ms

    @property
    def part_interval_s(self) -> float:
        """Bao lâu thì có một sản phẩm mới tới điểm chụp."""
        return self.part_pitch_mm / self.conveyor_speed_mm_s

    @property
    def parts_per_minute(self) -> float:
        """Năng suất dây chuyền, tính theo sản phẩm mỗi phút."""
        return 60.0 / self.part_interval_s if self.part_interval_s > 0 else 0.0


# ---------------------------------------------------------------------
# 3. PLC MITSUBISHI
# ---------------------------------------------------------------------


@dataclass
class PlcConfig:
    """Địa chỉ thiết bị trên PLC Mitsubishi (FX5U / Q series)."""

    backend: str = "sim"        # "sim" | "mc"
    host: str = "192.168.1.10"
    port: int = 5007            # SLMP TCP. Kiểm tra lại trong GX Works3.
    timeout_s: float = 2.0
    retries: int = 2

    # --- PC đọc, PLC ghi ---
    trigger: str = "M100"       # =1 khi sản phẩm đã tới đúng điểm chụp
    plc_ready: str = "M310"     # =1 khi PLC đang chạy, cho phép PC gửi kết quả

    # --- PC ghi, PLC đọc ---
    result_valid: str = "M200"  # PC bật lên khi D200/D201 đã có kết quả mới
    ng_flag: str = "M201"       # 1 = NG, phải loại sản phẩm này ra
    fault: str = "M202"         # 1 = PC hỏng/timeout, PLC tự xử theo cấu hình an toàn

    score: str = "D200"         # điểm NG, số nguyên 0..10000
    part_id: str = "D201"       # số thứ tự sản phẩm, lấy modulo 65536
    heartbeat: str = "D210"     # PC tăng đều; PLC canh, đứng yên = PC chết

    @property
    def heartbeat_period_s(self) -> float:
        """Nhịp tim tối thiểu. Chậm hơn 5 giây thì coi như PC mất kết nối."""
        return 5.0


# ---------------------------------------------------------------------
# 4. MODEL
# ---------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Kiến trúc CNN và cách học."""

    image_size: int = 192       # ảnh vuông đưa vào model
    grayscale: bool = False     # True thì nhẹ hơn 3 lần, nhưng mất thông tin màu

    # Số kênh ở tầng đầu, nhân đôi sau mỗi khối.
    #
    # 16 -> ~295 nghìn thông số, ~7 giây một epoch với 400 ảnh ở 128x128 trên
    # CPU. 32 -> ~1,18 triệu, chậm gấp bốn.
    #
    # Chọn 16 vì nó khớp với lượng dữ liệu thật: vài trăm tới vài nghìn ảnh.
    # Model lớn hơn dữ liệu thì chỉ học vẹt nhanh hơn, chứ không khôn hơn.
    width: int = 16
    dropout: float = 0.3

    # --- học ---
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 12          # dừng sớm nếu loss thi không giảm sau ngần này epoch
    seed: int = 0

    # --- ngưỡng quyết định ---
    # Điểm càng cao càng NG. Ngưỡng 0,5 là mặc định ngây thơ; `evaluate.py`
    # quét để tìm ngưỡng rẻ nhất theo giá tiền của hai loại sai.
    threshold: float = 0.5
    cost_false_reject: float = 1.0   # loại oan một sản phẩm tốt: mất bao nhiêu tiền
    cost_escape: float = 50.0        # để lọt một sản phẩm lỗi: mất bao nhiêu tiền

    # Trần cho tỉ lệ thổi oan, và đây KHÔNG phải con số thẩm mỹ.
    #
    # Đo được trên tập thi của dự án: với tỉ lệ chi phí 50:1, ngưỡng rẻ nhất
    # theo toán học là 0,044 — nó bắt được 100% lỗi và thổi oan **98,1%** sản
    # phẩm tốt. Toán học nói đúng: bỏ sót 5 sản phẩm lỗi đắt bằng thổi oan 250
    # sản phẩm tốt, mà ở đây chỉ phải thổi oan 51.
    #
    # Nhưng không dây chuyền nào chạy được như vậy. Thổi 98% sản phẩm ra thùng
    # phế nghĩa là dây chuyền dừng trong một giờ, và đó là khoản lỗ không nằm
    # trong mô hình chi phí hai số hạng.
    #
    # Nên ngưỡng phải rẻ nhất TRONG SỐ những ngưỡng vận hành được. 5% là mức
    # khởi đầu hợp lý cho hầu hết dây chuyền; siết xuống 1-2% nếu khách hàng
    # khó tính, nới lên 10% nếu sản phẩm rẻ.
    max_false_reject_rate: float = 0.05


# ---------------------------------------------------------------------
# Gộp lại
# ---------------------------------------------------------------------


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    line: LineConfig = field(default_factory=LineConfig)
    plc: PlcConfig = field(default_factory=PlcConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            line=LineConfig(**data.get("line", {})),
            plc=PlcConfig(**data.get("plc", {})),
            model=ModelConfig(**data.get("model", {})),
        )
