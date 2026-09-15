"""
imageops.py — Cắt, băm, và biến ảnh thành số.

File này tồn tại vì một lý do duy nhất: **huấn luyện và chạy thật phải tiền xử
lý giống hệt nhau**. Nếu lúc dạy model ảnh được resize bằng OpenCV còn lúc chạy
thật lại resize bằng PIL, model sẽ nhận hai loại ảnh hơi khác nhau và độ chính
xác tụt mà không có lỗi nào báo. Cách chữa: chỉ có MỘT hàm `preprocess()` trong
cả dự án, cả `train.py` lẫn `station.py` đều gọi nó.
"""

from __future__ import annotations

import cv2
import numpy as np

# Chuẩn hoá bằng hằng số cố định, KHÔNG dùng trung bình của tập huấn luyện.
# Dùng thống kê tập huấn luyện thì lúc chạy thật phải mang theo hai con số đó,
# và đổi tập dữ liệu là mọi model cũ hỏng. 0,5 là trung điểm, đủ tốt.
MEAN = 0.5
STD = 0.5


# ---------------------------------------------------------------------
# Cắt vùng quan tâm
# ---------------------------------------------------------------------


def crop_roi(frame: np.ndarray, roi: tuple[float, float, float, float] | None) -> np.ndarray:
    """Cắt vùng quan tâm, toạ độ tương đối 0..1. `None` thì trả lại nguyên ảnh.

    Sản phẩm nằm đúng một chỗ vì có cảm biến trigger, nên cắt được. Đây không
    phải chuyện thẩm mỹ: cắt 40% khung rồi mới thu nhỏ về 192x192 nghĩa là
    vết xước nhỏ chiếm **4 lần** số điểm ảnh so với thu cả khung.
    """
    if roi is None:
        return frame

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi
    left, right = int(round(x0 * w)), int(round(x1 * w))
    top, bottom = int(round(y0 * h)), int(round(y1 * h))

    left, right = max(0, min(left, w - 1)), max(left + 1, min(right, w))
    top, bottom = max(0, min(top, h - 1)), max(top + 1, min(bottom, h))
    return frame[top:bottom, left:right]


# ---------------------------------------------------------------------
# Tiền xử lý — chỉ một bản duy nhất cho cả dạy lẫn chạy
# ---------------------------------------------------------------------


def preprocess(
    image: np.ndarray,
    image_size: int = 192,
    grayscale: bool = False,
    fit: str = "crop",
) -> np.ndarray:
    """Ảnh BGR (bất kỳ cỡ nào) -> mảng float32 shape (C, H, W), giá trị -1..1.

    Đây là ranh giới duy nhất giữa "ảnh" và "số" trong dự án.

    `fit="crop"` (mặc định) cắt vuông ở giữa rồi mới thu nhỏ. `fit="squash"`
    ép thẳng về vuông — ĐỪNG DÙNG, chỉ giữ lại để so sánh.

    VÌ SAO `crop` LÀ MẶC ĐỊNH
    -------------------------
    Camera cho ảnh 1920x1080, model cần ảnh vuông. Ép thẳng về vuông nén trục
    ngang 128/1280 = 0,1 lần còn trục dọc 128/720 = 0,178 lần — hai trục bị
    nén LỆCH NHAU 1,78 lần. Đo trên một hình tròn bán kính 200 điểm ảnh:

        ép thẳng về vuông   40 x 71 px    méo 43,7%
        cắt vuông giữa      71 x 71 px    méo  0,0%

    Và cái giá không chỉ là hình dạng: một vết xước 3 điểm ảnh sau khi thu nhỏ
    còn 115 điểm ảnh đủ tương phản nếu ép vuông, nhưng còn **195** nếu cắt
    vuông — mất gần một nửa tín hiệu của đúng thứ cần tìm.

    Cắt được là nhờ cảm biến trigger: sản phẩm luôn nằm ở giữa khung.
    """
    if fit == "squash":
        resize_to = (image_size, image_size)
    else:
        height, width = image.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        image = image[top:top + side, left:left + side]
        resize_to = (image_size, image_size)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # INTER_AREA khi thu nhỏ: nó lấy trung bình vùng, giữ chi tiết tốt hơn
    # INTER_LINEAR. Với ảnh 2MP thu về 192 thì đây là lựa chọn đúng.
    interpolation = cv2.INTER_AREA if image.shape[0] > image_size else cv2.INTER_LINEAR

    if image.shape[0] != resize_to[1] or image.shape[1] != resize_to[0]:
        image = cv2.resize(image, resize_to, interpolation=interpolation)

    array = image.astype(np.float32) / 255.0
    array = (array - MEAN) / STD

    if array.ndim == 2:
        array = array[:, :, None]

    return np.ascontiguousarray(array.transpose(2, 0, 1))  # HWC -> CHW


# ---------------------------------------------------------------------
# Băm ảnh — để phát hiện ảnh trùng
# ---------------------------------------------------------------------


def dhash(image: np.ndarray, size: int = 8) -> int:
    """Băm khác biệt hai chiều, 128 bit. Ảnh giống nhau -> băm giống nhau.

    Dùng để chặn hai thứ rất hay xảy ra khi thu ảnh bằng tay:

    1. Bấm nhầm hai lần cho cùng một sản phẩm đang đứng yên.
    2. Bấm liên tục trong lúc băng tải chạy, ra 50 ảnh gần như y hệt của cùng
       một sản phẩm.

    Cả hai đều làm tập dữ liệu phình ra mà không thêm thông tin gì, và tệ hơn:
    nếu ảnh trùng rơi vào cả tập học lẫn tập thi thì điểm thi bị thổi lên.

    VÌ SAO HAI CHIỀU
    ----------------
    Băm khác biệt cổ điển chỉ so mỗi điểm ảnh với điểm BÊN PHẢI nó. Hệ quả
    không phải là "nhầm ảnh này với ảnh kia", mà tệ hơn: **mọi ảnh chỉ biến
    thiên theo chiều dọc đều cho ra đúng một giá trị băm là 0.** Mỗi hàng phẳng
    thì không có chênh lệch ngang nào để ghi lại, nên toàn bộ thông tin về
    chiều dọc biến mất.

    Trên sản phẩm công nghiệp, ảnh chỉ biến thiên theo chiều dọc là chuyện
    thường: sản phẩm sáng trên nền tối, vết xước ngang, bóng đổ ngang. Tất cả
    chúng đều băm ra 0 và bị coi là ảnh trùng của nhau.

    Nên băm ở đây gồm 128 bit chênh lệch (64 ngang + 64 dọc), cộng thêm 8 bit
    độ sáng trung bình.

    VÌ SAO CẦN THÊM 6 BIT ĐỘ SÁNG
    ------------------------------
    Băm khác biệt chỉ ghi lại chênh lệch giữa các điểm ảnh LÂN CẬN, không ghi
    lại mức sáng tuyệt đối. Đó là ưu điểm — ảnh cùng một sản phẩm chụp lệch
    sáng một chút vẫn được coi là trùng nhau.

    Nhưng nó có một điểm mù: một ảnh ĐỒNG NHẤT thì mọi phép so sánh đều sai,
    nên băm bằng 0 — bất kể sáng hay tối. Nghĩa là ảnh xám 40 và ảnh xám 240
    bị coi là cùng một tấm.

    Hệ quả thật: nếu sản phẩm lỗi chỉ khác sản phẩm tốt ở chỗ **tối hơn trên
    toàn bộ bề mặt** (một vết bẩn lớn, một lần xử lý bề mặt sai), công cụ thu
    ảnh sẽ tưởng đó là ảnh trùng và **bỏ luôn không chụp**. Mẫu lỗi biến mất
    khỏi tập dữ liệu, và không có gì báo động.

    8 bit cuối là độ sáng trung bình (0–255). Lệch nhau vài mức xám vì nhiễu
    hay nén JPEG thì vẫn coi là trùng; lệch hơn thì không.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ngang: so từng điểm với điểm bên phải
    horizontal = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits_h = (horizontal[:, 1:] > horizontal[:, :-1]).flatten()

    # Dọc: so từng điểm với điểm bên dưới
    vertical = cv2.resize(gray, (size, size + 1), interpolation=cv2.INTER_AREA)
    bits_v = (vertical[1:, :] > vertical[:-1, :]).flatten()

    value = 0
    for bit in np.concatenate([bits_h, bits_v]):
        value = (value << 1) | int(bit)

    # 8 bit độ sáng trung bình, để ảnh đồng nhất khác mức sáng không bị coi
    # là cùng một tấm. 8 bit chứ không phải 6: với 6 bit (chia 4) thì hai ảnh
    # xám 140 và 190 chỉ lệch nhau 2 bit, vẫn nằm trong dung sai và vẫn bị
    # coi là trùng — đo được khi chạy test.
    value = (value << 8) | int(np.clip(round(float(gray.mean())), 0, 255))
    return value


def hamming(a: int, b: int) -> int:
    """Số bit khác nhau giữa hai giá trị băm."""
    return int((a ^ b).bit_count())
