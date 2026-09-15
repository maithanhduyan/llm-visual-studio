# Thu dữ liệu bằng camera 2MP

Đây là bước quyết định cả dự án. Model không thể tốt hơn dữ liệu dạy nó, và
trong thị giác công nghiệp thì ba lỗi dưới đây phổ biến tới mức đáng gọi là
quy luật. Cả ba đều làm điểm kiểm tra **đẹp giả tạo**, và cả ba đều chỉ lộ ra
khi đã ra tới xưởng.

---

## Chuẩn bị chỗ chụp

Trước khi bấm nút nào, chốt bốn thứ. Đổi một trong bốn thứ này sau khi đã chụp
thì phải chụp lại từ đầu.

| Thứ | Đặt thế nào | Vì sao |
| --- | --- | --- |
| **Khoảng cách camera → sản phẩm** | Cố định, có giá đỡ cứng | Đổi khoảng cách là đổi tỉ lệ ảnh, model học thừa |
| **Đèn** | Cố định, **không đổi giữa hai lớp** | Xem bẫy 2 |
| **Nền** | Cố định, tương phản với sản phẩm | Nền đổi là model học nền |
| **Vị trí sản phẩm** | Có chặn cơ khí hoặc cảm biến trigger | Sản phẩm luôn nằm một chỗ thì mới cắt ROI được |

Đèn nên là **đèn LED một chiều**, không phải đèn huỳnh quang. Đèn huỳnh quang
nhấp nháy 50 Hz; ở 30 fps, mỗi khung hình rơi vào một pha khác nhau của lưới
điện nên độ sáng dao động theo chu kỳ.

Kiểm tra bằng:

```bash
python -m vision_qc camera
```

Mục **độ ổn định** phải nhỏ hơn 1%. Lớn hơn thì đọc phần cảnh báo trong output.

---

## Bẫy 1 — Nhiều khung hình của CÙNG MỘT sản phẩm

Bấm 20 lần cho một chi tiết đang đứng yên. Được 20 ảnh, nhưng chỉ có **một**
sản phẩm. Model học thuộc lòng chi tiết đó, và 20 ảnh đó còn làm hỏng tập thi
nếu chúng rơi vào cả hai phía.

**Cách chặn:** mỗi lần bấm là một sản phẩm mới. `capture.py` tự tăng `part_id`
và `dataset.py` chia tập theo `part_id`, nên cả cụm ảnh đi cùng một phía.

**Muốn nhiều ảnh cho một sản phẩm?** Đặt sản phẩm lên, bấm một lần. Nhấc ra,
đặt lại **hơi lệch**, bấm lần nữa. Giờ là hai sản phẩm — và đó mới là dữ liệu
thật, vì nó dạy model rằng vị trí đặt không quan trọng.

---

## Bẫy 2 — Hai lớp chụp trong hai điều kiện sáng khác nhau

Đây là bẫy tốn kém nhất, vì nó **không lộ ra ở bất kỳ con số nào** cho tới khi
ra xưởng.

```text
  buổi sáng:  chụp 200 ảnh OK
  buổi chiều: chụp 40 ảnh NG
  -> model đạt 99% trên tập thi
  -> ra xưởng: model đoán "sáng = tốt", và mọi sản phẩm lỗi buổi chiều đi lọt
```

Model chưa hề nhìn vào sản phẩm. Nó học đúng một điều: ảnh sáng là OK.

**Cách chặn:** trong **cùng một lượt chụp**, đổi qua đổi lại giữa OK và NG.

```text
  OK, OK, NG, OK, NG, NG, OK, NG, OK, OK, NG, ...
```

Không nhất thiết phải luân phiên hoàn hảo, nhưng **đừng bao giờ** chụp hết lớp
này rồi mới sang lớp kia.

**Kiểm tra bằng:**

```bash
python -m vision_qc dataset
```

Dòng `lệch` phải dưới 5%. Trên 5% thì `train.py` và `check` đều cảnh báo, và
cách chữa **không nằm ở model** — phải chụp lại.

---

## Bẫy 3 — Ảnh trùng và ảnh hỏng

Bấm nhầm hai lần, hoặc bấm liên tục trong lúc băng tải chạy.

`capture.py` băm mọi ảnh và chặn ảnh gần giống ngay lúc chụp. Sau khi chụp
xong vẫn nên quét lại:

```bash
python -m vision_qc dedup
```

---

## Chụp bao nhiêu là đủ?

Với CNN huấn luyện từ đầu, con số thực tế:

| | Tối thiểu | Nên có |
| --- | ---: | ---: |
| Sản phẩm **tốt** | 150 | 400+ |
| Sản phẩm **lỗi** | 50 | 150+ |
| **Sản phẩm khác nhau** | 200 | 550+ |

Chú ý chữ **sản phẩm**, không phải **ảnh**. 200 sản phẩm khác nhau tốt hơn
2000 khung hình của 20 sản phẩm.

Nếu loại lỗi có nhiều dạng (xước, móp, thiếu chi tiết), mỗi dạng cần ít nhất
30–50 sản phẩm. Nếu một dạng chỉ có 5 mẫu, model sẽ không học được dạng đó, và
tệ hơn: nó có thể học "dạng lỗi hiếm = OK" vì trong tập học dạng đó gần như
không xuất hiện.

### Khi sản phẩm lỗi quá ít

Trong xưởng thật, sản phẩm lỗi thường **rất** ít — đó chính là mục đích của
việc kiểm tra. Lúc đó có hai đường:

1. **Cố tình làm hỏng.** Lấy sản phẩm tốt và tạo lỗi giống thật: cào xước, gõ
   móp, bôi bẩn. Ghi lại rằng đây là lỗi nhân tạo — nó không giống lỗi thật
   100%, nhưng 50 mẫu nhân tạo vẫn tốt hơn 5 mẫu thật.
2. **Chỉ học từ ảnh tốt** (phát hiện bất thường). Chỉ dạy model "trông thế nào
   là bình thường", và bất cứ thứ gì khác là lỗi. Cách này không cài đặt trong
   dự án này, nhưng là hướng đáng làm khi lớp lỗi thật sự chỉ có vài mẫu.

---

## Quy trình

```bash
# 1. Kiểm tra chỗ chụp
python -m vision_qc camera

# 2. Chụp. Đổi qua đổi lại OK / NG trong cùng một lượt.
python -m vision_qc capture

# 3. Xem lại dữ liệu trước khi dạy
python -m vision_qc dataset

# 4. Nếu có ảnh trùng
python -m vision_qc dedup

# 5. Dạy
python -m vision_qc train
```

### Phím khi chụp

```text
SPACE hoặc 1   chụp vào lớp OK (tốt)
2 hoặc X       chụp vào lớp NG (lỗi)
u              xoá lần chụp vừa rồi
b              sang sản phẩm mới
r              bật/tắt vùng cắt ROI
[ ]            giảm/tăng phơi sáng
- =            giảm/tăng độ lợi
q hoặc ESC     thoát
```

### Bật ROI khi nào

Bật ROI (`r`) khi sản phẩm chỉ chiếm một phần khung. Xem
[README mục "Độ phân giải"](../README.md#độ-phân-giải-vết-xước-2-điểm-ảnh-không-tồn-tại)
— vết xước mảnh **không thể** tìm thấy nếu không cắt.

Sau khi bật ROI, `capture.py` lưu ảnh **đã cắt**. Nhớ ghi lại giá trị ROI vào
cấu hình, vì lúc chạy thật `camera.py` phải cắt đúng như vậy:

```python
# config.py
roi = (0.25, 0.15, 0.75, 0.85)
```

Hoặc ghi đè bằng tham số:

```bash
python -m vision_qc capture --roi 0.25 0.15 0.75 0.85
```

---

## Sau khi chụp xong

```bash
python -m vision_qc train       # dạy
python -m vision_qc eval        # đo, và tìm ngưỡng
python -m vision_qc station --source camera --plc sim    # chạy thử, chưa nối PLC
python -m vision_qc station --source camera --plc mc --host 192.168.1.10
```

Bước thứ tư đáng làm trước khi nối PLC thật: chạy với `--plc sim` và **đưa
sản phẩm thật qua camera**, xem nó thổi những gì. Chỉ khi nào số lần thổi oan
chấp nhận được thì mới nối PLC.
