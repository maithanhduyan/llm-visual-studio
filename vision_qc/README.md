# vision_qc — Camera 2MP nhìn ra sản phẩm lỗi, PLC Mitsubishi thổi nó ra

Một dây chuyền: camera chụp sản phẩm, một mạng tích chập **huấn luyện từ đầu**
chấm điểm nó, và một PLC Mitsubishi nhận quyết định để thổi sản phẩm lỗi ra
khỏi băng tải.

Không dùng model có sẵn. Không dùng thư viện PLC. PyTorch, NumPy, OpenCV — hết.

```bash
cd vision_qc
pip install -r ../requirements.txt
pip install opencv-python numpy

python -m vision_qc check      # kiểm tra toàn hệ thống, 10 mục có số đo
python -m vision_qc camera     # camera có gì, đo fps thật
python -m vision_qc capture    # THU ẢNH
python -m vision_qc train      # dạy model
python -m vision_qc eval       # đo, và tìm ngưỡng rẻ nhất
python -m vision_qc station    # chạy cả trạm, điều khiển PLC
```

Chưa có camera hay PLC vẫn chạy thử được toàn bộ:

```bash
python -m vision_qc synth --ok 200 --ng 200        # sinh ảnh sản phẩm giả
python -m vision_qc train --data data/synth --name synth
python -m vision_qc station --name synth --source synth --parts 30
```

Tài liệu khác:

- [`docs/thu-thap-du-lieu.md`](docs/thu-thap-du-lieu.md) — **đọc trước khi
  chụp tấm ảnh đầu tiên.** Ba cái bẫy làm hỏng dữ liệu, và cách chặn.
- [`docs/giao-thuc-mc.md`](docs/giao-thuc-mc.md) — tham chiếu giao thức Mitsubishi,
  đủ để tự viết lại hoặc để gỡ lỗi khi cắm vào PLC thật.

---

## Đây không phải LLM

Yêu cầu ban đầu là "xây dựng LLM bằng Python từ đầu". Bài toán này không phải
bài toán ngôn ngữ, nên thứ được xây ở đây là **CNN**, không phải transformer.

Phần "từ đầu" thì giữ nguyên vẹn, và đó mới là phần đáng nói:

| | |
| --- | --- |
| Không dùng pretrained | Trọng số khởi tạo bằng Kaiming, học từ 0 |
| Không dùng torchvision | Tăng cường ảnh viết tay bằng `affine_grid` + `grid_sample` |
| Không dùng thư viện PLC | Giao thức MC viết trên socket thô, từng byte |
| Không dùng sklearn | AUC tính bằng thống kê hạng |

Tổng cộng **5.756 dòng Python**, trong đó **4.065 dòng code**, 679 dòng
docstring và 388 dòng chú thích — cộng thêm 660 dòng test. Phần lớn docstring
giải thích **vì sao**, không phải **cái gì**: mỗi con số và mỗi lựa chọn thiết
kế đều kèm phép đo đã dẫn tới nó.

---

## Đường đi của một sản phẩm

```text
   cảm biến trigger
        |
        v
   [camera.py]      chụp 1920x1080, phơi sáng đã khoá      ~35 ms
        |
        v
   [imageops.py]    cắt vuông giữa -> 192x192 -> -1..1     ~2 ms
        |
        v
   [model.py]       CNN 296k thông số -> P(lỗi)            ~10 ms
        |
        v
   [evaluate.py]    so với ngưỡng -> OK / NG
        |
        v
   [plc.py]         ghi D200 + M201, bật M200, chờ ack     ~12 ms
        |
        v
   [PLC]            giữ xung van 100 ms -> THỔI
```

Ngân sách: **1475 ms**. Đã dùng: **33 ms**. Tỉ lệ: **2,3%**.

Con số 1475 ms đến từ đâu:

```text
khoảng cách camera -> van   300 mm
tốc độ băng tải             200 mm/s
                            --------
thời gian di chuyển        1500 ms
trừ chu kỳ quét PLC          10 ms
trừ độ trễ van               15 ms
                            --------
ngân sách                  1475 ms
```

Năng suất: **30 sản phẩm/phút**.

---

## Kết quả đo được

Trên **dữ liệu giả** (520 ảnh, 5 loại lỗi, chia theo sản phẩm). Nhắc lại: ảnh
giả không nói lên điều gì về độ chính xác trong xưởng — nó chỉ chứng minh
đường ống thông và model học được.

| | |
| --- | ---: |
| Thông số | 296.465 |
| Học | 416 ảnh / 208 sản phẩm |
| Thi | 104 ảnh / 52 sản phẩm |
| Thời gian học | 45 epoch × 14 s ≈ **10,5 phút** trên CPU |
| Loss học (epoch cuối) | 0,164 |
| Loss thi (tốt nhất) | **0,175** |
| **Đúng trên tập thi** | **94,2%** |
| **AUC** | **0,9723** |

Loss học 0,164 và loss thi 0,175 gần như trùng nhau — model **không học vẹt**.
Đây là điều đáng chú ý: 416 ảnh là rất ít cho một CNN huấn luyện từ đầu, và
cách để không học vẹt nằm ở ba chỗ — tăng cường ảnh, dropout, và dừng sớm theo
loss tập thi.

### Ở ngưỡng 0,5 (mặc định ngây thơ)

```text
đúng 94,2%   bắt được 90,4% lỗi   thổi oan 1,9%
bỏ sót 5 lỗi · loại oan 1 sản phẩm tốt
```

### Ở ngưỡng tìm được

```text
ngưỡng 0,3309   đúng 96,2%   bắt được 94,2% lỗi   thổi oan 1,9%
bỏ sót 3 lỗi · loại oan 1 sản phẩm tốt
tiền mất mỗi sản phẩm: 1,452  (so với 2,413 ở ngưỡng 0,5)
```

Cùng một model, cùng một dữ liệu, chỉ đổi ngưỡng: **bỏ sót giảm từ 5 xuống 3**
mà số lần thổi oan không tăng. Đó là toàn bộ giá trị của việc quét ngưỡng.

### Chạy cả trạm (40 sản phẩm, đọc ảnh từ đĩa)

| | |
| --- | ---: |
| Kiểm tra | 40 sản phẩm · thổi 24 (60,0%) |
| Đúng | 38 (**95,0%**) |
| **Bỏ sót** | 1 (**2,5%**) |
| Thổi oan | 1 (2,5%) |

Thời gian mỗi sản phẩm:

| Công đoạn | Trung bình |
| --- | ---: |
| Chụp (đọc JPEG từ đĩa) | 4,8 ms |
| Tiền xử lý | 2,0 ms |
| **Suy luận** | **13,1 ms** |
| Ghi PLC | 12,3 ms |
| **TỔNG p95** | **33,3 ms** |

**33,3 ms trên ngân sách 1475 ms — dùng 2,3%.** Camera thật sẽ thay công đoạn
chụp bằng ~35 ms, đưa tổng lên khoảng 65 ms, tức 4,4%. Vẫn còn thừa rất nhiều:
băng tải có thể chạy nhanh gấp 20 lần trước khi thời gian thành vấn đề.

Chạy với ảnh giả sinh trực tiếp thì p95 là 243 ms, vì công đoạn "chụp" phải vẽ
một ảnh 1280×720 từ đầu. Con số đó không nói lên điều gì về hệ thống thật.

---

## Camera 2MP: ba phát hiện đo được

Camera Logitech C922 Pro Stream, 1920×1080 = 2,07 MP. Cắm vào, mở lên, và ba
điều dưới đây lộ ra — cả ba đều là loại không đọc tài liệu mà biết được.

### 1. Thứ tự đặt tham số quyết định 5,6 lần tốc độ

| Backend | Thứ tự | FOURCC nhận được | fps đo được |
| --- | --- | --- | ---: |
| DSHOW | đặt FOURCC trước | `YUY2` | **4,7** |
| DSHOW | **đặt độ phân giải trước** | **`MJPG`** | **26,2** |
| DSHOW | đặt độ phân giải trước (đo lại) | `MJPG` | **30,1** |

Ở chế độ YUY2 thô, USB 2.0 không đủ băng thông cho 1080p nên camera **tự hạ
xuống ~5 fps mà không báo lỗi gì**. Băng tải chạy 200 mm/s thì 5 fps nghĩa là
mỗi khung hình cách nhau 40 mm — sản phẩm đã đi qua van trước khi chụp xong.

`camera.py` đặt độ phân giải trước, FOURCC sau. Đảo hai dòng là mất 5,6 lần.

### 2. Phơi sáng không thể dài hơn một khung hình

Ở 30 fps, khung hình là 1/30 s = 2^−4,9. Đặt phơi sáng −4 hay −3 thì camera
buộc phải hạ fps, và mọi con số thời gian ở trên thành sai.

Quét thật (độ sáng trung bình của ảnh, thang 0–255):

| phơi sáng | độ lợi | độ sáng | cháy sáng |
| ---: | ---: | ---: | ---: |
| −6 | 0 | 53,7 | 0% |
| −6 | 64 | 107,9 | 0% |
| **−5** | **32** | **91,9** | **0%** |
| −5 | 64 | 170,6 | 0,01% |
| −4 | 64 | 219,5 | **48,8%** |
| −3 | 96 | 235,1 | **76,4%** |

Chọn **−5 / 32**: sát trần cho phép, chưa cháy sáng, và độ lợi thấp.

Vì sao không chọn −6 / 64 cho cùng độ sáng? Vì **độ lợi khuếch đại cả nhiễu
cảm biến**, mà nhiễu thì model cũng học luôn. Cùng độ sáng thì luôn ưu tiên
phơi sáng dài hơn và độ lợi thấp hơn.

### 3. Driver không phải lúc nào cũng thật tháo

`camera.lock()` đặt 7 thông số rồi **đọc lại từng cái**. Kết quả trên C922:

```text
   OK   tắt cân trắng tự động      OK (0)
   OK   phơi sáng                  OK (-5)
   OK   độ lợi                     OK (32)
   OK   lấy nét                    OK (0)
   ??   tắt lấy nét tự động        đặt được, không đọc lại được (driver trả 2)
   ??   tắt phơi sáng tự động      đặt được, không đọc lại được (driver trả -1)
   !!   nhiệt độ màu               BỊ TỪ CHỐI (đọc lại -1)
```

DirectShow nhận lệnh rồi trả về −1 vì không có thang đo để đọc. Điều đó
**không** có nghĩa là lệnh bị bỏ qua — nhưng cũng không có nghĩa là nó có tác
dụng. Không phân biệt được từ xa.

Nên phép thử thật không phải là hỏi driver, mà là **đo hậu quả**:
`camera.verify_lock()` chĩa vào một cảnh đứng yên và xem độ sáng có đứng yên
không.

```text
độ sáng ổn định (0,050%) — khoá có tác dụng
driver từ chối: nhiệt độ màu
```

0,05% là đứng yên. Phơi sáng còn tự động thì con số này sẽ trôi vài phần trăm.
Cột `locks` chỉ là thông tin tham khảo; `verify_lock` mới là bằng chứng.

---

## Ba cái bẫy dữ liệu, và cách chúng được chặn

Một model không thể tốt hơn dữ liệu dạy nó. Trong thị giác công nghiệp, ba lỗi
dưới đây phổ biến tới mức đáng gọi là quy luật.

### Bẫy 1 — Rò rỉ giữa tập học và tập thi

Chụp 5 khung hình của cùng một sản phẩm đang đứng yên, rồi chia ngẫu nhiên 4
tấm vào tập học và 1 tấm vào tập thi. Năm tấm đó **giống nhau tới từng điểm
ảnh**. Model chỉ cần học thuộc lòng là được 100%.

`dataset.py` chia theo **sản phẩm** (`part_id`), không chia theo ảnh. Mỗi lần
bấm nút trong `capture.py` là một sản phẩm mới, và cả 5 khung hình của nó đi
cùng một tập.

Có test giữ bất biến này: `test_chia_tap_khong_ro_ri_san_pham`.

### Bẫy 2 — Model học ánh sáng thay vì học lỗi

Chụp 200 ảnh tốt buổi sáng, 40 ảnh lỗi buổi chiều. Model đạt 99% trên tập thi
và bằng 0 ngoài xưởng, vì nó học "sáng = tốt".

`dataset.py` đo `brightness_gap()`. Lệch quá 5% thì `train.py`, `dataset` và
`check` đều cảnh báo. Cách chữa không nằm ở model — phải **chụp lại, trộn hai
lớp trong cùng một lượt**.

Bộ sinh ảnh giả cũng tôn trọng điều này: độ sáng được lấy ngẫu nhiên **độc lập
với nhãn**. Đo được: lệch **0,72%**.

### Bẫy 3 — Ảnh trùng

Bấm nhầm hai lần cho cùng một sản phẩm. `imageops.dhash()` băm ảnh thành 128
bit và chặn ảnh gần giống.

Băm ở đây **hai chiều**, không phải một chiều như bản cổ điển. Bản cổ điển chỉ
so mỗi điểm ảnh với điểm bên phải, nên **mọi ảnh chỉ biến thiên theo chiều dọc
đều cho ra đúng một giá trị băm là 0** — mỗi hàng phẳng thì không có chênh
lệch ngang nào để ghi lại. Sản phẩm sáng trên nền tối, vết xước ngang, bóng đổ
ngang: tất cả đều băm ra 0 và bị coi là ảnh trùng của nhau.

---

## Độ phân giải: vết xước 2 điểm ảnh không tồn tại

Đây là phát hiện quan trọng nhất của dự án, và nó không liên quan gì tới model.

Camera cho ảnh 1920×1080. Model cần ảnh vuông 192×192. Giữa hai chỗ đó có một
phép thu nhỏ **5,6 lần**, và mọi chi tiết nhỏ hơn 5,6 điểm ảnh sẽ biến mất.

Đo cụ thể trên sản phẩm bán kính 198 điểm ảnh:

| vết xước rộng | còn lại ở đầu vào model | kết quả |
| ---: | ---: | --- |
| 2 px | 0,5 px | **không thể thấy** |
| 8 px | 2,1 px | thấy được |

Và đo trên ảnh giả, sau khi tiền xử lý đúng cách:

| loại lỗi | diện tích đổi | đánh giá |
| --- | ---: | --- |
| `dent` (móp) | 2.487 px | thấy rõ |
| `stain` (bẩn) | 3.795 px | thấy rõ |
| `spot` (đốm) | 1.062 px | thấy được |
| `scratch` (xước) | 396 px | thấy được |
| `chip` (sứt rìa) | 195 px | vừa đủ |

**Kết luận cho dây chuyền thật:** muốn tìm vết xước mảnh thì phải **cắt ROI**
quanh sản phẩm trước khi thu nhỏ. Cắt còn 40% khung rồi mới thu về 192 làm vết
xước to lên **2,5 lần** so với thu cả khung. Đó là toàn bộ lý do `CameraConfig.roi`
tồn tại — không phải để cho gọn ảnh.

---

## Cắt vuông giữa, không ép vuông

Camera cho 16:9, model cần 1:1. Ép thẳng về vuông nén trục ngang
`128/1280 = 0,1` lần còn trục dọc `128/720 = 0,178` lần — hai trục lệch nhau
**1,78 lần**.

Đo trên một hình tròn bán kính 200 điểm ảnh:

| cách | kết quả | méo |
| --- | --- | ---: |
| ép thẳng về vuông | 40 × 71 px | **43,7%** |
| **cắt vuông giữa rồi thu nhỏ** | 71 × 71 px | **0%** |

Và cái giá không chỉ là hình dạng. Một vết xước 3 điểm ảnh sau khi thu nhỏ còn
**115** điểm ảnh đủ tương phản nếu ép vuông, nhưng còn **195** nếu cắt vuông —
mất gần một nửa tín hiệu của đúng thứ cần tìm.

Có test giữ bất biến này: `test_cat_vuong_KHONG_lam_meo_hinh_hoc`.

---

## Model

```text
192x192x3
  -> 96x96x16     khối 1
  -> 48x48x32     khối 2
  -> 24x24x64     khối 3
  -> 12x12x128    khối 4
  -> gộp trung bình toàn cục -> 128 số -> 1 logit -> sigmoid -> P(lỗi)
```

**296.465 thông số**, khoảng 14 giây một epoch với 416 ảnh ở 192×192 trên CPU
(máy đo: Quadro T2000 nhưng PyTorch bản CPU — xem mục "Chạy bằng GPU").

### Vì sao một logit chứ không phải hai lớp

Hai lớp + softmax cho ra "OK 97% / NG 3%". Nhưng dây chuyền cần một câu trả lời
khác: *"đặt ngưỡng ở đâu để tốn ít tiền nhất"*. Ngưỡng nằm ở đâu trong không
gian softmax? Không rõ ràng.

Một logit + sigmoid cho ra thẳng P(lỗi) — một con số chạy liên tục từ 0 tới 1.
`evaluate.py` quét con số đó để tìm ngưỡng rẻ nhất.

### Vì sao CNN chứ không phải ViT

ViT cần nhiều dữ liệu hơn và nhiều thời gian hơn để tới cùng độ chính xác. Với
vài nghìn ảnh — đúng tình huống trong xưởng — CNN thắng.

### Bốn quyết định trong lúc học

| Quyết định | Vì sao |
| --- | --- |
| `BCEWithLogitsLoss`, không `CrossEntropy` | model có một đầu ra, xem trên |
| `pos_weight` = số OK / số NG | không cân bằng thì model đoán "OK" suốt là đạt 95%, mà đó đúng là kiểu sai tốn tiền nhất |
| **Kẹp gradient ở 1,0** | đo được: không kẹp thì loss tập thi nhảy 0,60 → 3,07 → 0,60 trong hai epoch liền |
| Dừng sớm theo loss tập **thi** | loss tập học luôn giảm, kể cả khi model đang học vẹt |
| Giữ trọng số epoch **tốt nhất** | epoch cuối là epoch học vẹt nhiều nhất |

---

## Ngưỡng: chỗ model biến thành tiền

Độ chính xác không phải con số quyết định. Hai loại sai **không đối xứng**:

```text
loại oan (FP)   sản phẩm tốt bị thổi ra thùng phế
                tốn = giá thành sản phẩm

bỏ sót (FN)     sản phẩm lỗi đi ra thị trường
                tốn = bảo hành + uy tín + có khi cả một đợt triệu hồi
```

Thường lệch nhau **50–1000 lần**. Vì vậy ngưỡng 0,5 là một lựa chọn tuỳ tiện,
và gần như luôn là lựa chọn sai.

`evaluate.py` quét **mọi giá trị điểm có thật trong dữ liệu** (không quét lưới
0,01 — ngưỡng đúng thường nằm lệch giữa lưới, và ở bài toán lệch 50 lần thì
lệch một nấc cũng đổi hàng chục lần số tiền) để tìm chỗ tốn ít nhất.

Ngưỡng đó được lưu vào `runs/<tên>_eval.json`, và `station` **tự đọc lại**.
Đây là điểm nối quan trọng nhất giữa hai bước: để hai bên tự chọn riêng thì
model được đánh giá ở một ngưỡng mà lại chạy thật ở ngưỡng khác.

Ngoài ra còn báo **AUC** — đo mức tách biệt, không phụ thuộc ngưỡng. Đây là con
số đáng tin nhất về chất lượng model: 0,5 = đoán bừa, 1,0 = hoàn hảo.

### Nhưng toán học thuần cho ra câu trả lời không dùng được

Đây là phát hiện đáng chú ý nhất của phần này. Với tỉ lệ chi phí 50:1, ngưỡng
rẻ nhất **theo toán học** là **0,044**:

```text
bắt được 100% lỗi · thổi oan 98,1% sản phẩm tốt
bỏ sót 0 lỗi · loại oan 51 sản phẩm tốt
```

Toán đúng: bỏ sót 5 sản phẩm lỗi đắt bằng thổi oan 250 sản phẩm tốt, mà ở đây
chỉ phải thổi oan 51. Nhưng **không dây chuyền nào chạy được như vậy** — thổi
98% sản phẩm ra thùng phế nghĩa là dây chuyền dừng, và khoản lỗ đó không nằm
trong mô hình chi phí hai số hạng.

Nên `evaluate.py` chỉ xét những ngưỡng có tỉ lệ thổi oan dưới
`ModelConfig.max_false_reject_rate` (mặc định 5%), và **nói to lên khi ràng buộc
đang chặn**:

```text
    RÀNG BUỘC ĐANG CHẶN. Bỏ ràng buộc thì ngưỡng rẻ nhất
    theo toán học sẽ thổi oan hơn 5% sản phẩm tốt —
    đúng về số học nhưng dây chuyền không chạy được như vậy.
    Muốn bắt nhiều lỗi hơn thì phải làm model TỐT HƠN
    (AUC cao hơn), không phải hạ ngưỡng.
```

Dòng cuối là bài học: khi ràng buộc chặn, **hạ ngưỡng không phải là cách**.
Ngưỡng chỉ dịch chuyển điểm cân bằng trên một đường cong đã cố định; muốn dịch
cả đường cong thì phải có model tốt hơn hoặc dữ liệu tốt hơn.

---

## PLC Mitsubishi: giao thức MC tự viết

~380 dòng trên socket thô, không cần thư viện. Chi tiết đầy đủ ở
[`docs/giao-thuc-mc.md`](docs/giao-thuc-mc.md).

Khung đọc 1 từ ở `D100`, so **từng byte** với tài liệu SH-080008:

```text
50 00              subheader 3E
00                 network no
FF                 PC no
FF 03              module I/O no = 0x03FF
00                 module station no
0C 00              độ dài dữ liệu = 12
10 00              monitoring timer = 4 giây
01 04              command 0x0401 = đọc hàng loạt
00 00              subcommand 0x0000 = theo từ
A8                 mã thiết bị D
64 00 00           đầu thiết bị = 100
01 00              1 điểm
```

`python -m vision_qc check` so khung này với tài liệu mỗi lần chạy.

### Chỗ dễ sai nhất

`request data length` đếm từ **monitoring timer** (offset 9), không phải từ đầu
khung. Lỗi này đã thật sự xảy ra khi viết PLC giả: bên nhận đọc thừa đúng 6
byte rồi treo, và thông báo là *"PLC đóng kết nối giữa chừng"* — chẳng liên
quan gì tới nguyên nhân thật.

### X, Y, B, W đánh số hệ 16

`X10` là **16**, không phải 10. Sai một đơn vị ở đây là đọc nhầm đầu vào.

---

## Bắt tay với PLC, và ba lớp bảo vệ

```text
PLC -> PC   M100 = 1        "sản phẩm đã tới điểm chụp"
PC  -> PLC  D200 = điểm     điểm lỗi, 0..10000
PC  -> PLC  D201 = số thứ tự sản phẩm
PC  -> PLC  M201 = 0 / 1    OK / NG
PC  -> PLC  M200 = 1        "kết quả đã sẵn sàng"
PLC         đọc, thổi nếu M201 = 1 (giữ xung 100 ms)
PLC -> PC   M200 = 0        "đã nhận" — chính việc PLC xoá M200 là ack
```

Dùng chính việc PLC xoá M200 làm tín hiệu xác nhận, không thêm bit riêng: ít
bit hơn thì ít chỗ để lập trình sai hơn.

Ba lớp bảo vệ, không phải cho đẹp:

| Lớp | Chống chuyện gì |
| --- | --- |
| **Nhịp tim `D210`** | PC treo giữa ca. Không có nó thì PLC vẫn chạy và mọi sản phẩm lỗi đi thẳng ra thị trường |
| **Bit lỗi `M202`** | PC tự nói "tôi đang hỏng". PLC quyết định làm gì — dừng băng tải hay thổi hết |
| **Chặn ghi đè** | Ghi kết quả mới lên kết quả PLC chưa xử lý. Có test riêng: `check_plc_guard` |

**Hướng hỏng an toàn:** không suy luận được thì mặc định là **thổi**. Thà loại
oan một sản phẩm tốt còn hơn để lọt một sản phẩm lỗi. Đổi được bằng
`--fail-open`.

---

## Chạy không cần phần cứng

Cả trạm chạy được trên máy tính, kể cả trình tự bắt tay và ngân sách thời gian:

```bash
python -m vision_qc station --source synth --parts 40 --name synth
```

`SynthSource` mang theo **sự thật** (sản phẩm này có lỗi thật hay không), nên
trạm đếm được cả **số lỗi bị bỏ sót thật sự** chứ không chỉ số lần thổi. Đó là
khác biệt giữa "thổi 12 sản phẩm" và "bỏ sót 3 sản phẩm lỗi".

Hai backend PLC dùng chung một giao diện, nên `station.py` chạy y hệt trên cả
hai:

| Backend | Nói chuyện với |
| --- | --- |
| `sim` | `SimRejectPlc` trong tiến trình, bắt chước cả độ trễ quét PLC và việc tự xoá M200 |
| `mc` | `McRejectPlc` qua TCP — PLC Mitsubishi thật |

Đổi bằng `--plc mc --host 192.168.1.10 --port 5007`.

---

## Bộ sinh ảnh giả, và một lỗi nhãn nó từng có

`synth.py` vẽ sản phẩm kim loại trên băng tải, với 5 loại lỗi: xước, móp, bẩn,
sứt rìa, đốm.

Nó **không thay thế được dữ liệu thật** — không có vết dầu, không có bụi, không
có phản chiếu kim loại thật, không có rung băng tải. Độ chính xác trên ảnh giả
không nói lên điều gì về độ chính xác trong xưởng.

### Hai lỗi thật đã tìm ra trong bộ sinh

**1. Nhãn sai.** Vết xước lấy điểm bắt đầu ngẫu nhiên trên cả ảnh 1280×720 rồi
kéo một đoạn dài 158–475 px. Phần lớn số lần, đoạn thẳng đó **không chạm vào
sản phẩm** — sản phẩm chỉ chiếm một vòng tròn bán kính 198 px ở giữa. Kết quả
đo được: `max|Δ| = 0,000`. Ảnh dán nhãn NG nhưng giống hệt ảnh OK.

Đó là **nhãn sai**, và nó tệ hơn cả một vết xước mờ: model được dạy rằng những
tấm ảnh này là lỗi, nên nó học cách bỏ qua chúng.

Nay có lưới an toàn: `_apply_defect()` vẽ lỗi rồi **kiểm tra ảnh có thật sự
đổi không**. Không đổi thì thử loại lỗi khác, và nếu vẫn không được thì **báo
lỗi to** chứ không âm thầm trả về ảnh sai nhãn.

**2. Kích thước lỗi tính bằng điểm ảnh tuyệt đối.** Vết xước rộng 1–4 px trên
1280×720, thu về 192 thì còn 0,3–1,1 px — dưới ngưỡng nhiễu. Bộ sinh vẽ ra một
vết xước mà model không thể thấy, rồi ta kết luận "model dở".

Nay mọi kích thước lỗi tính theo **bán kính sản phẩm**, không theo điểm ảnh.
Đúng như ngoài xưởng: vết xước là một tính chất **của sản phẩm**, nó rộng bao
nhiêu phần trăm đường kính.

---

## Mỗi file trả lời đúng MỘT câu hỏi

| File | Câu hỏi |
| --- | --- |
| `config.py` | Các con số của dây chuyền là gì? |
| `camera.py` | Lấy ảnh thế nào, và khoá được những gì? |
| `imageops.py` | Ảnh biến thành số thế nào? |
| `capture.py` | Thu ảnh ra sao cho khỏi hỏng dữ liệu? |
| `synth.py` | Chưa có sản phẩm thật thì thử bằng gì? |
| `dataset.py` | Chia tập thế nào cho khỏi tự lừa mình? |
| `augment.py` | Bịa thêm dữ liệu từ dữ liệu đang có thế nào? |
| `model.py` | Mạng tích chập gồm những gì? |
| `train.py` | Dạy nó thế nào? |
| `evaluate.py` | Đo nó thế nào, và đặt ngưỡng ở đâu? |
| `mc.py` | Nói chuyện với PLC Mitsubishi thế nào? |
| `plc.py` | Gửi quyết định và giữ an toàn thế nào? |
| `station.py` | Cả dây chuyền ghép lại ra sao? |
| `checks.py` | Làm sao biết mọi thứ còn đúng? |

---

## Test

```bash
python -m pytest -q      # 72 bài, chạy trong 23 giây
```

Những bài đáng chú ý, vì chúng giữ những bất biến dễ mất nhất:

| Bài test | Giữ điều gì |
| --- | --- |
| `test_khung_doc_mot_tu_d100` | Khung MC khớp **từng byte** với tài liệu |
| `test_chia_tap_khong_ro_ri_san_pham` | Không sản phẩm nào nằm ở cả hai tập |
| `test_cat_vuong_KHONG_lam_meo_hinh_hoc` | Hình tròn vẫn là hình tròn |
| `test_bam_phan_biet_duoc_chieu_doc` | Ảnh chỉ biến thiên dọc không bị coi là trùng nhau |
| `test_so_am_di_ve_nguyen_ven` | Số có dấu qua PLC không bị hỏng |
| `test_moi_loai_loi_deu_de_lai_dau_vet` | Ảnh NG thật sự khác ảnh OK — không có nhãn sai |
| `test_loi_con_thay_duoc_sau_khi_thu_nho` | Lỗi không bị thu nhỏ tới mức vô hình |
| `test_do_sang_khong_tuong_quan_voi_nhan` | Không có lối tắt "sáng = tốt" |
| `test_ten_file_doc_lai_duoc` | `capture.py` ghi ra thì `dataset.py` đọc được |

---

## Chạy bằng GPU

Máy đo có GPU Quadro T2000 (4 GB) nhưng PyTorch ở đây là **bản CPU**
(`2.7.1+cpu`), nên mọi số đo trong tài liệu này là số đo CPU. Đó là lựa chọn có
chủ ý: dự án chạy được trên máy không có GPU, và với 296 nghìn thông số thì
không cần GPU.

Muốn dùng GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Rồi thêm `.to("cuda")` cho model và dữ liệu trong `train.py` và `station.py`.
Với bài toán này, lợi ích chính không phải là học nhanh hơn (10 phút thành 1
phút) mà là **suy luận nhanh hơn** — 10 ms xuống dưới 2 ms, hữu ích khi băng
tải chạy nhanh hơn nhiều.

Nhưng trước khi tối ưu tốc độ, hãy nhìn lại bảng ngân sách: p95 đang dùng 33
ms trên 1475 ms. Tốc độ chưa phải là nút cổ chai.

---

## Còn thiếu gì

Nói thẳng, để không ai tưởng dự án này xong rồi:

| Thiếu | Ghi chú |
| --- | --- |
| **Dữ liệu thật** | `data/raw/` đang trống. Đây là việc tiếp theo, và là việc quan trọng nhất. Xem [`docs/thu-thap-du-lieu.md`](docs/thu-thap-du-lieu.md) |
| Chương trình phía PLC | Repo này chỉ có phía PC. Phía PLC cần viết: đọc M200, giữ xung van 100 ms, canh nhịp tim D210, và tự xử khi M202 = 1 |
| Mật khẩu truy cập từ xa của PLC | `mc.py` chưa gửi lệnh mở khoá (`0xC200`) |
| Kiểm tra ảnh mờ | Chưa chặn ảnh mất nét do rung băng tải. Nên thêm trước khi chạy thật |
| Đo độ trễ trên PLC thật | Số đo hiện tại là trên PLC giả. Mạng công nghiệp thật sẽ chậm hơn |
| Phát hiện bất thường | Khi lớp lỗi chỉ có vài mẫu, hướng đúng là học **chỉ từ ảnh tốt** thay vì phân loại hai lớp |
| Nhiều khung hình mỗi sản phẩm | `--frames N` đã có và lấy trung bình, nhưng chưa đo xem nó giúp được bao nhiêu |
| Lệnh `random read/write` của MC | Chỉ cài `batch read/write`, đủ dùng cho việc này |

### Những chỗ cố tình làm đơn giản

| | |
| --- | --- |
| `SimMcPlc` không kiểm tra hệ đếm | PLC thật đánh số X, Y, B, W theo hệ 16; PLC giả bỏ qua chi tiết đó |
| Tăng cường ảnh theo lô | Dùng chung một mức sáng/tương phản cho cả lô, không phải từng ảnh |
| Chưa có hàng đợi | Mỗi sản phẩm một khung hình, chưa gom lô. Với 30 sản phẩm/phút thì chưa cần |
| BatchNorm dao động | Với 104 ảnh thi, thống kê chạy của BatchNorm chưa hội tụ nên loss thi thỉnh thoảng vọt lên rồi về. Lưu epoch tốt nhất nên không ảnh hưởng kết quả cuối |
