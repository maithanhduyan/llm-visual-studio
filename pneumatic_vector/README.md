# Động cơ đẩy vector khí nén

Một ống hình trụ dài 1,2 m chứa khí nén. Phóng lên 10–15 m, rồi hạ xuống
chạm đất êm. Bộ ra quyết định cấp cao là **một model nhỏ tự huấn luyện** —
cùng loại model với Cấp 3, chỉ thu nhỏ lại.

Đây là dự án Cấp 5 của bộ `llm-visual-studio`, nhưng nó khác bốn cấp trước ở
một chỗ: **lần này model không viết chữ, nó lái một vật thể bay.** Cái nó viết
ra là lệnh, và lệnh đó được đưa xuống một bộ điều khiển PID đang giữ cho con
tàu khỏi lộn nhào.

```bash
cd pneumatic_vector

python -m pneumatic_vector check      # kiểm tra mọi thứ có chạy đúng (104 mục)
python -m pneumatic_vector expert     # xem chuyên gia viết tay bay
python -m pneumatic_vector train      # dạy model bắt chước chuyên gia
python -m pneumatic_vector dagger     # chữa bệnh "lạc vào chỗ thầy chưa tới"
python -m pneumatic_vector eval       # đo model đã lưu
python -m pneumatic_vector why        # vì sao model hỏng — nhìn lúc nó bắt đầu hãm
python -m pneumatic_vector fly        # ghi chuyến bay ra file
python -m pneumatic_vector viewer     # mở trình xem 3D
```

---

## 1. Con tàu

```
        ┌───────────────┐  ← mũi nhọn
        │               │
        │  khí nén      │   ống dài 1,2 m, đường kính 10 cm
        │  10 bar       │   chứa 112 g không khí
        │               │
        ├───────┬───────┤  ← trọng tâm ở giữa
        │       │       │
        │       ▼       │  ← vòi phun, nghiêng được ±12°
        └───────┴───────┘
                ↓ khí phụt ra
```

| | |
|---|---|
| Khối lượng vỏ | 0,60 kg |
| Khối lượng khí nén | 0,112 kg |
| **Tổng** | **0,712 kg** (nặng 7,0 N) |
| Lực đẩy tối đa | **150,8 N — hơn trọng lượng 21,6 lần** |
| Cổ vòi phun | đường kính 12 mm |
| Vòi phun nằm dưới trọng tâm | 0,62 m |
| Quán tính quay | 0,035 kg·m² |

Con số 21,6 lần là lý do bài toán này thú vị: con tàu **thừa sức bay lên**, cái
khó là hạ xuống. Lực đẩy mạnh gấp 21 lần trọng lượng nghĩa là chỉ cần mở van
5% là đã lơ lửng được — và cũng có nghĩa là mở van sai một nhịp là vọt lên mất.

### Ba thứ khiến việc điều khiển khó

**1. Khí cạn dần.** Áp suất tụt từ 10 bar xuống 1 bar trong chuyến bay. Lưu
lượng khí phụt ra giảm từ 227 g/s xuống 9 g/s. Lực đẩy không phải là hằng số —
nó phụ thuộc vào thứ đang cạn dần.

```
10,0 bar → 227,0 g/s      2,0 bar → 45,4 g/s
 6,0 bar → 136,2 g/s      1,2 bar → 20,3 g/s
 4,0 bar →  90,8 g/s      1,05 bar → 9,0 g/s
```

**2. Con tàu là con lắc ngược.** Trọng tâm ở trên, lực đẩy ở dưới. Tâm khí động
lại nằm *trên* trọng tâm, nên nghiêng một chút là nó **tự ngã thêm**. Giống giữ
cây chổi dựng trên lòng bàn tay. Tắt bộ điều khiển một giây là con tàu lộn nhào.

**3. Lực đẩy làm hai việc cùng lúc.** Nghiêng vòi phun vừa tạo lực ngang, vừa
tạo mô-men xoay. Đó là toàn bộ nguyên lý lái vector:

```
τ = −L · T · δ          mô-men xoay CHỈ phụ thuộc góc nghiêng VÒI PHUN
```

Không phụ thuộc con tàu đang nghiêng bao nhiêu. Hệ quả: lực đẩy càng yếu thì
lái càng yếu — gần cạn khí thì mở van hết cũng không xoay nổi con tàu.

---

## 2. Hai tầng, hai nhịp

Đây là chỗ phân công quan trọng nhất của cả dự án:

```
   LLM (model nhỏ)          20 Hz    nghĩ chậm, quyết định LỚN
        ↓  LEN / GIU / ROI / HAM
   PID (ba tầng)          1000 Hz    nghĩ nhanh, quyết định nhỏ
        ↓  ga + góc vòi phun
   vật lý                 1000 Hz
```

Chênh nhau **50 lần**. Đó không phải chi tiết kỹ thuật vặt — đó chính là lý do
phải tách làm hai tầng. LLM không thể chạy ở 1000 Hz, và cũng không cần: quyết
định "lên 12 mét" thì 20 lần mỗi giây là quá đủ.

Model chỉ được phép nói **bốn chữ**:

| Lệnh | Nghĩa |
|---|---|
| `LEN` | mở van, bay lên |
| `GIU` | giữ ở đỉnh, giảm vận tốc lên về 0 |
| `ROI` | cắt ga hoàn toàn, rơi tự do |
| `HAM 6.5` | hãm theo giản đồ, số là gia tốc hãm mong muốn (m/s²) |

`LEN` và `GIU` **không** kèm con số. Lúc đầu chúng có kèm độ cao mục tiêu
(`LEN 12.0`), nhưng model nhỏ chép lại con số rất kém — đo ra thì nó viết đúng
*lệnh* 100% mà chỉ chép đúng *con số* 85%. Mà độ cao mục tiêu thì hệ thống đã
biết rồi (nó nằm trong đề bài), nên bắt model chép lại là bắt nó làm việc vô
ích mà lại hay sai. Bỏ con số đi thì nó chỉ còn phải chọn một trong bốn lệnh.

---

## 3. Chiến thuật: rơi tự do rồi hãm ở cuối

Thử tính xem. Con tàu có 112 g khí. Giữ cho nó lơ lửng tốn khoảng **10 g khí
mỗi giây**. Hạ từ từ 12 m với vận tốc 1,5 m/s mất 8 giây — tốn 80 g. Vừa đủ
nhưng không còn dư để sửa sai.

Cách các tàu thật làm: **rơi tự do rồi hãm ở cuối.**

```
   rơi tự do 6,6 m   →   tốn 0 g khí
   hãm ở 6 m cuối    →   tốn ~20 g
```

Đo được: chuyên gia rơi tự do **6,6 m mà áp suất đứng yên ở 5,71 bar** — không
tốn một gam khí nào. Đổi lại, phải canh đúng lúc bắt đầu hãm. Hãm sớm quá thì
tốn khí, hãm muộn quá thì đâm xuống đất.

**Đó chính là quyết định khó nhất trong cả bài toán — và là việc của model.**

---

## 4. Chuyên gia viết tay

`expert.py` là một cái máy trạng thái, không học gì cả. Nó là "người thầy".

```
LEN → GIU → ROI → HAM → XONG
```

| Đề bài | Lên cao nhất | Chạm đất | Khí đã dùng | Thời gian |
|---|---|---|---|---|
| 10 m | 11,39 m | 1,97 m/s | 67 g / 112 g | 6,5 s |
| 12 m | 13,41 m | 1,72 m/s | 81 g / 112 g | 7,5 s |
| 15 m | 16,43 m | 1,84 m/s | 98 g / 112 g | 8,7 s |

**30/30 chuyến đạt.** Đạt nghĩa là: lên tới ít nhất 10 m, lên đúng tầm (lệch
không quá 2 m so với đề bài), và chạm đất dưới 2,0 m/s.

Chỗ đáng chú ý: con tàu **vượt tầm khoảng 1,4 m** ở mọi đề bài. Nó chuyển sang
`GIU` khi tới 98% độ cao mục tiêu, nhưng lúc đó còn đang bay lên 1,9 m/s nên
vẫn trôi thêm. Muốn khít hơn thì phải chuyển sớm hơn — nhưng như vậy lại tốn
khí hơn, và 112 g thì không dư dả gì.

Cái khó thật sự nằm ở chỗ này, trong `ExpertPilot.available_deceleration`:

> lực hãm yếu dần khi khí cạn

Nếu cứ tưởng lúc nào cũng hãm được 8 m/s² thì sẽ bắt đầu hãm quá muộn, và đâm
xuống đất. Nên giản đồ hạ cánh phải lấy `a` theo khả năng **thật** của con tàu
lúc đó, không phải một con số cố định.

---

## 5. Model nhỏ

`policy.py` dựng lại kiến trúc của Cấp 3 (`deepseek_lite`), thu nhỏ:

| | |
|---|---|
| d_model | 64 |
| Số tầng | 2 |
| Đầu | 4 đầu, 2 đầu KV (GQA) |
| Chuyên gia | 4, chọn 2, cộng 1 chuyên gia dùng chung |
| Thứ tự tối đa | 48 |
| **Thông số** | **121.220** |
| Ký tự | 26 |

Nó đọc trạng thái và viết ra lệnh:

```
"12.0 7.5 -1.5 4.0 10.0 -> "   model viết tiếp   "LEN"
 └──┬──┘ └──────┬──────┘
  đề bài     trạng thái
```

Năm con số, và **con số đầu tiên là đề bài**. Chỗ này từng viết thiếu, và hậu
quả rất dễ thấy: model bay lên ~13,5 m với *mọi* đề bài. Nó không ngu — nó chỉ
không được cho biết đề bài là gì. Chuyên gia biết (vì `ExpertPilot` giữ
`mission`), model thì không, nên nó học được mỗi hành vi trung bình của mọi đề
bài.

### Vẫn phải có lớp chắn

Model nhỏ sẽ có lúc viết ra thứ vô nghĩa — `HAM 999`, hay một chuỗi khoảng
trắng. Nếu đưa thẳng xuống PID thì con tàu rơi. Nên giữa model và cơ cấu chấp
hành luôn có một lớp kiểm tra:

```python
if first_word not in COMMANDS:   self.invalid += 1
if decision.command == "HAM" and not (1.0 <= decision.value <= 15.0):
    self.invalid += 1
    decision = Decision("HAM", 6.5)
```

Mọi hệ thống thật đều làm vậy: LLM ra quyết định, nhưng không bao giờ được
nối thẳng vào động cơ.

### Học bắt chước thì chưa đủ

Dạy model bắt chước chuyên gia: 120 chuyến bay → 3.120 ví dụ. Học 2500 bước
mất **165 giây** trên CPU, loss 3,318 → 0,291, loss thi 0,278.

Nhìn vào loss thì tưởng ngon. Nhưng cho nó tự bay: **47%** — trong khi chuyên
gia đạt 100%. Đây là **bệnh lạc vào chỗ thầy chưa từng tới**:

> Lúc học, model chỉ thấy những trạng thái mà *chuyên gia* đi qua. Ra bay
> thật, chỉ cần lệch một chút là nó rơi vào trạng thái chuyên gia **chưa từng
> ở đó** — và ở đó thì nó chưa được dạy gì cả. Lệch một chút thành lệch
> nhiều, rồi hỏng.

```
lúc học:   thầy đi đường A, trò học đường A
ra bay:    trò lệch sang đường B -> chưa học gì về B -> lệch tiếp
```

(Con số 47% này là của một lần chạy cụ thể. Cùng cách dạy đó, một lần chạy
khác đã cho 7%. Vì sao lại chênh nhiều thế, xem mục
**"Vì sao bảng kết quả này chỉ là MỘT lần chạy"** ở cuối — đó là bài học quan
trọng nhất về cách đọc mọi con số trong tài liệu này.)

### Thuốc: DAgger

Cho trò bay, nhưng **thầy ngồi cạnh sửa ngay trên đường trò đi**:

```
1. Trò bay (bằng model hiện có)
2. Ở mỗi chỗ trò phải quyết định, hỏi thầy: "chỗ này thầy làm gì?"
3. Ghi lại (trạng thái TRÒ đang ở, câu trả lời của THẦY)
4. Dạy lại model trên dữ liệu cũ + dữ liệu mới
5. Lặp lại vài vòng
```

Điểm mấu chốt: dữ liệu mới ghi lại **những trạng thái mà TRÒ đi qua**, không
phải thầy. `checks.py` kiểm tra đúng điều này — sau một vòng DAgger, **176/176
trạng thái ghi được là trạng thái mà dữ liệu học bắt chước thuần không hề có.**

<!-- KẾT QUẢ DAGGER -->

### Kết quả DAgger — và một kết quả không như mong đợi

Đo trên cùng một bộ đề bài (10, 12, 15 m), model tự bay:

| | Chuyên gia | Vòng 0 (bắt chước thuần) | Vòng 1 | Vòng 2 | Vòng 3 | **Đo lần cuối** |
|---|---|---|---|---|---|---|
| 10 m | 10/10 | | 0/5 | 0/5 | 0/5 | **0/10** |
| 12 m | 10/10 | | 1/5 | 3/5 | 4/5 | **9/10** |
| 15 m | 10/10 | | 3/5 | 5/5 | 4/5 | **9/10** |
| **Tổng** | **30/30 (100%)** | **7/15 (47%)** | 4/15 (27%) | 8/15 (53%) | 8/15 (53%) | **18/30 (60%)** |

Ba điều đáng nói, và điều thứ hai thì không vui:

**1. DAgger có giúp, nhưng ít hơn nhiều so với kỳ vọng.** 47% lên 60%. Không
phải "sửa xong là bay được".

**2. Đường đi không hề thẳng: vòng 1 làm mọi thứ TỆ ĐI (47% → 27%).** Phải tới
vòng 2 mới vượt được vòng 0. Nếu dừng ở vòng 1 thì kết luận sẽ là "DAgger
phản tác dụng" — và kết luận đó sẽ sai, chỉ vì dừng quá sớm.

**3. Model hỏng ở đúng một đề bài, và hỏng rất có quy tắc: 10 m → 0/10.** Không
phải ngẫu nhiên. Cả 10 chuyến đều chạm đất 7,92–10,28 m/s. Hỏng có hệ thống
nghĩa là có nguyên nhân cụ thể, và tìm ra được:

```bash
python -m pneumatic_vector why
```

```text
CHUYÊN GIA VIẾT TAY
  đề bài |  cao lúc bắt đầu hãm |  đang rơi | mức an toàn | chạm đất | kết quả
    10.0  |                5.10 |    10.97 |       10.59 |     1.97 |    ĐẠT
    12.0  |                6.57 |    11.43 |       12.02 |     1.72 |    ĐẠT
    15.0  |                7.78 |    12.80 |       13.08 |     1.84 |    ĐẠT

MODEL ĐÃ HỌC
  đề bài |  cao lúc bắt đầu hãm |  đang rơi | mức an toàn | chạm đất | kết quả
    10.0  |                4.41 |    11.40 |        9.84 |     8.46 |   HỎNG
    12.0  |                7.62 |    10.65 |       12.94 |     1.92 |    ĐẠT
    15.0  |                8.36 |    12.39 |       13.57 |     1.76 |    ĐẠT
```

**Model học đúng phần VẬN TỐC, nhưng bỏ mất phần ĐỘ CAO.** Vận tốc lúc bắt đầu
hãm thì nó học rất sát chuyên gia (10,6–12,4 m/s so với 11,0–12,8 m/s). Nhưng
*độ cao* lúc bắt đầu hãm thì bị co lại vào giữa: chuyên gia hãm từ 5,10 m tới
7,78 m tuỳ đề bài, còn model chỉ từ 4,41 m tới 8,36 m.

Quy tắc thật phụ thuộc vào **độ cao còn lại** — `v ≤ sqrt(2·a·h)` — chứ không
phải vào vận tốc. Ở đề bài 10 m, model hãm ở 4,41 m khi đang rơi 11,40 m/s,
trong khi chuyên gia hãm ở 5,10 m khi mới rơi 10,97 m/s. **Lệch 0,69 m ở độ
cao hãm đủ để chạm đất 8,46 m/s thay vì 1,97 m/s.**

Đây là bài học đáng nhớ nhất của cả dự án:

> **Ở gần biên của bài toán, một sai số nhỏ là chết.** Ở đề bài 15 m, sai số
> ấy vẫn còn 0,6 m đường hãm dự phòng nên không sao. Ở đề bài 10 m thì không
> còn gì để dự phòng.

### Hai lỗi trong chính đường ống DAgger, tìm ra vì kết quả vô lý

Vòng 1 cho 27% — tệ hơn cả vòng 0. Con số vô lý thì phải đi tìm nguyên nhân,
chứ không phải báo cáo rồi đi tiếp. Tìm ra hai lỗi, cả hai đều nằm ở phần
*ghép nối*, không nằm ở thuật toán:

**Lỗi 1 — không dồn dữ liệu.** Chữ "Aggregation" trong tên DAgger là để **dồn**
dữ liệu qua các vòng. Bản đầu tiên gọi `dagger_round(..., [], ...)` — mỗi vòng
vứt sạch dữ liệu cũ và chỉ học trên những chỗ model vừa bay hỏng. Model mất
luôn đường bay chuẩn mà nó đã học được. Sửa: truyền `data` tích luỹ vào, và
`dagger_round` trả về dữ liệu cũ nguyên vẹn cùng dữ liệu mới.

**Lỗi 2 — ghi dày gấp sáu lần.** `dataset.generate` chỉ ghi 1 trong mỗi 6 lần
hỏi, vì giai đoạn `LEN` kéo dài mấy giây còn `HAM` chưa đầy một giây — ghi hết
thì dữ liệu lệch hẳn. `dagger_round` thì ghi **mọi** lần hỏi: 9.737 ví dụ từ 70
chuyến, gấp 5 lần mật độ của dữ liệu chuyên gia. Trộn hai nguồn vào nhau là
giai đoạn `LEN` lấn át `HAM`. Sửa: cho `dagger_round` ghi thưa giống hệt
`dataset.generate` → còn 1.701 ví dụ.

Sau khi sửa cả hai, số ví dụ mỗi vòng và tỉ lệ các lệnh mới khớp với dữ liệu
chuyên gia:

| | LEN | GIU | ROI | HAM |
|---|---|---|---|---|
| dữ liệu chuyên gia | 45,9% | 16,0% | 16,0% | 22,2% |
| dữ liệu DAgger thêm vào | 51,5% | 16,7% | 19,3% | 12,4% |

Vẫn còn lệch ở `HAM` (12,4% so với 22,2%) — vì trò bay dở nên nó ít khi tới
được chỗ mà thầy sẽ hãm. Đó là một phần lý do DAgger ở đây chỉ giúp được 13
điểm phần trăm.

### Vì sao bảng kết quả này chỉ là MỘT lần chạy

Đáng nói rõ, vì nó ảnh hưởng tới cách đọc mọi con số ở trên.

Cùng một seed, cùng một cách gọi hàm, hai lần chạy cho ra hai model khác nhau —
vì giữa hai lần đó tôi sửa một lỗi vật lý, và lỗi đó làm **30 trong 3.120 dòng**
dữ liệu chuyên gia thay đổi (1,0%). 30 dòng đó nằm đúng ở chỗ chuyển giao
`LEN → GIU`, tức là chỗ khó nhất. Kết quả: val loss 0,2776 so với 0,2654, và
vòng 0 nhảy từ **2/30 lên 7/15**.

> Sửa 1% dữ liệu, kết quả đổi từ 7% lên 47%. Với 3.120 ví dụ và một model 121
> nghìn thông số, **một lần chạy không đủ để kết luận điều gì**. Bảng ở trên
> là một lần chạy nhất quán từ đầu tới cuối, nên so sánh *giữa các vòng trong
> bảng* là đáng tin; còn con số tuyệt đối thì nên đọc là "khoảng 50-60%", chứ
> đừng đọc là "đúng 60%".

Muốn chắc thì phải chạy nhiều seed rồi lấy trung bình — việc đó chưa làm.

---

## 6. Thiết kế cánh đuôi: có cần không, và nó **không** làm được gì

```bash
python -m pneumatic_vector fins
```

Câu hỏi đặt ra: *"khi rơi tự do, lực cản không khí làm mất ổn định, nên gắn
thêm hai cánh hai bên để dẫn hướng cho rơi chính xác"*.

**Nguyên nhân thì đúng một nửa.** Không phải lực cản làm mất ổn định. Lực cản
chỉ làm con tàu chậm lại — nó không tạo mô-men nào quanh trọng tâm. Thứ làm mất
ổn định là **tâm khí động nằm TRÊN trọng tâm**:

```
τ = +K·q·S·θ        nghiêng càng nhiều -> mô-men càng mạnh -> ngã thêm
```

Đây đúng là con lắc ngược. Lực cản là số hạng hoàn toàn khác trong cùng phương
trình, và nó không gây ra chuyện này.

**Cánh đuôi sửa đúng chỗ đó.** Cánh kéo tâm khí động xuống dưới trọng tâm, làm
mô-men đổi dấu:

```
τ = −2·q·A·L·θ      nghiêng càng nhiều -> mô-men càng mạnh -> DỰNG LẠI
```

Điều kiện tĩnh ổn định: `2·A·L > K·S`, tức mỗi cánh cần **111 cm²** (gắn cách
trọng tâm 0,50 m). Với ống dài 1,2 m đường kính 10 cm thì đó là hai cánh cỡ
11 cm × 10 cm.

### Đo được: cánh giữ thân tàu thẳng rất tốt

Cắt ga hoàn toàn rồi thả từ 12 m — lúc đó lực đẩy bằng 0 nên vòi phun không
lái được gì, con tàu rơi tự do thật:

| Cấu hình | Nghiêng tối đa khi rơi |
|---|---|
| không cánh | **178°** (lộn nhào hẳn) |
| một nửa cỡ | 42° |
| vừa đủ trung tính | **3,5°** |
| gấp đôi | 3,5° |

### Đo được: nhưng cánh KHÔNG dẫn hướng

Đây là chỗ dễ ngộ nhận nhất. Ném con tàu đi ngang 3 m/s rồi thả:

| Cấu hình | Rơi lệch | Nghiêng tới | vx lúc chạm |
|---|---|---|---|
| không cánh | 4,65 m | 119,78° | +2,85 m/s |
| cánh 111 cm² mỗi bên | 4,63 m | 1,34° | +2,82 m/s |
| cánh 221 cm² mỗi bên | 4,62 m | 0,41° | +2,78 m/s |

Cánh đổi độ nghiêng từ **120° xuống 0,4°** — sửa được hoàn toàn việc lộn nhào.
Mà quỹ đạo rơi chỉ đổi **2 cm trên 4,65 m, tức 0,4%**.

> Cánh chỉ làm thân tàu quay theo chiều gió. Nó **không tạo lực ngang**, nên
> không đổi được chỗ rơi. Muốn rơi đúng chỗ thì phải có thứ ĐẨY NGANG.

### Thứ đẩy ngang được thì đã có sẵn

| Cấu hình | Rơi lệch | vx lúc chạm |
|---|---|---|
| cắt ga, không điều khiển | 4,65 m | +2,85 m/s |
| **PID lái bằng vòi phun** | **1,34 m** | **−0,38 m/s** |
| PID + cánh vừa đủ | 1,74 m | −0,58 m/s |

Vòi phun **đảo được chiều trôi ngang**: vx lúc chạm từ +2,85 xuống −0,38 m/s.
Đó mới là dẫn hướng.

Và gắn thêm cánh vào thì lại **tệ hơn** (1,74 m so với 1,34 m): cánh chống lại
chính động tác nghiêng mà bộ điều khiển cần để lái.

### Và cánh to thì làm hỏng việc lái

| Cấu hình | Đạt | Nghiêng khi rơi | Trôi ngang | Khí |
|---|---|---|---|---|
| không cánh | **15/15** | 1,87° | 0,46 m | 98,4 g |
| một nửa (55 cm²) | **15/15** | **0,94°** | 0,48 m | 98,5 g |
| vừa đủ (111 cm²) | 10/15 | 0,35° | 0,51 m | 98,4 g |
| gấp đôi (221 cm²) | **0/15** | 0,48° | 0,63 m | 99,2 g |

Con tàu lái bằng cách **nghiêng thân** cho lực đẩy chĩa sang ngang. Cánh ép
thân tàu bám theo chiều gió — tức là chống lại đúng động tác lái đó. Cánh càng
ổn định thì lái càng khó. Đây là đánh đổi kinh điển giữa **ổn định** và **cơ
động**, và ở đây nó đo được: 15/15 → 10/15 → 0/15.

Đáng chú ý: **không tốn thêm khí.** Diện tích cản tăng từ 79 lên 101 cm², mà
khí dùng vẫn 98,4 g — vì lúc bay lên con tàu thẳng đứng nên cánh nằm dọc theo
dòng khí. Cái giá thật của cánh không phải khí, mà là nó chống lại việc lái.

### Kết luận

1. **Chẩn đoán:** không phải lực cản gây mất ổn định, mà là tâm khí động nằm
   trên trọng tâm.
2. **Cánh đuôi sửa được điều đó**, và sửa rất tốt: 178° → 3,5°.
3. **Nhưng cánh không dẫn hướng.** Quỹ đạo rơi đổi 0,4%. Cánh không tạo lực
   ngang, nên không thể làm nó rơi chính xác hơn.
4. **Trên con tàu này, cỡ cánh đủ để tĩnh ổn định lại làm hỏng việc lái**
   (15/15 → 10/15 → 0/15), vì con tàu lái bằng cách nghiêng thân.
5. **Nếu vẫn muốn gắn thì gắn NHỎ** — khoảng một nửa cỡ trung tính (55 cm² mỗi
   bên): giữ nguyên 15/15, nghiêng giảm một nửa, không tốn khí, không hại việc
   lái. Cánh nhỏ đóng vai trò **giảm dao động**, không phải ổn định.
6. **Muốn rơi chính xác thì phải cải thiện dẫn hướng**, không phải thêm cánh.
   Cụ thể là chỗ đang bỏ trống: **lúc rơi tự do thì cắt ga, mà cắt ga là mất
   hết quyền lái.** Hiện tại bộ điều khiển trả về `Command(0,0,0)` khi `cut`,
   nên suốt 6,6 m rơi con tàu không được sửa gì cả.

Con tàu ở chế độ chạy sống có sẵn thanh **Cánh đuôi** để thử trực tiếp: kéo
lên và đọc dòng "Ổn định" trong bảng số liệu — nó nói ngay cần bao nhiêu cm²
nữa để trung tính.

---

## 7. Trình xem 3D

`viewer/` là một trang web dùng **Bun** làm máy chủ và **three.js** để vẽ. Python
mô phỏng rồi ghi chuyến bay ra `flights/*.json`; trình duyệt chỉ việc đọc lại
mà phát. Không cần Python chạy nền.

```bash
cd viewer
bun install        # lần đầu
bun run server.ts  # rồi mở http://localhost:4180
```

Trình xem có hai chế độ.

### Chạy sống — vật lý và model chạy trong trình duyệt

Đây mới là phần đáng nói. Ở chế độ này **không có file chuyến bay nào cả**:
`vehicle.py`, `physics.py`, `controller.py`, `expert.py` và cả `deepseek_lite`
đã được dịch sang TypeScript, và trình duyệt tự chạy hết.

```
   vật lý          1.000 bước mỗi giây, trong trình duyệt
   model ra lệnh      20 lần mỗi giây, trong trình duyệt
   màn hình vẽ        60 lần mỗi giây
```

Đo được trên Chrome chạy thật: **1.004 bước vật lý mỗi giây** thời gian thật, và
mỗi lần model ra quyết định tốn **26,7 ms** (20 Hz cần dưới 50 ms, nên còn dư).
Cả hai con số này do `check-live.ts` đo, không phải ước lượng.

Kéo thanh chỉnh tham số thì con tàu **bay lại ngay** với tham số mới, vì không
có gì được tính sẵn:

| Tham số | Đo được |
|---|---|
| Đề bài 8 m → 17 m | đỉnh 9,30 m → 18,39 m |
| Vỏ 0,35 kg → 1,15 kg (sau 2,5 giây) | còn 8,04 bar → 4,53 bar |
| Nạp 4,5 bar → 15 bar (sau 2,5 giây) | còn 1,01 bar → 12,06 bar |
| Mất ổn định 0 → 3,5 | nghiêng tối đa 2,4° → 6,9° |

Chạy sống với đề bài 12 m cho ra **đỉnh 13,41 m, chạm đất 1,93 m/s** — đúng
bằng con số của bản Python.

### Phát lại — xem lại chuyến bay Python đã ghi

Vật lý ở đây **không** chạy trong trình duyệt. Python mô phỏng ở 1.000 Hz, ghi
ra `flights/*.json` ở 50 Hz, rồi trình duyệt đọc lại mà vẽ. Nghĩa là **95% số
bước vật lý không bao giờ được hiện**, và hình dạng luồng khí cũng như hướng
con tàu là quy ước thị giác chứ không phải mô phỏng.

Giữ lại chế độ này vì nó cho xem những chuyến bay đã đo — kể cả những chuyến
hỏng — thứ mà chạy sống không tái hiện lại được.

### Hình vẽ

Con tàu 3D có luồng khí phụt theo mức mở van, vệt bay, vòng chia độ cao mỗi
5 m, HUD số liệu sống, danh sách những lần model đổi lệnh.

Toạ độ phải đổi trục một lần cho gọn — mô phỏng dùng `z` làm độ cao, three.js
dùng `y`:

```
three.js (x, y, z)  =  mô phỏng (x, z, y)
```

### Bản dịch có thật sự giống bản gốc không?

Đây là câu hỏi duy nhất đáng hỏi, và nó phải được trả lời bằng số đo chứ không
phải bằng "trông có vẻ chạy". `check-port.ts` chạy đúng cùng một việc ở cả hai
bên rồi so từng bước:

| Tầng | So cái gì | Kết quả |
|---|---|---|
| Vật lý thuần | 4.000 bước × 13 con số = **40.000 con số** | lệch tuyệt đối 5,6e-9 · tương đối 3,8e-7 |
| Chuyên gia + PID | cả chuyến bay, **7.547 bước** | lệch 3,3e-9 |
| Model | 28 câu đưa vào | **28/28 viết ra y hệt**, logits lệch 3,6e-6 trên thang 9,0 |
| Đầu-cuối | 3 chuyến bay do model lái, từng bước | lệch 3,5e-9 |

```bash
cd viewer
bun run check-port.ts     # 14/14 — bản TypeScript khớp bản Python
bun run check-live.ts     # 13/13 — chế độ sống chạy thật (cần Chrome + CDP)
bun run check-viewer.ts   # 17/17 — chế độ phát lại
bun run shot-viewer.ts ./anh
```

Ba file kiểm tra này cần Chrome đang mở với `--remote-debugging-port=9333`.

Một chỗ đáng ghi lại vì nó suýt làm tôi kết luận sai: bài kiểm tra vật lý ban
đầu lái vòi phun bằng một hàm sin cố định, không quan tâm con tàu đang nghiêng
bao nhiêu. Con tàu là con lắc ngược, nên nó lộn nhào thật, và tới bước 4.000
thì `omega_theta` lên tới −2,4e11 rad/s — một con số vô nghĩa về mặt vật lý. Ở
đó sai số làm tròn bị khuếch đại thành 61, và bài kiểm tra báo hỏng trong khi
bản dịch không hề sai. **Hệ bất ổn thì khuếch đại sai số**; muốn so hai bản
dịch với nhau thì phải giữ hệ trong vùng còn lành mạnh, và phải so bằng
`allclose` chứ không so riêng sai số tuyệt đối hay tương đối.

---

## 8. Kiểm tra

```bash
python -m pneumatic_vector check
```

**104 mục.** Phần lớn trong đó là những lỗi **đã từng xảy ra thật** trong dự án
này, không phải kiểm tra cho có:

| Lỗi đã từng có | Hậu quả | Kiểm tra nào bắt |
|---|---|---|
| Công thức dòng chảy dưới âm thanh viết ngược tỉ số áp suất | số hạng trong căn luôn âm → **động cơ tắt ngóm dưới 1,9 bar**, đúng lúc cần hãm nhất | "dưới 1,9 bar động cơ vẫn phải chạy" |
| Gộp `flow <= 0` chung với "hết khí" | **cắt ga là mất sạch khí**: áp suất tụt 4,00 → 1,01 bar giữa lúc rơi tự do | "cắt ga thì áp suất giữ nguyên" |
| Đọc vận tốc *sau* bước va chạm (vật lý ghim `vz` về 0) | chuyến bay nào cũng "hạ cánh 0,00 m/s", kể cả những cú đâm | "chuyến bay vẫn ghi nhận vận tốc chạm đất thật, khác 0" |
| Trạng thái thiếu độ cao mục tiêu | model bay lên ~13,5 m với **mọi** đề bài | "con số ĐẦU TIÊN phải là độ cao mục tiêu" |
| Bắt model chép lại con số đề bài | viết đúng lệnh 100% mà chép đúng số chỉ 85% | "LEN và GIU KHÔNG kèm con số" |
| Vận tốc phụt tính ở áp suất *cuối* bước | đốt sạch khí trong một bước dài mà lực đẩy bằng **0 N** | "đốt sạch khí trong một bước dài vẫn sinh ra lực đẩy" |
| Bộ đếm lệnh hỏng đọc lệnh *đã* được sửa | `invalid_rate` mãi mãi bằng 0, giấu đi việc model viết rác | "chữ không hiểu được cũng bị đếm" |
| Nạp chuyến bay bất đồng bộ không chặn câu trả lời về muộn | đổi chuyến bay nhanh thì trang hiện số liệu chuyến này, ô chọn ghi chuyến khác | "đổi chuyến bay liên tiếp thì số liệu khớp" |
| `BufferGeometry.setFromPoints` ghi vào đệm **đã có**, không thay cả mảng điểm | gọi lần đầu với 1 điểm là đệm chỉ còn 1 chỗ → **vệt bay không bao giờ hiện**, ở bất kỳ thời điểm nào | "vệt bay thật sự được vẽ ra (tắt đi thì ảnh phải khác)" |
| `LiveFlight.reset()` không xoá `peak` và `maxTilt` | kéo thanh chỉnh tham số xong, bảng vẫn ghi "Lên cao nhất 18,43 m" của chuyến **trước** trong khi chuyến mới chỉ lên 13,41 m | "tắt mất ổn định thì cả chuyến bay nghiêng ít hơn" |
| Bộ điều khiển tự tính lại số hạng khí động bằng công thức cũ | `physics.step` biết về cánh đuôi còn `controller.attitude_gimbal` thì không, nên gắn cánh vào là bộ điều khiển đẩy vòi phun quá mạnh theo chiều ngược lại | "cánh to hơn mức trung tính thì con tàu mới thật sự tĩnh ổn định" |
| `fin_area` ghi là "tổng hai cánh" nhưng công thức `2·A·L` chỉ đúng nếu `A` là **mỗi** cánh | trình xem gửi diện tích gấp đôi nhãn ghi — kéo thanh tới 200 cm² thì mô hình nhận 400 cm² | "tính được cỡ cánh cần để vừa đủ trung tính" |
| Đổi tham số mà không xoá dòng trạng thái cũ | trang vẫn ghi "Chạm đất 1,93 m/s — ĐẠT" của chuyến vừa rồi trong khi con tàu mới chưa cất cánh | "bay hết chuyến thì tự dừng và báo kết quả" |
| Khử trùng trọng số theo `id()` thay vì theo storage | `torch.load` dựng lại hai tensor khác object nhưng chung vùng nhớ, nên bảng tra bị đếm hai lần: 122.892 thay vì 121.228 | "số thông số xuất ra khớp với model đã lưu" |
| `del LearnedPilot.write` để hoàn tác monkeypatch | xoá hẳn phương thức khỏi **lớp**, mọi mục kiểm tra chạy sau đó chết với `no attribute 'write'` | "đề bài đưa vào phải tới được ĐÚNG model bên trong" |

Cái cuối đáng nói thêm, vì nó là bài học đắt nhất của cả dự án. Tên hàm
`setFromPoints` nghe như *"thay cả danh sách điểm"*. Không phải. Nó ghi vào
đệm đã có, và chỉ ghi được tối đa bằng số điểm mà đệm đó **đã chứa** — vượt
quá thì nó bỏ im lặng, chỉ cảnh báo một dòng ra console.

Mà lần gọi đầu tiên luôn là lúc đồng hồ ở giây 0, tức là **một** điểm. Đệm
thành ra chỉ có một chỗ. Đường vẽ bằng một điểm thì không vẽ ra gì cả. Nên vệt
bay không bao giờ hiện — dù mã nguồn trông rất hợp lý, và mọi mục kiểm tra
khác vẫn "đạt" một cách vô nghĩa.

Chỉ có cách soi **điểm ảnh thật** mới bắt được: chụp ảnh lúc bật vệt bay, chụp
lại lúc tắt, rồi so hai tấm. Giống hệt nhau nghĩa là chưa từng có gì được vẽ.
Đó chính là mục kiểm tra số 16 trong `check-viewer.ts`.

Sau khi sửa, đếm điểm ảnh xanh theo từng giai đoạn — vệt bay dài dần đúng như
phải thế:

| Đang phát | 1% | 25% | 45% | 62% | 85% | 99% |
|---|---|---|---|---|---|---|
| Điểm ảnh vệt bay | 0 | 134 | 388 | 488 | 851 | **974** |

Ngoài ra còn kiểm tra cả những chỗ nối giữa các file mà trình biên dịch không
thể thấy: mọi phần tử `main.ts` tìm đều có trong `index.html`, mọi hàm nó gọi
đều có trong `scene.ts`, và `scene.ts` chỉ đọc những trường mà `simulate.py`
thật sự ghi ra.

---

## 9. Các file

```
pneumatic_vector/
├── pneumatic_vector/
│   ├── vehicle.py       con tàu, khí nén, lực đẩy        (vật lý tĩnh)
│   ├── physics.py       8 con số trạng thái, mô-men xoay (vật lý động)
│   ├── controller.py    PID ba tầng, chạy 1000 Hz        (tầng trong)
│   ├── expert.py        chuyên gia viết tay + 4 lệnh     (người thầy)
│   ├── simulate.py      chạy một chuyến bay, ghi log
│   ├── dataset.py       sinh dữ liệu <trạng thái> -> <lệnh>
│   ├── policy.py        model đã học làm bộ ra quyết định
│   ├── train.py         học bắt chước
│   ├── dagger.py        DAgger — sửa bệnh lệch đường
│   ├── explain.py       vì sao model hỏng — nhìn lúc nó bắt đầu hãm
│   ├── fins.py          cánh đuôi: có cần không, cần bao nhiêu, không làm được gì
│   ├── export.py        xuất trọng số + dữ liệu đối chiếu cho trình duyệt
│   ├── checks.py        104 mục kiểm tra
│   └── __main__.py      cửa vào
├── viewer/              Bun + three.js, chạy được cả vật lý lẫn model
│   ├── server.ts        máy chủ nhỏ, phục vụ flights/*.json và model.json
│   ├── index.html       trang, hai chế độ
│   ├── src/scene.ts     cảnh 3D: con tàu, luồng khí, vệt bay
│   ├── src/physics.ts   ← bản dịch của vehicle/physics/controller/expert
│   ├── src/model.ts     ← bản dịch của deepseek_lite + policy
│   ├── src/live.ts      vòng chạy sống: vật lý 1000 Hz trong trình duyệt
│   ├── src/main.ts      hai chế độ, thanh chỉnh tham số, HUD
│   ├── src/style.css
│   ├── check-port.ts    bản dịch có khớp bản gốc không (14 mục)
│   ├── check-live.ts    chế độ sống có chạy thật không (13 mục)
│   ├── check-viewer.ts  chế độ phát lại (17 mục)
│   └── shot-viewer.ts   chụp ảnh cảnh 3D ở từng giai đoạn
├── data/                dữ liệu huấn luyện
├── runs/                model đã lưu
└── flights/             chuyến bay ghi ra cho chế độ phát lại
```

Đọc theo thứ tự này là hiểu hết: `vehicle.py` → `physics.py` →
`controller.py` → `expert.py` → `dataset.py` → `policy.py` → `train.py` →
`dagger.py`. Muốn hiểu phần chạy trong trình duyệt thì đọc
`viewer/src/physics.ts` song song với `physics.py` — hai file cố tình viết
giống nhau từng dòng để đối chiếu được.

---

## 10. Điều dự án này **không** làm

Nói thẳng, để không ai tưởng nhầm:

- **Không phải học tăng cường.** Đây là học bắt chước. Model **không bao giờ
  giỏi hơn thầy** — nó chỉ bắt chước, kể cả những chỗ thầy làm dở. Muốn giỏi
  hơn thì phải dùng RL, và đó là một dự án khác.
- **Mô hình vật lý là bản rút gọn.** Lực cản dùng công thức đơn giản, mô-men
  khí động là một số hằng nhân với áp suất động. Đủ để bài toán khó lên, không
  đủ để tin khi phóng thật.
- **Không có nhiễu cảm biến, không có trễ truyền lệnh.** Model đọc trạng thái
  chính xác tuyệt đối, và lệnh của nó có hiệu lực ngay lập tức. Thực tế thì
  không như vậy.
- **Giới hạn 2 chiều nghiêng.** Con tàu đối xứng tròn nên góc "vặn" quanh trục
  dọc không ảnh hưởng gì tới việc bay — bỏ đi cho dễ hiểu.
- **Chạy sống trong trình duyệt là bản dịch, không phải bản gốc.** Nó khớp
  bản Python tới 1e-9 trên 40.000 con số, nhưng vẫn là *hai* bộ mã phải sửa
  cùng nhau. Đó là cái giá của việc chạy được mà không cần Python.
- **Chưa bay thật.** Toàn bộ là mô phỏng trên CPU.
