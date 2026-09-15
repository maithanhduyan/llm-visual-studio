# Vì sao LLM làm toán khó

Không phải vì máy tính không biết cộng. Máy tính cộng giỏi hơn người. Khó là
khó ở chỗ **model phải tự nghĩ ra phép cộng bằng chính cơ chế của nó** — và cơ
chế đó không được thiết kế để tính.

---

## 1. Model không có vòng lặp

Một chương trình cộng hai số trông như thế này:

```python
nho = 0
for moi_cot in cac_cot:          # <- VÒNG LẶP
    tong = cot_a + cot_b + nho
    ket_qua.append(tong % 10)
    nho = tong // 10
```

Cái `for` là thứ làm nên phép cộng nhiều chữ số. Nó cho phép **làm bao nhiêu
cột cũng được**, tuỳ độ dài số.

Model không có `for`. Nó có đúng `n_layers` tầng, và con số đó cố định:

```text
3 tầng  ->  nhiều nhất 3 bước tính
```

Cộng số hai chữ số cần 2 bước. Ba chữ số cần 3 bước. Mười chữ số cần 10 bước
— mà model chỉ có 3 tầng.

Đây là lý do sâu xa nhất: **model không thể tính ngầm nhiều bước hơn số tầng
nó có.** Số tầng cố định, độ dài số thì không.

---

## 2. Số nhớ làm phép cộng thành bài toán tuần tự

Nếu không có số nhớ, mỗi cột cộng độc lập — model làm cả 10 cột **cùng một
lúc**, một tầng là đủ:

```text
  12         1+3=4
+ 34    ->   2+4=6      hai cột, làm song song được
  ────
  46
```

Nhưng có nhớ thì cột sau **phải chờ** cột trước:

```text
  17         7+5=12  ->  viết 2, NHỚ 1  ─┐
+ 25         1+2+1=4 ->  viết 4         <┘ cột này cần biết cột kia
  ────
  42
```

Không làm song song được nữa. Đây là điểm mà mọi model ngôn ngữ đều vấp.

---

## 3. Cách chữa: bắt model viết ra

Nếu model không tính ngầm được, thì cho nó **giấy nháp**.

```text
   trả lời thẳng:   17+25=42
   viết từng bước:  17+25=[7+5=12][1+2+1=4]=42
```

Điều kỳ diệu nằm ở chỗ: sau khi viết ra `[7+5=12]`, thì bước sau
`[1+2+1=4]` **không còn là bài toán nữa** — nó chỉ là đọc lại và làm theo.
Mọi thứ cần thiết (số 1 để nhớ) đã nằm sẵn trong câu, ở ngay trước mắt.

Nói cách khác: viết ra từng bước biến bài toán **tuần tự trong đầu** thành
bài toán **tuần tự trên giấy**. Mà tuần tự trên giấy thì model làm được, vì
mỗi bước chỉ cần nhìn lại câu phía trước — việc mà attention sinh ra để làm.

Đây chính là ý tưởng của **chain-of-thought**: không làm model thông minh hơn,
mà cho nó chỗ để viết.

---

## 4. Định dạng viết ra rất quan trọng

Không phải cứ viết dài là được. Định dạng phải làm cho **mỗi bước giống hệt
bước trước** — cùng một khuôn, chỉ khác số. Có vậy model mới học được cái
khuôn thay vì học thuộc từng câu.

Định dạng dùng trong dự án này:

```text
17+25=[7+5=12][1+2+1=4]=42
```

Ba tính chất khiến nó học được:

**Số nhớ không cần ghi riêng.** Nó nằm sẵn trong kết quả cột trước: chữ số
hàng chục của `12` chính là số nhớ. Bước sau chỉ việc cộng ba số.

**Đáp án đọc ra được từ chính các bước.** Lấy chữ số cuối của mỗi cột, thêm
số nhớ cuối:

```text
  95+15=[5+5=10][9+1+1=11]=110
          │        │  │
          │        │  └── nhớ cuối = 1
          │        └───── chữ số cuối cột 2 = 1
          └────────────── chữ số cuối cột 1 = 0
                          110
```

**Cột nào cũng giống cột nào.** Khác độ dài số thì chỉ khác SỐ cột, không
khác hình dạng. Nên số ba chữ số chỉ là "thêm một cột nữa" — hợp lý hơn hẳn
so với việc phải đoán một đáp án có bốn chữ số.

---

## 5. Cái gì còn khó, kể cả có giấy nháp

Viết ra từng bước **không** giải quyết được mọi thứ. Ba chỗ còn khó:

**Số quá dài.** Model vẫn phải viết từng cột, và mỗi cột tốn một bước sinh
chữ. Cộng hai số 20 chữ số là 20 bước — làm được, nhưng chậm, và chỉ cần sai
một cột là sai hết.

**Số thập phân.** Dấu phẩy không thẳng cột như chữ số. Phải học thêm một quy
tắc nữa (canh dấu phẩy trước, rồi cộng như số nguyên).

**Số âm.** Dấu trừ đổi hoàn toàn cách làm: phải so sánh hai số trước, rồi
quyết định kết quả âm hay dương. Thêm một tầng logic nữa.

Ba chỗ này đều giải quyết được bằng cùng một cách: **cho nhiều ví dụ hơn, và
viết rõ quy tắc ra thành từng bước.**

---

## 6. Vì sao dự án này tách số một chữ số ra làm hai bậc

Bảng cộng một chữ số chỉ có **100 bài**. Model học thuộc được, và khi đã
thuộc rồi thì đạt gần 100%.

Nhưng đó là **trí nhớ**, không phải **phép tính**. Cách phân biệt rất đơn
giản: cho nó bài dài hơn.

```text
   9+7=16        thuộc bảng cộng    ->  dễ
  19+27=46       phải làm hai cột   ->  khó
 119+227=346     phải làm ba cột    ->  rất khó, nếu chưa được dạy cách viết
```

Một model thuộc lòng bảng cộng sẽ làm đúng `9+7` và **sai** `119+227`. Nó
không có gì để suy ra cả — trong đầu nó, `119+227` chỉ là một dãy ký tự chưa
từng thấy.

Đó là lý do bậc 1 và 2 tách riêng, và đó là lý do bậc 5 tồn tại: để thử xem
model học được **quy tắc** hay chỉ học được **đáp án**.
