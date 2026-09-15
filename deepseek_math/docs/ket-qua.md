# Kết quả

Năm model, mỗi model 366.732 thông số, học 1200 bước (~2 phút trên CPU).
Thử bằng bài **chưa từng thấy**, 120-200 bài mỗi ô.

---

## Bảng chính

```text
model  học gì                            1 chữ số   2 chữ số   3 chữ số   4 chữ số
─────  ────────────────────────────────  ─────────  ─────────  ─────────  ─────────
bậc 1  cộng 1 chữ số, không nhớ (55 bài)   100.0%      0.0%       0.0%       0.0%
bậc 2  cộng 1 chữ số, có nhớ (45 bài)      100.0%      0.0%       0.0%       0.0%
bậc 3  cộng 2 chữ số, trả lời thẳng          0.0%     57.5%       0.0%       0.0%
bậc 4  cộng 2 chữ số, viết từng bước         0.0%    100.0%       0.0%       0.0%
bậc 5  cộng 1-3 chữ số, viết từng bước       8.7%    100.0%      14.7%       0.0%
```

Cột "1 chữ số" của bậc 3-5 là 0% (và 8,7%) vì các model đó **chưa từng học số
một chữ số** — chúng chỉ học số hai chữ số. Đó cũng là một điểm đáng chú ý:
model học số hai chữ số không tự làm được số một chữ số.

---

## Phát hiện 1 — Học thuộc không suy ra được gì

Bậc 1 và bậc 2 đạt **100%** trên bảng cộng đã học, và **0%** với số hai chữ số.

Cả hai model đều đạt 100% trên chính bảng của mình. Nhưng cho chúng bài
`47+25` thì cả hai đều thất bại hoàn toàn.

Điều này nghe hiển nhiên, nhưng nó là **định nghĩa của việc học vẹt**: model
không có gì để suy ra cả. Trong "đầu" nó, `47+25` chỉ là một dãy ký tự chưa
từng thấy, không liên quan gì tới `4+7` mà nó đã thuộc.

> Một model đạt 100% bảng cộng **không** có nghĩa là nó biết cộng.

---

## Phát hiện 2 — Viết ra từng bước thì làm được

Cùng một tập bài (số hai chữ số), cùng một kiến trúc, cùng số bước học.
Chỉ khác chỗ có bắt model viết ra từng cột hay không:

| Cách dạy | Làm đúng số 2 chữ số | loss cuối |
| --- | --- | --- |
| Trả lời thẳng `17+25=42` | **57,5%** | 1,087 |
| Viết từng bước `17+25=[7+5=12][1+2+1=4]=42` | **100%** | 0,484 |

Viết ra từng bước vừa **đúng hơn 42 điểm phần trăm**, vừa có **loss thấp hơn
một nửa**. Loss thấp hơn nghĩa là bài toán trở nên dễ đoán hơn — model không
phải đoán mò, nó chỉ việc đọc bước trước rồi viết bước sau.

Vì sao: cộng có nhớ là bài toán **tuần tự**. Cột sau cần biết số nhớ của cột
trước. Model không có vòng lặp bên trong, số tầng thì cố định, nên không tính
ngầm được. Viết ra giấy thì mọi thứ cần thiết đã nằm ngay trước mắt.

---

## Phát hiện 3 — Học toàn số hai chữ số thì không làm được số ba chữ số

Đây là chỗ bất ngờ nhất. Bậc 4 đạt **100%** với số hai chữ số, nhưng **0%**
với số ba chữ số. Nhìn kỹ thì thấy nó không hề "cố rồi sai" — nó **không biết
là phải làm thêm cột**:

```text
bài:  132+345      đáp án đúng: 477

bậc 4 viết:  [3+5=8][9+3=12]=128
             ^^^^^^ ^^^^^^^^
             chỉ có HAI cột, dù bài có ba chữ số
```

Nó học được đúng câu **"bài toán này có hai cột"**, chứ không học được quy tắc
**"số cột bằng số chữ số"**. Cả hai cách viết đều cho ra hai cột trong suốt
quá trình học, nên với model, "hai cột" chính là một phần của bài toán.

Đây là bài học quan trọng nhất của cả dự án:

> Dạy model một **quy tắc** thì nó suy rộng được.
> Dạy model một **độ dài cụ thể** thì nó học luôn cái độ dài đó như một phần
> của đề bài.

Và đây là lý do bậc 5 tồn tại: cho nó thấy nhiều độ dài khác nhau lúc học.

---

## Phát hiện 4 — Cho nhiều độ dài thì suy rộng được, nhưng chưa đủ giỏi

Bậc 5 học số 1, 2 và 3 chữ số trộn lẫn (4500 bước, gấp gần bốn lần bậc 4).
So sánh trực tiếp với bậc 4:

| | 1 chữ số | 2 chữ số | **3 chữ số** | 4 chữ số |
| --- | --- | --- | --- | --- |
| bậc 4 (chỉ học 2 chữ số) | 0% | 100% | **0%** | 0% |
| bậc 5 (học trộn 1-3 chữ số) | 8,7% | 100% | **14,7%** | 0% |

Bậc 5 **không** giỏi bằng bậc 4 ở số hai chữ số — cả hai đều 100% — nhưng ở số
ba chữ số thì bậc 4 đứng ở 0% còn bậc 5 làm được 14,7%.

Điều đáng chú ý là **cách nó sai**. Bậc 4 viết hai cột cho bài ba chữ số; bậc 5
viết **đúng ba cột**:

```text
bài:  754+214      đáp án đúng: 968

bậc 5 viết:  [4+4=8][7+1=8][7+2=9]=988
             ^^^^^^^ ^^^^^^^ ^^^^^^^
             ba cột — đếm đúng rồi!
                      ^^^^^^^
                      nhưng đọc nhầm chữ số: phải là 5+1, không phải 7+1
```

Nó đã học được **quy tắc "số cột bằng số chữ số"** — đó là điều bậc 4 không
học được. Cái còn thiếu là **đọc đúng chữ số ở đúng vị trí**, và phần đó thì
cần model to hơn hoặc học lâu hơn.

Nói cách khác: bậc 5 đã vượt qua được rào cản về **thuật toán**, nhưng còn
vướng ở phần **đọc dữ liệu vào**.

---

## Tóm lại: trình tự dạy

```text
  bậc 1-2   bảng cộng một chữ số         học thuộc là xong, KHÔNG suy rộng
     │
     │  nhận ra: thuộc lòng không giúp gì cho bài dài hơn
     v
  bậc 3     số hai chữ số, trả lời thẳng  phải TÍNH, mà tính ngầm thì không xong
     │
     │  nhận ra: model cần chỗ để viết
     v
  bậc 4     số hai chữ số, viết từng bước 100% — nhưng chỉ với hai chữ số
     │
     │  nhận ra: nó học luôn "có hai cột" như một phần của đề bài
     v
  bậc 5     nhiều độ dài, viết từng bước   phải học QUY TẮC, không học độ dài
```

Mỗi bậc không chỉ thêm dữ liệu — nó sửa một **hiểu sai cụ thể** của bậc trước.
Đó là toàn bộ ý tưởng của việc dạy theo trình tự.
