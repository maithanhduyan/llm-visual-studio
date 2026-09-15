# deepseek_math — dạy một LLM làm toán, từng bước một

> Bốn cấp trước hỏi *"model hoạt động thế nào"*. Dự án này hỏi câu khác:
> ***"model học được kỹ năng gì, và phải dạy theo trình tự nào"***.

Model vẫn là kiến trúc của Cấp 3, chỉ thu nhỏ và đổi bộ từ vựng thành 13-15
ký tự số. Việc học ở đây **không nằm ở kiến trúc** — nằm ở dữ liệu và trình tự.

```bash
cd deepseek_math
python -m deepseek_math steps        # xem năm bậc
python -m deepseek_math demo         # xem bài mẫu
python -m deepseek_math experiment   # chạy thí nghiệm chính (~10 phút)
```

---

## Năm bậc

```text
bậc  học gì                             số bài   cách viết
───  ─────────────────────────────────  ───────  ────────────────
 1   cộng 1 chữ số, không nhớ (tổng ≤9)   55     3+4=7
 2   cộng 1 chữ số, có nhớ (tổng ≥10)     45     7+5=12
 3   cộng 2 chữ số                         ∞     17+25=42
 4   cộng 2 chữ số                         ∞     17+25=[7+5=12][1+2+1=4]=42
 5   cộng 1-3 chữ số                       ∞     (như bậc 4)
```

**Vì sao bậc 1 và 2 tách riêng:** cả bảng cộng một chữ số chỉ có 100 bài.
Model **học thuộc** được, không cần hiểu gì. Tách ra để thấy rõ: học thuộc thì
dễ, mà học thuộc thì không suy ra được bài dài hơn.

**Vì sao bậc 3 và 4 là cùng một tập bài:** đây là thí nghiệm chính. Cùng bài,
cùng model, chỉ khác chỗ có bắt model **viết ra từng bước** hay không.

**Vì sao có bậc 5:** bậc 4 học toàn số hai chữ số. Nó có làm được số ba chữ
số không? Câu trả lời làm bất ngờ — xem phần kết quả.

---

## Dữ liệu huấn luyện ở đâu?

**Không có file dữ liệu nào là bắt buộc — nó được SINH RA ngay lúc học.**

```python
text = build_text(level, examples=20000, seed=seed)   # train.py
```

Đó là chủ ý, và là thứ làm thí nghiệm này làm được: muốn thử thì cứ sinh bài
**mới**, chắc chắn model chưa từng thấy. Các cấp trước phải đi nhặt một bài
văn rồi học thuộc nó, nên phải có `data.txt`. Ở đây thì không.

Nhưng sinh ra lúc chạy thì thiếu hai thứ, nên có thêm hai lệnh:

```bash
python -m deepseek_math export            # ghi dữ liệu cả 5 bậc ra data/
python -m deepseek_math data --show 5     # xem dữ liệu đang có
python -m deepseek_math teach 4 --data data/bac4.txt   # học từ file
```

Repo có sẵn `data/bac1.txt` … `data/bac5.txt`, mỗi file 300 bài, để **mở ra
đọc được ngay**. Ví dụ `data/bac4.txt`:

```text
# Bậc 4 — Cộng hai chữ số, viết từng bước
# 300 bài, hạt giống 11
67+81=[7+1=8][6+8=14]=148
69+67=[9+7=16][6+6+1=13]=136
75+85=[5+5=10][7+8+1=16]=160
```

Một điều dễ hiểu nhầm: **model không học từng dòng**. Nó đọc cả file như một
đoạn văn liền mạch và đoán ký tự tiếp theo. Mỗi dòng chỉ là một bài; dấu
xuống dòng là thứ model học để biết bài nào đã hết.

---

## Cách viết từng bước

```text
   17+25=[7+5=12][1+2+1=4]=42
          │        │
          │        └── cột chục: 1+2+1  (số 1 là số NHỚ từ "12")
          └─────────── cột đơn vị: 7+5=12
```

Ba tính chất khiến định dạng này học được:

**Số nhớ không cần ghi riêng** — nó nằm sẵn trong kết quả cột trước. Chữ số
hàng chục của `12` chính là số nhớ, nên bước sau chỉ việc cộng ba số.

**Đáp án đọc ra được từ chính các bước** — lấy chữ số cuối của mỗi cột, thêm
số nhớ cuối cùng.

**Cột nào cũng giống cột nào** — khác độ dài số thì chỉ khác SỐ cột, không
khác hình dạng.

---

## Ba câu hỏi mà thí nghiệm trả lời

```bash
python -m deepseek_math experiment
```

Nó dạy cả năm model rồi thử từng model với số **2, 3, và 4 chữ số** — toàn bài
chưa từng thấy. Bảng kết quả in ra theo dạng:

```text
bậc  học gì                               2 chữ số   3 chữ số   4 chữ số
───  ──────────────────────────────────  ─────────  ─────────  ─────────
 1   Cộng một chữ số, không nhớ               0.0%       0.0%       0.0%
 2   Cộng một chữ số, có nhớ                  0.0%       0.0%       0.0%
 3   Cộng hai chữ số, trả lời thẳng            ?          ?          ?
 4   Cộng hai chữ số, viết từng bước           ?          ?          ?
 5   Cộng 1-3 chữ số, viết từng bước           ?          ?          ?
```

*(Bảng đầy đủ ở phần Kết quả bên dưới.)*

---

## Kết quả

Xem [`docs/ket-qua.md`](docs/ket-qua.md) — có bảng đầy đủ và phân tích.

```text
model  học gì                            1 chữ số   2 chữ số   3 chữ số   4 chữ số
─────  ────────────────────────────────  ─────────  ─────────  ─────────  ─────────
bậc 1  cộng 1 chữ số, không nhớ            100.0%      0.0%       0.0%       0.0%
bậc 2  cộng 1 chữ số, có nhớ               100.0%      0.0%       0.0%       0.0%
bậc 3  cộng 2 chữ số, trả lời thẳng          0.0%     57.5%       0.0%       0.0%
bậc 4  cộng 2 chữ số, viết từng bước         0.0%    100.0%       0.0%       0.0%
bậc 5  cộng 1-3 chữ số, viết từng bước       8.7%    100.0%      14.7%       0.0%
```

Bốn phát hiện:

1. **Học thuộc bảng cộng không suy ra được gì.** Bậc 1 và 2 đạt 100% trên bảng
   đã học, và **0%** với số hai chữ số.

2. **Viết ra từng bước thì làm được.** Cùng bài hai chữ số: trả lời thẳng được
   **57,5%**, viết từng bước được **100%** — và loss thấp hơn một nửa (0,48 so
   với 1,09).

3. **Học toàn số hai chữ số thì không làm được số ba chữ số** — dù có viết
   từng bước. Bậc 4 không hề "cố rồi sai": nó viết **hai cột** cho bài ba chữ
   số, vì nó học được đúng câu "bài này có hai cột".

4. **Cho nhiều độ dài lúc học thì suy rộng ra được** — bậc 5 viết đúng **ba
   cột** và làm đúng 14,7% (bậc 4: 0%). Nhưng nó còn đọc nhầm chữ số ở vị trí,
   nên chưa giỏi. Rào cản về **thuật toán** đã vượt qua; còn vướng ở phần
   **đọc dữ liệu vào**.

Đọc thêm: [`docs/vi-sao-kho.md`](docs/vi-sao-kho.md) — vì sao LLM làm toán khó,
và vì sao viết ra từng bước lại chữa được.

---

## Cấu trúc dự án

```text
deepseek_math/
├── pyproject.toml
├── README.md
├── docs/
│   ├── vi-sao-kho.md       vì sao LLM làm toán khó
│   └── ket-qua.md          bảng kết quả và phân tích
│
├── deepseek_math/
│   ├── __init__.py
│   ├── __main__.py         CLI: steps / demo / data / export / teach / test / experiment
│   ├── paths.py
│   ├── base.py             lấy kiến trúc của Cấp 3
│   ├── problems.py         sinh bài toán + hai cách viết + đọc đáp án
│   ├── curriculum.py       năm bậc
│   ├── dataset.py          ghi dữ liệu sinh ra xuống file, và đọc lại
│   ├── model.py            model nhỏ (366 nghìn thông số)
│   ├── train.py            dạy một bậc (sinh dữ liệu, hoặc đọc từ file)
│   ├── evaluate.py         chấm điểm bằng bài MỚI
│   └── checks.py           thí nghiệm chính
├── data/                   dữ liệu mẫu, sinh sẵn để mở mà đọc
└── runs/                   model sau khi học
```

---

## Dạy và thử từng bậc

```bash
python -m deepseek_math teach 4              # dạy bậc 4
python -m deepseek_math test 4               # thử bậc 4
python -m deepseek_math test 4 --count 500   # thử nhiều bài hơn
```

Model chỉ có **366.732 thông số** và học một bậc trong **~2 phút** trên CPU.
Nhỏ vậy là cố ý: muốn thấy rõ cái gì đến từ **dữ liệu** chứ không phải từ
**kích thước model**.
