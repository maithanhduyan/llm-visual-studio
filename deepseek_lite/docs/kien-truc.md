# Kiến trúc Cấp 3

Sáu ý tưởng mà Cấp 3 thêm vào so với Cấp 2. Mỗi ý tưởng trả lời một câu hỏi,
và mỗi ý tưởng nằm trong đúng một file.

---

## 1. GQA — Grouped-Query Attention

**Câu hỏi: làm sao để bộ nhớ đệm nhỏ hơn mà không mất chất lượng?**

Ở Cấp 2, mỗi tầng có 6 đầu Q, 6 đầu K, 6 đầu V. Nhưng khi sinh chữ, thứ phải
cất vào đệm là K và V — và đệm đó phình ra theo độ dài câu.

GQA cho các đầu Q dùng chung K/V:

```text
Cấp 2 (MHA)          Cấp 3 (GQA)
Q0 Q1 Q2 Q3 Q4 Q5    Q0 Q1 Q2 Q3 Q4 Q5
K0 K1 K2 K3 K4 K5    K0       K1
V0 V1 V2 V3 V4 V5    V0       V1
                     \_____/
                  3 đầu Q dùng chung 1 cặp K/V
```

Đệm nhỏ đi `n_heads / n_kv_heads` lần — ở đây là 3 lần. Chất lượng gần như
không giảm, vì K và V của các đầu vốn đã khá giống nhau.

Trong code: `attention.py`, ba dòng `q_proj` / `k_proj` / `v_proj` có kích
thước khác nhau, và một dòng `repeat_interleave` để nhân bản K/V lên cho đủ.

---

## 2. Cửa sổ trượt — Sliding Window

**Câu hỏi: câu dài thì làm sao cho đỡ mệt?**

Đọc một câu dài, em không cần nhớ hết cả câu — em chỉ cần nhớ đoạn vừa đọc.
Mỗi token chỉ được nhìn lại `window` token gần nhất.

Điểm dễ bị hiểu sai: **che bằng mặt nạ thì không nhanh hơn**. Ma trận attention
vẫn là T×T, chỉ là phần bị che được nhân với 0. Muốn nhanh thật thì phải
**cắt hẳn** K/V cũ đi.

Trong `attention.py` có một đoạn làm đúng việc đó:

```python
if cache is not None and self.window > 0 and T == 1:
    if available > self.window:
        k = k[:, :, -self.window:]
        v = v[:, :, -self.window:]
```

Chỉ cắt được khi `T == 1` (đang sinh từng chữ). Lúc học thì `T` lớn, token đầu
tiên trong đoạn vẫn cần nhìn xa hơn, nên không cắt được.

Có một bài test riêng cho việc này: `test_cache_gives_identical_results` chạy
với `window=4` để chắc rằng bản cắt và bản không cắt cho ra kết quả giống hệt.

---

## 3. KV cache

**Câu hỏi: viết chữ thứ 100 có phải tính lại 99 chữ trước không?**

Không. K và V của token cũ không bao giờ thay đổi, nên tính một lần rồi cất đi.

```text
Không đệm:  "Học"           -> tính 1 token
            "Học mãi"       -> tính 2 token
            "Học mãi thì"   -> tính 3 token      tổng: 1+2+3+... = O(n²)

Có đệm:     "Học"           -> tính 1 token, cất K/V
            "Học mãi"       -> tính 1 token mới, tra K/V cũ
            "Học mãi thì"   -> tính 1 token mới, tra K/V cũ    tổng: O(n)
```

Q thì không cất, vì Q là câu hỏi của token mới; token cũ hỏi xong rồi thì thôi.

Đệm còn cần RoPE biết vị trí thật: sinh chữ thứ 100 thì nó phải ở vị trí 100,
không phải vị trí 0. Đó là lý do `rope.py` có tham số `start_pos`.

Bằng chứng: `python -m deepseek_lite benchmark` in ra kết quả sinh chữ có đệm
và không đệm, rồi khẳng định chúng giống hệt nhau.

---

## 4. Chuyên gia dùng chung — Shared Expert

**Câu hỏi: có việc gì mà token nào cũng cần không?**

Có. Giữ cho câu văn trôi chảy, nhớ giới từ, nhớ dấu câu — việc cơ bản mà
token nào cũng cần. Nếu bắt router phải chọn một chuyên gia cho việc đó ở
mọi token thì vừa phí vừa dễ chọn sai.

Nên DeepSeek-V3 để một chuyên gia **ai cũng hỏi**, không qua router:

```text
Input
  |
  +-------------------------> Chuyên gia dùng chung   (luôn chạy)
  |
  +--> Router --> 2 / 8 chuyên gia riêng --> cộng lại
  |
Output
```

Cấp 2 không có thứ này. Cấp 3 có `n_shared = 1`.

---

## 5. Chia việc không cần hàm phạt

**Câu hỏi: làm sao bắt router chia việc đều mà không làm hỏng loss?**

Cấp 2 dùng cách thông thường: cộng thêm một hàm phạt vào loss.

```python
# Cấp 2
loss = loss_dự_đoán + 0.01 * aux_loss      # <- aux_loss làm lệch loss chính
```

Cách này có một nhược điểm thật: model bị buộc phải vừa đoán chữ vừa giữ
router cân bằng, hai việc không liên quan đến nhau.

DeepSeek-V3 làm khác. Không đụng vào loss. Thay vào đó mỗi chuyên gia có một
**điểm thiên vị**, chỉnh tay sau mỗi bước học:

```python
load   = số_lần_được_chọn / tổng_số_lần_chọn
target = 1 / n_experts
expert_bias += tốc_độ * sign(target - load)
```

Chuyên gia nào nhận nhiều hơn mức đều thì `target - load < 0` → bị trừ điểm.
Chuyên gia ít việc thì được cộng điểm. Ba dòng, không gradient, không đụng loss.

Trong `moe.py`, đó là hàm `update_bias()`. Trong `train.py`, nó được gọi ngay
sau `optimizer.step()`.

---

## 6. Hyper-connections — nhiều dòng suy nghĩ

**Câu hỏi: một dòng residual có đủ không?**

Cấp 2 chỉ có một dòng:

```text
x -----------------------------> x + việc
```

Cấp 3 giữ `n` dòng, mỗi dòng có tiếng nói riêng (`alpha`) và độ to riêng
khi nhận kết quả (`beta`):

```text
dòng 1 --\
dòng 2 ----> trộn (alpha) --> Attention/MoE --> tản ra (beta)
dòng 3 ---->                                        |
dòng 4 --/                                          v
         ^------------------------------ cộng vào từng dòng
```

Khi `n_streams = 1` thì `alpha = 1`, `beta = 1`, và nó trở về **đúng** đường
tắt bình thường của Cấp 2. Nhờ vậy bật/tắt không cần sửa code ở đâu khác —
có một bài test riêng xác nhận điều này.

Đây là bản đơn giản của hyper-connections. Bản mHC thật trong hình còn có
ràng buộc hình học giữa các dòng — phức tạp hơn nhiều, để Cấp 4.

---

## Tổng kết: một token đi qua Cấp 3

```text
"Học mãi thì"
     |  tokenizer
[41, 62, 88, 41 ...]
     |  embedding
[B, T, 192]
     |  nhân thành 4 dòng
[B, T, 4, 192]
     |
     |  Block x 3
     |    trộn 4 -> 1 dòng
     |    RMSNorm
     |    Attention:  GQA (6Q / 2KV) + RoPE + cửa sổ trượt 128 + KV cache
     |    tản 1 -> 4 dòng
     |    trộn 4 -> 1 dòng
     |    RMSNorm
     |    MoE: 1 chuyên gia dùng chung + chọn 2 / 8 chuyên gia riêng
     |    tản 1 -> 4 dòng
     |
     |  gộp 4 dòng -> 1
RMSNorm
     |  LM Head
điểm cho 145 chữ
     |
chữ tiếp theo
```
