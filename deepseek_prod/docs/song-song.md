# Song song hoá — chia model ra nhiều máy

Cấp 4 có bốn cách chia, và chúng không thay thế nhau mà **xếp chồng** được.
Tài liệu này giải thích từng cách, cái giá phải trả, và cách chúng phối hợp.

```text
                    ┌─────────────────────────────────┐
                    │  world_size = tp × pp × dp      │
                    └─────────────────────────────────┘

    tp = 2, pp = 2, dp = 1  ->  4 máy

              dp = 0
    ┌───────────────────────────────────────┐
    │            pp = 0                     │
    │   máy 0 (tp 0)      máy 1 (tp 1)      │   tầng 0-1
    ├───────────────────────────────────────┤
    │            pp = 1                     │
    │   máy 2 (tp 0)      máy 3 (tp 1)      │   tầng 2
    └───────────────────────────────────────┘
```

---

## 1. Data parallel (dp)

**Chia gì:** dữ liệu. Máy nào cũng giữ **nguyên** model, mỗi máy đọc một lô
dữ liệu khác nhau.

**Nói chuyện gì:** sau mỗi bước, `all_reduce` gradient rồi chia cho `dp`.

**Khi nào dùng:** model nhét vừa một máy, nhưng muốn học nhanh hơn.

**Cái giá:** mỗi máy phải chứa đủ model + gradient + trạng thái optimizer.
Model 1 tỷ thông số thì mỗi máy tốn khoảng 16 GB. Đây chính là lý do ZeRO ra đời.

---

## 2. Tensor parallel (tp) — cắt NGANG

**Chia gì:** từng phép nhân. Một ma trận `[out, in]` bị chẻ đôi theo cột hoặc
theo hàng.

```text
    CỘT:  y = x · Wᵀ          HÀNG:  y = x · Wᵀ
    W chia theo chiều ra             W chia theo chiều vào

    máy 0: W[0:out/2, :]            máy 0: W[:, 0:in/2]
    máy 1: W[out/2:out, :]          máy 1: W[:, in/2:in]

    y₀ = x · W₀ᵀ  (nửa đầu)         y₀ = x₀ · W₀ᵀ  (một phần của tổng)
    y₁ = x · W₁ᵀ  (nửa sau)         y₁ = x₁ · W₁ᵀ
         │                                │
    không cần nói chuyện            all_reduce -> y đầy đủ
```

**Vì sao ghép hai kiểu:** một tầng transformer thường có dạng
"cột → hoạt hoá → hàng". Ghép vậy thì **mỗi tầng chỉ phải nói chuyện đúng một
lần**, thay vì sau mỗi phép nhân.

**Nói chuyện gì:** `all_reduce` sau mỗi lớp hàng (row-parallel).

**Trong model này:** `q_proj`, `k_proj`, `v_proj` cắt theo cột — mỗi máy giữ
vài đầu (head). `out_proj` cắt theo hàng. Các chuyên gia SwiGLU cũng vậy.

**Ba chi tiết dễ sai:**

1. **Bias của lớp hàng chỉ được cộng MỘT lần**, sau `all_reduce`. Máy nào
   cũng cộng thì bias bị nhân lên `tp` lần.
2. **Số đầu phải chia theo.** Máy giữ 3 đầu Q thì `n_heads` phải là 3, không
   phải 6. Nếu không, lúc ghép các đầu lại sẽ sai kích thước.
3. **Router của MoE KHÔNG được cắt.** Mọi máy phải chọn cùng một chuyên gia.
   Hai máy chọn khác nhau thì cộng lại ra kết quả vô nghĩa.

---

## 3. Pipeline parallel (pp) — cắt DỌC

**Chia gì:** các tầng. Máy 0 giữ tầng 0-1, máy 1 giữ tầng 2, ...

**Vấn đề:** làm tuần tự thì lúc nào cũng chỉ một máy chạy.

**Giải pháp:** chia lô thành nhiều **vi lô** rồi bơm lần lượt.

```text
    thời gian ->
    máy 0:  [m0][m1][m2][m3]                [b3][b2][b1][b0]
    máy 1:      [m0][m1][m2][m3]        [b3][b2][b1][b0]
    máy 2:          [m0][m1][m2][m3]  ...
```

Cách này gọi là **GPipe**. Bản tinh chỉnh hơn là **1F1B** (xen kẽ forward và
backward để đỡ tốn bộ nhớ) — ở đây làm GPipe cho dễ hiểu.

**Hai lỗi đã sập khi làm phần này, cả hai đều rất đáng nhớ:**

**Lỗi 1 — `recv` trần làm đứt đồ thị.** Nhận activation bằng `dist.recv` ghi
thẳng vào một tensor thường thì **không có nút autograd nào ở đó**. `loss.backward()`
sẽ chạy ngược tới chỗ đó rồi **dừng**, không gửi gradient về máy trước. Máy
trước ngồi chờ mãi. Phải bọc `recv` thành một `torch.autograd.Function`, và
chiều ngược của nó chính là `send` gradient.

**Lỗi 2 — thiếu `requires_grad` thì `autograd.Function` không dựng đồ thị.**
Đã bọc thành `autograd.Function` rồi mà vẫn đứt, vì đầu vào là
`torch.empty(...)` — không cần gradient thì autograd **không tạo `grad_fn`**
cho đầu ra. Phải cho tensor giữ chỗ đó `requires_grad=True`.

**Lỗi 3 — weight tying bị đứt giữa hai máy.** `deepseek_lite` dùng chung trọng
số giữa Token Embedding và LM Head. Cắt dọc thì hai thứ đó nằm ở hai máy khác
nhau, nên mỗi máy chỉ nhận **một nửa** gradient của bảng tra. Không cộng lại
thì bảng tra học sai — mà **loss vẫn ra đúng**, nên rất khó thấy. Chỉ phát
hiện được vì bài kiểm tra so cả gradient chứ không chỉ so loss.

---

## 4. Expert parallel (ep) — mỗi máy vài chuyên gia

**Chia gì:** các chuyên gia trong MoE. Máy 0 giữ chuyên gia 0-3, máy 1 giữ 4-7.

**Vì sao hợp lý:** mỗi token chỉ hỏi 2 trong 8 chuyên gia. Bắt máy nào cũng
giữ cả 8 là thừa.

**Nói chuyện gì:** `all_reduce` phần kết quả, vì mỗi máy chỉ tính được phần
của mình.

**Một cái bẫy rất khó thấy:** đầu ra của MoE được **cộng vào dòng residual**,
mà dòng residual thì máy nào cũng giữ một bản. Nếu máy 0 trả
"chuyên gia dùng chung + phần của mình" còn máy 1 chỉ trả "phần của mình" thì
hai máy **trôi đi hai hướng khác nhau**, và mọi tầng sau tính trên hai dữ
liệu khác nhau. Phải gom tất cả vào một tensor rồi `all_reduce`, để máy nào
cũng ra **đúng một kết quả**.

**Khác gì cách làm thật:** hệ thật dùng `all-to-all` để chỉ gửi token tới
đúng máy cần. Ở đây máy nào cũng giữ cả bó token rồi trả phần lớn bằng 0 —
kết quả y hệt, chỉ tốn băng thông hơn.

---

## 5. ZeRO — chia trạng thái optimizer

Adam giữ cho mỗi tham số **hai** con số nữa. Model 1 tỷ thông số thì optimizer
chiếm thêm 8 GB.

Các máy trong nhóm `dp` vốn giữ bản sao giống hệt nhau, nên trạng thái
optimizer của chúng cũng giống hệt nhau — vô ích. ZeRO chia ra:

```text
    1. mọi máy tính gradient trên lô của mình
    2. all_reduce gradient          -> máy nào cũng có gradient đầy đủ
    3. MỖI MÁY CHỈ CẬP NHẬT PHẦN CỦA MÌNH
    4. broadcast phần vừa cập nhật  -> model lại giống nhau
    5. quay lại bước 1
```

**Một cái bẫy:** bước 4 phải lặp qua **mọi** tham số trên **mọi** máy, theo
cùng thứ tự, với cùng một `src`. Nếu máy nào cũng chỉ phát phần của mình thì
hai máy gọi `broadcast` trên hai tensor khác nhau — không khớp và treo.

| Giai đoạn | Chia thêm gì | Bộ nhớ mỗi máy |
| --- | --- | --- |
| ZeRO-1 | trạng thái optimizer | model + gradient + 1/dp optimizer |
| ZeRO-2 | thêm gradient | model + 1/dp (gradient + optimizer) |
| ZeRO-3 | thêm cả tham số | 1/dp tất cả |

Ở đây làm ZeRO-1. Kết quả đo: trạng thái optimizer từ 13.456 KB xuống
7.445 KB, và model sau 3 bước **giống hệt** cách chạy thường (lệch 0,00e+00).

---

## Xếp chồng thì sao?

Bốn cách này **nhân với nhau**: `world = tp × pp × dp`, còn `ep` dùng chung
nhóm `tp`. Ví dụ `tp=2, pp=2, dp=2` là 8 máy, mỗi máy giữ một phần tư model
(tp×pp) và có hai bản sao dữ liệu (dp).

Hệ thật còn có **context parallel** (chia cả chiều dài câu) và **sequence
parallel** — không làm ở đây vì model này câu chỉ dài 512 token.

---

## Vì sao phải kiểm tra bằng SỐ chứ không bằng mắt

Cắt model ra nhiều máy mà nhìn loss thấy bình thường thì **chưa chứng minh
được gì**. Ba trong bốn lỗi ở trên đều cho ra loss trông rất hợp lý:

| Lỗi | Loss | Gradient |
| --- | --- | --- |
| `recv` trần | treo | — |
| thiếu `requires_grad` | treo | — |
| weight tying đứt | **đúng** | sai 37% |
| EP trả khác nhau giữa các máy | vẫn hội tụ | sai dần |

Nên `checks.py` so **cả loss lẫn gradient**, và so trên **từng tham số** chứ
không so một con số tổng.
