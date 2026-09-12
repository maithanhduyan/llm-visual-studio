# Những gì không chạy được trên máy này

Cấp 4 nói về quy mô, mà máy này chỉ có **12 nhân CPU, không GPU**. Nên có
những kỹ thuật trong roadmap **không thể làm thật** ở đây.

Tài liệu này liệt kê thẳng, kèm cách tự kiểm chứng là máy thật sự không làm
được — chứ không phải tôi lười.

---

## 1. FlashAttention

**Là gì:** một cách tính attention nhanh hơn và tốn ít bộ nhớ hơn, bằng cách
không bao giờ ghi cả ma trận `T × T` ra bộ nhớ.

**Vì sao không làm được:** cần GPU NVIDIA. PyTorch có sẵn cơ chế chọn backend:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    y = F.scaled_dot_product_attention(q, k, v)
```

Trên CPU, đoạn này báo lỗi vì backend đó không tồn tại. Cách tự kiểm:

```bash
python -c "import torch; print(torch.backends.cuda.is_built())"
# -> False
```

**Trong dự án này:** `deepseek_lite/attention.py` dùng
`F.scaled_dot_product_attention` — đúng hàm mà hệ thật dùng. Trên GPU nó sẽ
tự chọn kernel flash. Trên CPU nó rơi về bản `math`. **Cùng một dòng code**,
chỉ khác phần cứng.

---

## 2. FP8

**Là gì:** lưu và tính toán bằng số 8 bit thay vì 16 hay 32. DeepSeek-V3 huấn
luyện bằng FP8.

**Vì sao không làm được:** PyTorch có kiểu dữ liệu:

```python
torch.float8_e4m3fn   # có, tạo được tensor
```

Nhưng **tính toán** trên kiểu đó cần GPU Hopper (H100) trở lên. Trên CPU chỉ
tạo được tensor rồi phải đổi về float để tính — vô nghĩa.

**Làm được gì thay thế:** int8, xem `quantize.py`. Ý tưởng giống nhau (nén
trọng số xuống 8 bit), chỉ khác cách biểu diễn số. Kết quả đo: model nhỏ đi
**3,5 lần**, chữ mà model chọn giống **100%**.

Điểm đáng chú ý: trên CPU, int8 **không nhanh hơn** mà còn chậm hơn (16 ms →
29 ms). Nén 8 bit giúp **bộ nhớ**, không giúp tốc độ — trên GPU thì khác, vì
ở đó nút thắt là băng thông bộ nhớ chứ không phải số phép tính.

---

## 3. 1M context

**Là gì:** model nhớ được cả một cuốn sách.

**Vì sao không làm được ở đây:** không phải vì phần cứng, mà vì **chưa huấn
luyện**. Model này học ở 512 token. Muốn nó nhớ 1 triệu token thì phải:

1. Đổi cách tính vị trí (RoPE scaling — NTK / YaRN)
2. **Huấn luyện lại ở ngữ cảnh dài** — bước này mới là bước tốn kém

Chỉ đổi RoPE scaling mà không huấn luyện lại thì model nhận được câu dài hơn
nhưng **không hiểu gì** ở đoạn xa — vì những vị trí đó chưa từng xuất hiện
lúc học.

**Làm được gì thay thế:** `deepseek_lite` có **cửa sổ trượt** và **đệm K/V
chia khối**, nên về mặt kỹ thuật nó sinh được câu dài tuỳ ý. Nhưng "sinh được"
khác "hiểu được".

---

## 4. Speculative decoding

**Là gì:** dùng một model nhỏ đoán trước 4-5 chữ, rồi model to kiểm lại một
lượt. Đoán đúng thì được 4-5 chữ với giá của 1.

**Vì sao không làm ở đây:** kỹ thuật này **chạy được trên CPU**, nhưng cần có
sẵn một model nháp nhỏ hơn. Trong repo này:

| Model | Thông số | Làm nháp cho ai? |
| --- | --- | --- |
| `tiny_gpt` | 348.960 | `mini_deepseek`, `deepseek_lite` |
| `mini_deepseek` | 3.442.832 | `deepseek_lite` |
| `deepseek_lite` | 1.833.624 | — |

Vấn đề: cả ba đều được huấn luyện trên **bộ từ vựng khác nhau** (74 / 88 /
145 ký tự). Model nháp phải dùng chung từ vựng với model chính thì mới so
được. Nên muốn làm thật thì phải huấn luyện một cặp model cùng từ vựng.

Đây là việc làm được, chỉ chưa làm.

---

## 5. MLA và MTP

**MLA (Multi-head Latent Attention)** — nén K/V thành một vector latent nhỏ
thay vì cắt bớt số đầu như GQA. Đây là kỹ thuật đặc trưng nhất của DeepSeek-V2/V3.

**MTP (Multi-Token Prediction)** — đoán nhiều chữ một lúc, dùng làm đầu phụ
lúc huấn luyện.

Cả hai **làm được trên CPU**, nhưng mỗi cái là một thay đổi kiến trúc đủ lớn
để đáng một dự án riêng. Ở đây chọn GQA (đơn giản hơn, cùng mục đích: làm
đệm K/V nhỏ đi) và không làm MTP.

---

## 6. Chia model ra nhiều MÁY thật

**Đã làm:** nhiều **tiến trình** trên cùng một máy, nói chuyện qua `gloo`.

**Chưa làm:** nhiều máy thật, nói chuyện qua `nccl` và mạng InfiniBand.

Sự khác biệt không chỉ ở tốc độ:

| | Cùng một máy (gloo) | Nhiều máy (nccl) |
| --- | --- | --- |
| Băng thông | ~10 GB/s | ~100 GB/s (NVLink) hoặc ~25 GB/s (InfiniBand) |
| Độ trễ | rất thấp | cao hơn nhiều |
| Hỏng hóc | hiếm | máy chết là chuyện thường, phải có checkpoint và khởi động lại |

Ở quy mô nhiều máy, **độ trễ mạng** trở thành nút thắt thật, và người ta phải
đổi cả cách chia (ít tensor parallel hơn, nhiều pipeline parallel hơn) để đỡ
phải nói chuyện nhiều. Đó là lý do `tp` thường chỉ dùng trong cùng một máy.

---

## 7. Windows: `torchrun` không chạy được

Không phải giới hạn của Cấp 4 mà là của Windows. `torchrun --standalone` báo:

```
RendezvousConnectionError: The connection to the C10d store has failed.
```

và `multiprocessing.spawn` thì treo. Cách chạy được (xem `launcher.py`):

```python
multiprocessing.Process  +  init_method="file:///..."
```

Nếu bạn chạy dự án này trên Linux, có thể thay `launcher.py` bằng `torchrun`
bình thường — phần còn lại không đổi.

---

## Vậy Cấp 4 này có thật không?

Có. Bảy kỹ thuật đã làm đều **chạy thật và được đo thật**:

- Bốn cách chia model, mỗi cách đều cho ra **kết quả y hệt** chạy một máy
- Đệm chia khối, continuous batching, nén 8 bit — đều có số đo

Bốn kỹ thuật không làm được vì **phần cứng** (FlashAttention, FP8) hoặc vì
**cần huấn luyện lại ở quy mô lớn** (1M context, và model nháp cho speculative
decoding).

Điều quan trọng là **cách suy nghĩ thì giống hệt**. Cắt một ma trận ra hai
máy, cho gradient đi ngược qua đường ống, chia trạng thái optimizer, cấp phát
bộ nhớ theo khối — đó là những bài toán thật, và ở đây chúng được giải thật,
chỉ trên quy mô nhỏ hơn.
