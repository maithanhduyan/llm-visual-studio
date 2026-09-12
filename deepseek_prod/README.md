# deepseek_prod

> **Cấp 4** trong lộ trình: đưa một LLM ra chạy thật.

Cấp 1, 2, 3 trả lời câu hỏi *"model hoạt động thế nào"*.
Cấp 4 trả lời câu hỏi khác hẳn: *"làm sao để nó chạy được ở quy mô lớn"*.

**Cấp 4 không viết lại model.** Nó lấy nguyên model của Cấp 3
(`deepseek_lite`) rồi dựng hai nửa hệ thống quanh nó. Đó là toàn bộ ý tưởng.

```bash
cd deepseek_prod
python -m deepseek_prod info      # xem model và cách chia
python -m deepseek_prod check     # chứng minh từng tính năng bằng số
```

---

## Đã làm gì, và đo được gì

Mỗi dòng dưới đây là một phép đo thật, không phải lời hứa. Chạy
`python -m deepseek_prod check` để tự kiểm lại.

### A. Huấn luyện ở quy mô

| Kỹ thuật | Chứng minh | Kết quả đo |
| --- | --- | --- |
| **Tensor parallel** | 2 máy cắt ngang, so logits với 1 máy | lệch **4,3e-06** · mỗi máy giữ 936.600 thông số · 3 đầu Q thay vì 6 |
| **Pipeline parallel** | 2 máy cắt dọc, so **loss và gradient** với 1 máy | loss lệch **9,5e-07** · gradient lệch **4,2e-07** |
| **Expert parallel** | 2 máy chia 8 chuyên gia | lệch **3,3e-06** · mỗi máy giữ 4/8 chuyên gia |
| **ZeRO-1** | chia trạng thái optimizer cho 2 máy | model sau 3 bước giống hệt (**0,00e+00**) · optimizer 13.456 KB → 7.445 KB |

### B. Phục vụ ở quy mô

| Kỹ thuật | Chứng minh | Kết quả đo |
| --- | --- | --- |
| **Paged KV cache** | so logits với đệm liền mạch của Cấp 3 | lệch **0,00e+00** · cùng chỗ bộ nhớ phục vụ **12,8 lần** số câu |
| **Continuous batching** | 6 câu hỏi, gom lô so với chạy lần lượt | logits lệch **5,3e-06** · nhanh hơn **1,9 lần** |
| **Quantization (int8)** | nén model, so chữ mà model chọn | nhỏ đi **3,5 lần** · chữ chọn giống **100%** |

---

## Chạy thử

```bash
pip install -r ../requirements.txt      # chỉ cần PyTorch

cd deepseek_prod

python -m deepseek_prod info            # model có gì
python -m deepseek_prod check           # chạy hết các phép kiểm chứng
python -m deepseek_prod check tp pp     # chỉ chạy vài phần
python -m deepseek_prod check --world 3 # thử với 3 máy
```

**Cần model của Cấp 3 trước.** Nếu chưa có:

```bash
cd ../deepseek_lite
python -m deepseek_lite train
```

Cấp 4 tự tìm thư mục `deepseek_lite/` bên cạnh, không cần cài đặt gì thêm.

---

## Cấu trúc dự án

```text
deepseek_prod/
├── pyproject.toml
├── README.md               file này
├── docs/
│   ├── song-song.md        bốn cách chia model, và các lỗi đã sập
│   └── han-che.md          những gì KHÔNG chạy được trên máy này, và vì sao
│
├── deepseek_prod/
│   ├── __init__.py
│   ├── __main__.py         CLI: info / check / bench
│   ├── paths.py            đường dẫn, kể cả sang Cấp 3
│   ├── base_model.py       lấy model của Cấp 3 ra dùng
│   ├── config.py           cấu hình chia model (tp/pp/dp/ep/zero)
│   │
│   ├── launcher.py         chạy nhiều tiến trình (Windows cần cách riêng)
│   ├── parallel.py         tôi là ai trong lưới, và nhóm nào
│   ├── tensor_parallel.py  cắt NGANG: ColumnParallel / RowParallel
│   ├── pipeline.py         cắt DỌC: chia tầng, chia vi lô, gửi gradient
│   ├── expert_parallel.py  mỗi máy vài chuyên gia
│   ├── zero.py             chia trạng thái optimizer
│   │
│   ├── paged_cache.py      đệm K/V chia khối
│   ├── engine.py           máy phục vụ: continuous batching
│   ├── quantize.py         nén 8 bit
│   │
│   └── checks.py           chứng minh từng tính năng bằng số
│
├── data/data.txt           bài học (bản sao của Cấp 3, để chạy độc lập)
└── runs/                   đầu ra
```

---

## Bốn cách chia model

Chi tiết đầy đủ ở [`docs/song-song.md`](docs/song-song.md). Tóm tắt:

```text
tp  cắt NGANG   một phép nhân bị chẻ ra nhiều máy
pp  cắt DỌC     mỗi máy giữ vài tầng
dp  dữ liệu     mỗi máy giữ cả model, chia nhau dữ liệu
ep  chuyên gia  mỗi máy giữ vài chuyên gia

world_size = tp × pp × dp        (ep dùng chung nhóm tp)
```

Ví dụ chạy trong code:

```python
from deepseek_prod import Parallel, ParallelConfig
from deepseek_prod.tensor_parallel import tensor_parallelize

parallel = Parallel(ParallelConfig(tp=2, pp=1, dp=1))
model = tensor_parallelize(model, parallel)
```

---

## Vì sao phải kiểm tra bằng SỐ

Ba trong bốn lỗi gặp khi làm phần song song đều cho ra **loss trông rất hợp
lý**. Nếu chỉ nhìn loss thì không phát hiện được:

| Lỗi | Loss | Gradient |
| --- | --- | --- |
| `recv` trần làm đứt đồ thị autograd | treo | — |
| thiếu `requires_grad` ở tensor giữ chỗ | treo | — |
| weight tying đứt giữa hai máy | **đúng** | sai 37% |
| EP trả giá trị khác nhau giữa các máy | vẫn hội tụ | sai dần |

Nên `checks.py` so **cả loss lẫn gradient**, và so trên **từng tham số**.

---

## Một chỗ thú vị: gom lô làm đổi kết quả

Chạy 6 câu hỏi theo kiểu gom lô và theo kiểu lần lượt, **chữ sinh ra có thể
khác nhau một ký tự**. Đó không phải lỗi:

Gom lô thì đệm K/V phải lót cho các câu dài bằng nhau, nên kích thước tensor
khác đi, và **thứ tự cộng số thực** trong phép attention cũng khác. Ở chỗ hai
chữ gần bằng điểm nhau thì chỉ cần lệch 1e-6 là kết quả đã lật.

Nên bài kiểm chứng so **logits** (lệch 5,3e-06) chứ không so ký tự. Đây là
tính chất đã biết của suy luận theo lô, hệ thật cũng vậy.

---

## Chưa làm gì

Roadmap của Cấp 4 có 11 mục. Ở đây làm được 7. Bốn mục còn lại **không chạy
được trên máy CPU này**, hoặc cần thời gian huấn luyện thật:

| Chưa làm | Vì sao |
| --- | --- |
| **FlashAttention** | cần GPU NVIDIA. PyTorch có chọn được backend này qua `sdpa_kernel`, nhưng CPU chỉ có bản `math`. |
| **FP8** | kiểu dữ liệu `float8_e4m3fn` có trong PyTorch, nhưng **tính toán** FP8 cần GPU Hopper (H100) trở lên. |
| **Speculative decoding** | làm được trên CPU, nhưng model nhỏ ở đây không có model nháp nào nhỏ hơn để làm bản nháp. |
| **1M context** | cần huấn luyện lại ở ngữ cảnh dài. Model này học ở 512 token. |
| **MLA, MTP** | hai kỹ thuật riêng của DeepSeek-V3, mỗi cái đáng một dự án. |

Xem [`docs/han-che.md`](docs/han-che.md) để biết chi tiết, kèm cách tự kiểm
chứng là máy này thật sự không làm được.

---

## Ba cấp trước

| Cấp | Thư mục | Dạy gì |
| --- | --- | --- |
| 1 | `tiny_gpt/` | Attention là gì — một file, 302 dòng |
| 2 | `mini_deepseek/` | MoE là gì — 12 file |
| 3 | `deepseek_lite/` | Vì sao LLM chạy được thật — package + 41 test |
| 4 | `deepseek_prod/` | Làm sao chạy ở quy mô lớn — **thư mục này** |
