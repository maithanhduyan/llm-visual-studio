# deepseek_lite

> Một LLM nhỏ kiểu DeepSeek, viết để học. **Cấp 3** trong lộ trình.

Cấp 1 (`tiny_gpt/`) dạy Attention là gì. Cấp 2 (`mini_deepseek/`) dạy MoE là gì.
Cấp 3 dạy những thứ khiến một LLM **chạy được thật**: đệm K/V, GQA, cửa sổ
trượt, chuyên gia dùng chung, và chia việc không cần hàm phạt.

Đây là một **package Python thật**: cài được, import được, có test, có CLI.
Nhưng vẫn giữ chú thích tiếng Việt để đọc hiểu từng dòng.

---

## Có gì mới so với Cấp 2

| | Cấp 2 — `mini_deepseek/` | Cấp 3 — `deepseek_lite/` |
| --- | --- | --- |
| Attention | MHA: 6 đầu Q = 6 đầu K/V | **GQA**: 6 đầu Q, 2 đầu K/V → đệm nhỏ 3 lần |
| Đệm K/V | không có, sinh chữ phải tính lại cả câu | **có**, mỗi chữ chỉ tính 1 token |
| Câu dài | nhìn hết cả câu | **cửa sổ trượt**, cắt hẳn K/V cũ |
| Chuyên gia | 4 chuyên gia to | **8 chuyên gia nhỏ + 1 dùng chung** |
| Chia việc | hàm phạt cộng vào loss | **điểm thiên vị**, không đụng loss |
| Residual | 1 dòng | **nhiều dòng** (hyper-connections) |
| Vị trí | RoPE tính lại mỗi lần | RoPE **tính sẵn**, có `start_pos` |
| Học | vòng lặp đơn giản | **lịch học**, **chia bài học/thi**, gom lô |
| Đóng gói | 12 file rời | **package** + test + CLI + pyproject |

---

## Chạy thử

Không cần cài gì cả — chỉ cần đứng ở thư mục dự án:

```bash
pip install -r ../requirements.txt      # chỉ cần PyTorch

cd deepseek_lite

python -m deepseek_lite info            # xem model có gì (chưa cần học)
python -m deepseek_lite train           # dạy nó học
python -m deepseek_lite generate "Học mãi thì"
python -m deepseek_lite benchmark       # đo xem đệm K/V nhanh hơn bao nhiêu
```

Muốn học nhanh cho có kết quả: `python -m deepseek_lite train --steps 400`.
Muốn đổi bài học: sửa `data/data.txt`.

Chú ý: **phải dùng `python -m`**, đừng chạy `python deepseek_lite/train.py`.
Vì đây là package, các file import lẫn nhau bằng đường dẫn tương đối.

### Vài tham số hay dùng

```bash
python -m deepseek_lite train --steps 2000 --n-layers 4     # model sâu hơn
python -m deepseek_lite train --window 0                    # tắt cửa sổ trượt
python -m deepseek_lite train --n-streams 1                 # về đường tắt thường
python -m deepseek_lite train --n-experts 16                # nhiều chuyên gia hơn
python -m deepseek_lite generate "Chuyên gia" --temperature 0.5 --tokens 300
python -m deepseek_lite benchmark --tokens 200 --long-prompt 400
```

---

## Kết quả chạy thật

### Học

`python -m deepseek_lite train` — 1000 bước, 228 giây trên CPU:

```text
  bước     loss        lr  loss thi  chia đều
     0    4.983  3.00e-05     4.927     3.91x
   200    0.286  2.92e-03     2.950     1.40x
   600    0.100  1.42e-03     3.672     1.18x
  1000    0.051  3.00e-04     4.151     1.16x
```

Hai điều đáng chú ý ở đây.

**Cột `chia đều` đi từ 3,91x xuống 1,16x.** Đó là cơ chế chia việc không cần
hàm phạt đang chạy: lúc đầu router dồn việc cho vài chuyên gia, cuối thì chia
gần đều (1,00x là hoàn hảo). Không có hàm phạt nào được cộng vào loss.

**Loss phần học 0,055 nhưng loss phần thi 4,10 — model học vẹt hoàn toàn.**

Đó không phải lỗi, mà là số học. Bài học có 4.264 ký tự còn model có 1,83 triệu
thông số, tức gần **430 thông số cho mỗi ký tự**. Model thừa sức thuộc lòng
từng chữ, và nó chọn cách đó thay vì học quy luật.

Để so sánh: đoán bừa thì loss là `ln(145) = 4,98`. Phần thi được 4,10 nghĩa là
model có học được chút gì đó về tần suất ký tự — nhưng chỉ có thế.

Điều thú vị: trong `data.txt` có hẳn một mục tên **"BÀI 11: HỌC VẸT VÀ HỌC THẬT"**
giải thích hiện tượng này. Model đã thuộc lòng lời giải thích đó trong khi đang
làm đúng việc đó.

Muốn nó học thật thì cách duy nhất là **cho đọc nhiều hơn** — vài trăm nghìn ký
tự trở lên. Kiến trúc 3 tầng này còn thừa sức; thiếu là thiếu dữ liệu.

### Đo tốc độ

`python -m deepseek_lite benchmark`:

| Phép đo | Kết quả |
| --- | --- |
| Đệm K/V có đúng không | logits lệch 5,7e-06, chữ sinh ra **giống hệt** |
| Nhanh hơn bao nhiêu | 51 → 130 token/giây (**2,5 lần**) |
| GQA tiết kiệm bộ nhớ | 2304 KB → 768 KB (**3,0 lần**) |
| Cửa sổ trượt, batch 1 | 0,87x — **chậm hơn** |
| Cửa sổ trượt, batch 16 | 2,16x |
| Cửa sổ trượt, batch 64 | **4,30x** |

Dòng "batch 1 chậm hơn" là kết quả thật, và đáng giữ lại thay vì giấu đi.
Ở batch 1, mỗi phép tính nhỏ đến mức chi phí gọi hàm lấn át phần tính toán,
nên cắt bớt K/V còn tốn hơn khoản tiết kiệm được. Cửa sổ trượt chỉ có lợi khi
phục vụ nhiều câu cùng lúc — đúng như hệ thống thật vẫn làm.

---

## Dùng như thư viện

```python
from deepseek_lite import Config, DeepSeekLite, Tokenizer, KVCache

model = DeepSeekLite(Config(vocab_size=145))
print(model)
#   thông số      : 1,833,624 (mỗi token chỉ dùng 831,384)
#   tầng          : 3
#   đầu Q / K-V   : 6 / 2  (GQA)
#   cửa sổ trượt  : 64
#   chuyên gia    : 8 riêng + 1 dùng chung, chọn 2
#   dòng suy nghĩ : 4

total, active = model.parameter_counts()
logits, loss = model(tokens, targets)
```

Sinh chữ có đệm K/V:

```python
from deepseek_lite.generate import load_checkpoint, generate

model, tokenizer = load_checkpoint()
print(generate(model, tokenizer, "Học mãi thì", max_new_tokens=200))
```

---

## Cấu trúc dự án

```text
deepseek_lite/
├── pyproject.toml          metadata, dependency, cấu hình pytest/ruff
├── README.md               file này
├── docs/kien-truc.md       giải thích 6 ý tưởng mới của Cấp 3
├── data/data.txt           bài học
├── runs/                   model sau khi học xong (không commit)
│
├── deepseek_lite/          package
│   ├── __init__.py         API công khai
│   ├── __main__.py         CLI: info / train / generate / benchmark
│   ├── paths.py            mọi đường dẫn nằm ở một chỗ
│   ├── config.py           các con số + kiểm tra hợp lệ
│   ├── tokenizer.py        chữ -> số
│   ├── data.py             đọc bài, chia bài học/thi, bốc lô
│   ├── rope.py             vị trí, tính sẵn cos/sin
│   ├── cache.py            đệm K/V
│   ├── attention.py        GQA + cửa sổ trượt + đệm
│   ├── mlp.py              SwiGLU
│   ├── moe.py              chuyên gia dùng chung + chia việc
│   ├── mhc.py              nhiều dòng suy nghĩ
│   ├── block.py            một tầng
│   ├── model.py            ghép tất cả lại
│   ├── train.py            vòng lặp học
│   ├── generate.py         sinh chữ
│   └── benchmark.py        đo tốc độ
│
└── tests/                  41 bài test
    ├── conftest.py
    ├── test_tokenizer.py
    ├── test_rope.py
    ├── test_cache.py
    ├── test_attention.py
    ├── test_moe.py
    └── test_model.py
```

---

## Test

```bash
python -m pytest
```

41 bài test, chạy trong nửa giây. Bài quan trọng nhất là
`test_cache_gives_identical_results`: nó chạy sinh từng chữ **có đệm** và
**không đệm**, rồi khẳng định hai kết quả giống hệt nhau. Nếu bài đó fail thì
đệm K/V đang tính sai, và mọi con số tốc độ đều vô nghĩa.

Vài bài đáng chú ý:

| Bài test | Kiểm tra điều gì |
| --- | --- |
| `test_is_causal` | không token nào nhìn được vào tương lai |
| `test_sliding_window_hides_the_past` | cửa sổ trượt che đúng phần cần che |
| `test_cache_gives_identical_results` | có đệm và không đệm cho ra y hệt |
| `test_bias_punishes_overused_and_rewards_underused` | thuật toán chia việc đúng chiều |
| `test_shared_expert_always_contributes` | tắt hết chuyên gia riêng vẫn có đầu ra |
| `test_single_stream_matches_plain_residual` | `n_streams=1` quay về đường tắt thường |
| `test_gradients_reach_every_parameter` | không thông số nào đứng ngoài đồ thị |

---

## Những chỗ cố tình làm đơn giản

1. **MoE chạy bằng vòng lặp Python.** Bản thật nhóm token theo chuyên gia rồi
   chạy song song trên GPU. Cùng ý tưởng, khác tốc độ. Đây là chỗ chậm nhất
   của dự án — đổi sang cách nhóm sẽ nhanh hơn đáng kể, nhưng code khó đọc hơn.
2. **Cửa sổ trượt chỉ có lợi khi batch lớn.** Lúc HỌC, mặt nạ chỉ che chứ
   không bỏ được tính toán. Lúc SINH CHỮ thì cắt hẳn được, nhưng ở batch 1
   khoản tiết kiệm còn nhỏ hơn chi phí cắt — đo ra thì **chậm hơn 0,87x**.
   Từ batch 16 trở lên mới nhanh hơn (2,16x), batch 64 thì 4,30x. Bản thật
   dùng kernel block-sparse để nhanh cả lúc học.
3. **Tokenizer theo ký tự**, không phải BPE. Vocab 145 chữ thay vì 129.000.
4. **Đệm K/V cấp phát sẵn `max_seq_len`** và không cuộn vòng, nên sinh quá
   `max_seq_len` token là dừng.
5. **Chỉ có một lớp mHC đơn giản.** Bản thật có ràng buộc hình học giữa các dòng.
6. **Không có Multi-Token Prediction (MTP)** — DeepSeek-V3 có, nhưng nó là
   một đầu phụ nữa, để Cấp 4.

---

## Lên Cấp 4 thì thêm gì

```text
Tensor / Pipeline / Expert parallelism    chia model ra nhiều GPU
FlashAttention                            attention nhanh hơn và tốn ít nhớ hơn
FP8 / quantization                        số 8 bit thay vì 32 bit
Multi-Token Prediction                    đoán nhiều chữ một lúc
MLA (Multi-head Latent Attention)         nén K/V thành vector latent
Checkpoint sharding                       lưu model to hơn RAM
1M context                                cửa sổ trượt cuộn vòng + ngữ cảnh thưa
```

Lúc đó dự án không còn là "cho bé lớp 5 đọc" được nữa. Cấp 3 là chỗ dừng hợp lý.
