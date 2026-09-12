# Mini DeepSeek — một LLM mà bé lớp 5 cũng đọc hiểu

> ⚠️ Đây là bản **DeepSeek-V4.1-Flash-inspired Mini**: giữ lại *ý tưởng* kiến trúc
> trong hình, nén thành ~12 file Python nhỏ để học và chạy thật trên CPU.
> **Không phải** bản implementation 552B chính thức.

Model thật của DeepSeek: **552 tỷ thông số**. Kiến trúc thì giống nhau về ý tưởng.

Repo có bốn cấp, đi từ dễ đến khó:

| | `tiny_gpt/` — Cấp 1 | `mini_deepseek/` — Cấp 2 | `deepseek_lite/` — Cấp 3 | `deepseek_prod/` — Cấp 4 |
| --- | --- | --- | --- | --- |
| Thông số | 349 nghìn | 3,4 triệu | 1,8 triệu | *dùng lại Cấp 3* |
| Số file | **1 file duy nhất** | 12 file | **package + test + CLI** | **package + 7 phép kiểm chứng** |
| Có gì | Attention + MLP | thêm RoPE, SwiGLU, **MoE**, mHC | thêm **GQA**, **cửa sổ trượt**, **KV cache**, chuyên gia dùng chung | thêm **chia model ra nhiều máy**, **phục vụ nhiều người** |
| Sinh chữ | tính lại cả câu | tính lại cả câu | **đệm K/V** | **gom lô, đệm chia khối** |
| Học trong | ~35 giây | ~5 phút | ~4 phút | không huấn luyện — dùng model Cấp 3 |
| Dành cho | đọc một mạch là hiểu | hiểu kiến trúc DeepSeek thật | hiểu vì sao LLM chạy được thật | hiểu vì sao nó chạy được ở **quy mô lớn** |

Model thật của DeepSeek: **552 tỷ thông số**. Kiến trúc thì giống nhau về ý tưởng.

---

## Chạy thử

```bash
pip install -r requirements.txt

# Cách nhanh nhất — một file, một lệnh, ~35 giây:
cd tiny_gpt
python tiny_gpt.py

# Đầy đủ hơn — thêm RoPE, SwiGLU, MoE (~5 phút):
cd ../mini_deepseek
python main.py           # 1. Xem model có gì (chưa cần học)
python train.py          # 2. Dạy nó học
python generate.py       # 3. Cho nó nói

# Gần với LLM thật — thêm GQA, cửa sổ trượt, KV cache (~4 phút):
cd ../deepseek_lite
python -m deepseek_lite info       # xem model có gì
python -m deepseek_lite train      # dạy nó học
python -m deepseek_lite benchmark  # đo xem đệm K/V nhanh hơn bao nhiêu

# Quy mô lớn — chia model ra nhiều máy, phục vụ nhiều người (dùng lại model Cấp 3):
cd ../deepseek_prod
python -m deepseek_prod info       # xem model và cách chia
python -m deepseek_prod check      # 7 phép kiểm chứng, mỗi phép có số đo
```

Muốn thử nhanh: `python train.py 200` (200 bước, ~1 phút).
Muốn đổi bài học: sửa `mini_deepseek/data.txt`, rồi chạy lại `train.py`.

---

## Cấp 1 — TinyGPT: cả một LLM trong MỘT file

`tiny_gpt/tiny_gpt.py` là điểm bắt đầu. Nó **cố tình thiếu** mọi thứ hoa mỹ:

| Thiếu gì | Vì sao |
| --- | --- |
| MoE | chỉ một MLP chung cho mọi token — không cần chia chuyên gia |
| RoPE | dùng bảng vị trí học được (`nn.Embedding`) — dễ hiểu hơn |
| RMSNorm | dùng `nn.LayerNorm` có sẵn của PyTorch |
| SwiGLU | dùng MLP thường: `Linear → GELU → Linear` |
| mHC | chỉ một dòng residual |

Đổi lại, file có đúng **7 mục**, chính là 7 câu hỏi trong bảng dưới đây — đọc một
mạch từ trên xuống là hiểu hết, không phải mở file nào khác:

```text
1. CHỮ BIẾN THÀNH SỐ THẾ NÀO?         chars, stoi, itos, encode, decode
2. CÁC CON SỐ                          block_size, n_embd, n_head, n_layer
3. AI NHÌN VÀO TỪ NÀO?                 class Attention      (Q, K, V + mặt nạ tam giác)
4. AI SUY NGHĨ BÊN TRONG THẾ NÀO?      class MLP
5. MỘT TẦNG GỒM NHỮNG GÌ?              class Block          (2 việc + 2 đường tắt)
6. GHÉP TẤT CẢ LẠI                     class TinyGPT
7. LÀM SAO AI HỌC / LÀM SAO AI NÓI?    get_batch, generate, vòng lặp học
```

Điểm hay nhất khi chạy: nó in ra **cả trước lẫn sau khi học**. Trước khi học, nó
nói linh tinh; sau 1000 bước, nó viết được tiếng Việt. Đó là toàn bộ phép thuật
của LLM, gói trong **302 dòng — chỉ 172 dòng là code thật**, còn lại là chú thích
cho bé đọc.

So sánh với Cấp 2: `mini_deepseek/` có 12 file, 859 dòng, 3,4 triệu thông số.
Cùng một ý tưởng, chỉ thêm RoPE, SwiGLU, MoE và mHC.

---

## Cấp 3 — `deepseek_lite/`: vì sao LLM chạy được thật

Cấp 2 dạy MoE là gì. Nhưng Cấp 2 sinh chữ bằng cách chạy lại **cả câu** mỗi lần —
viết 200 chữ thì chữ thứ 200 phải tính 200 lần. Không LLM thật nào làm vậy.

Cấp 3 thêm sáu thứ, tất cả đều là chiêu thật của DeepSeek-V3:

| Thêm gì | Giải quyết chuyện gì |
| --- | --- |
| **GQA** | 6 đầu Q dùng chung 2 đầu K/V → đệm nhỏ đi 3 lần |
| **KV cache** | viết chữ thứ 100 không phải tính lại 99 chữ trước |
| **Cửa sổ trượt** | câu dài chỉ nhìn lại đoạn gần nhất, và **cắt hẳn** K/V cũ |
| **Chuyên gia dùng chung** | một chuyên gia ai cũng hỏi, lo việc cơ bản |
| **Chia việc không hàm phạt** | không đụng vào loss, chỉ chỉnh "điểm thiên vị" |
| **Hyper-connections** | nhiều dòng suy nghĩ song song thay vì một |

Khác biệt lớn nhất về cách làm: đây là một **package Python thật** — có
`pyproject.toml`, 41 bài test, CLI riêng, và chạy bằng `python -m`.

```bash
cd deepseek_lite

python -m deepseek_lite info        # xem model có gì (chưa cần học)
python -m deepseek_lite train       # học 1000 bước, ~4 phút
python -m deepseek_lite benchmark   # đo đệm K/V nhanh hơn bao nhiêu
python -m pytest                    # 41 bài test, chạy trong nửa giây
```

Số đo thật: đệm K/V nhanh hơn **2,5 lần**, GQA tiết kiệm **3,0 lần** bộ nhớ, và
router tự chia đều việc cho 8 chuyên gia (3,91x → 1,16x) **mà không cần thêm
hàm phạt nào vào loss**.

Cấp 3 cũng trung thực về giới hạn của mình: với 4.264 ký tự bài học và 1,83
triệu thông số, model **học vẹt hoàn toàn** (loss học 0,055 / loss thi 4,10).
Đó là bài học về dữ liệu, không phải lỗi kiến trúc — và `train.py` tự cảnh báo
điều đó.

Đọc chi tiết: [`deepseek_lite/README.md`](deepseek_lite/README.md) ·
[`deepseek_lite/docs/kien-truc.md`](deepseek_lite/docs/kien-truc.md)

---

## Xem tận mắt: Studio — cả ba cấp

Học xong rồi nhưng `loss = 0.03` vẫn chỉ là một con số vô hình. Studio mở nắp hộp
cả ba model ra cho xem bên trong, và đổi qua lại giữa chúng để so sánh:

| Phần | Cho thấy gì |
| --- | --- |
| **1. Kiến trúc model** | Sơ đồ 3D toàn bộ model. **Ngang** = dòng chảy, **sâu** = tầng thứ mấy, **đứng** = nhánh (đường tắt vòng lên trên, chuyên gia toả xuống dưới). **Bấm vào một bộ phận** để xem nó nhận từ đâu, gửi tới đâu, và làm gì. |
| **2. Không gian embedding 3D** | Mỗi ký tự là một điểm, chiếu từ 96–192 chiều xuống 3 chiều bằng PCA. Kéo chuột để xoay. Tô màu theo chuyên gia → thấy chuyên gia có chuyên môn hoá theo vùng nghĩa không. |
| **3. Attention map** | Token nào đang nhìn vào token nào. Chọn được từng tầng và từng đầu (head). |
| **4. Chuyên gia** | Mỗi token chọn 2 trong 4 (hoặc 8) chuyên gia, và tin mỗi chuyên gia bao nhiêu. Ẩn đi ở Cấp 1 vì Cấp 1 chưa có chuyên gia. |
| **5. Loss** | Loss tụt thế nào qua từng bước. Bật được thang log và đường thứ hai (aux loss ở Cấp 2, độ lệch chia việc ở Cấp 3). |

```bash
cd studio
bun install              # chỉ để lấy three.js cho phần 3D
bun run server.ts        # rồi mở http://localhost:4173
```

Gõ bất kỳ câu nào vào ô trên cùng rồi bấm **Phân tích** — model chạy thật và trả về
attention thật, không phải hình minh hoạ.

Studio cần ba thứ:

- **Bun** (>= 1.2) để chạy máy chủ và đóng gói TypeScript.
- **three.js** cho phần 3D — đây là thư viện duy nhất, cài bằng `bun install`.
- **Python + PyTorch** như ở trên. `server.ts` giữ sẵn một tiến trình Python
  (`analyze.py --serve`), nên mỗi câu chỉ mất vài chục mili-giây thay vì vài giây
  nạp lại model.

```text
studio/
├── server.ts     # Bun: trả trang web + giữ tiến trình Python
├── analyze.py    # Python: nạp cả 3 model, trả về attention + chuyên gia + PCA + sơ đồ
├── types.ts      # kiểu dữ liệu dùng chung cho cả hai bên
├── index.html
└── src/
    ├── main.ts       # nối năm phần lại, và bộ chọn model
    ├── arch3d.ts     # sơ đồ kiến trúc 3D (three.js)
    ├── space3d.ts    # không gian embedding 3D (three.js)
    ├── attention.ts  # vẽ bản đồ attention
    ├── experts.ts    # vẽ đường đi của chuyên gia
    ├── loss.ts       # vẽ biểu đồ loss
    ├── format.ts
    └── style.css
```

Học lại xong thì chỉ cần tải lại trang — `analyze.py` tự nhận ra file model vừa
thay đổi và nạp lại. Sửa `analyze.py` cũng vậy, không phải khởi động lại máy chủ.

---

## Mỗi file trả lời đúng MỘT câu hỏi (Cấp 2 — `mini_deepseek/`)

> Ở Cấp 1 (`tiny_gpt/tiny_gpt.py`), bảy câu hỏi này nằm trong bảy MỤC của cùng
> một file. Cùng câu hỏi, chỉ khác cách chia.

| File | Câu hỏi |
| --- | --- |
| `config.py` | Các con số của model là gì? |
| `tokenizer.py` | Chữ biến thành số thế nào? |
| `rope.py` | AI biết token nào đứng trước, token nào đứng sau nhờ đâu? |
| `attention.py` | AI nhìn vào từ nào? |
| `mlp.py` | AI suy nghĩ bên trong thế nào? (SwiGLU) |
| `moe.py` | AI chọn chuyên gia nào? (Router + Experts) |
| `mhc.py` | Nếu AI có 4 "dòng suy nghĩ" song song thì sao? |
| `block.py` | Một tầng AI gồm những gì? |
| `model.py` | Ghép tất cả lại thế nào? |
| `train.py` | Làm sao AI học? |
| `generate.py` | Làm sao AI nói? |
| `main.py` | Chạy thử tất cả. |

---

## Một token đi qua model như thế nào?

```text
                  "Học mãi thì"
                        |
                  [tokenizer.py]          chữ -> số
                        |
                   [ 41, 62, 88, ... ]
                        |
                   Embedding              số -> vector
                        |
        +---------------v----------------+
        |        Block  x  n_layers      |
        |                                |
        |   RMSNorm                      |
        |      v                         |
        |   RoPE + Attention   <-- các token trao đổi với nhau
        |      v                         |
        |   x = x + ...        <-- đường tắt (residual)
        |      v                         |
        |   RMSNorm                      |
        |      v                         |
        |   MoE: Router -> 2 / 4 chuyên gia
        |      v                         |
        |   x = x + ...        <-- đường tắt (residual)
        +---------------v----------------+
                        |
                    RMSNorm
                        |
                     LM Head               vector -> điểm cho MỌI chữ
                        |
                  chữ tiếp theo: " "
```

---

## Bản mini giữ lại gì từ kiến trúc lớn?

| Trong hình (bản lớn) | Trong bản mini |
| --- | --- |
| Vocabulary 129K (BPE) | `Tokenizer` theo từng ký tự (~88 chữ) |
| Embedding 5.120 | `d_model = 128` |
| Attention sparse / CSA | `Attention` dense + `is_causal=True` |
| RoPE | `rope.py` (đúng công thức, chỉ nhỏ hơn) |
| MoE + Router | `moe.py`, 4 chuyên gia, chọn `top_k = 2` |
| SwiGLU | `mlp.py` (đúng công thức) |
| mHC — 4 residual streams | `mhc.py` (bản phác thảo, xem ghi chú bên dưới) |
| Context 1M | `max_seq_len = 256` |
| KV cache | chưa có — `generate.py` tính lại cả câu mỗi bước |
| 552B thông số | 3.4M thông số |

---

## Những chỗ cố tình làm đơn giản

Đây là những chỗ bản mini **khác** bản thật, để đọc cho dễ:

1. **MoE chạy bằng vòng lặp Python** (`for expert in ...`). Bản thật nhóm token
   theo chuyên gia và chạy song song trên GPU. Cùng một ý tưởng, khác tốc độ.
2. **RoPE tính lại mỗi lần forward**, không cache `cos/sin`. Bản thật cache sẵn.
3. **Không có KV cache.** Sinh 200 token = 200 lần forward cả câu. Bản thật
   chỉ tính token mới.
4. **`mhc.py` là bản phác thảo**, không phải mHC chính thức. Công thức thật có
   ràng buộc hình học giữa các dòng.
5. **Tokenizer theo ký tự.** Bản thật dùng BPE. Muốn thử BPE thì thay `tokenizer.py`.
6. **Có thêm `aux_loss`** (không có trong bản nháp đầu tiên): nếu không phạt,
   router sẽ dồn hết token cho 1–2 chuyên gia và các chuyên gia còn lại
   không bao giờ được học. `aux_loss = 2.0` là chia đều hoàn hảo.

---

## Bốn cấp độ để lớn dần

| Cấp | Thêm gì | Trạng thái |
| --- | --- | --- |
| **1. TinyGPT** | Embedding → Attention → MLP → LM Head | **xong** — `tiny_gpt/tiny_gpt.py`, 349k thông số, 1 file |
| **2. Mini DeepSeek** | RoPE, RMSNorm, SwiGLU, MoE, mHC | **xong** — `mini_deepseek/`, 3,4M thông số, 12 file |
| **3. DeepSeek-inspired** | GQA, cửa sổ trượt, KV cache, chuyên gia dùng chung, chia việc không hàm phạt, hyper-connections | **xong** — `deepseek_lite/`, 1,8M thông số, package + 41 test |
| **4. Production LLM** | Tensor/pipeline/expert parallelism, ZeRO, paged KV cache, continuous batching, quantization | **xong** — `deepseek_prod/`, 7 kỹ thuật có số đo (xem [`deepseek_prod/README.md`](deepseek_prod/README.md)) |

Bốn mục của Cấp 4 **không chạy được trên máy CPU này** và đã ghi rõ lý do:
FlashAttention và FP8 cần GPU NVIDIA, 1M context cần huấn luyện lại ở ngữ
cảnh dài, và speculative decoding cần một model nháp dùng chung từ vựng.
Xem [`deepseek_prod/docs/han-che.md`](deepseek_prod/docs/han-che.md).

---

## Nguyên tắc

```text
KHÔNG làm:                          LÀM:
deepseek/                           mini_deepseek/
├── 200 files                       ├── tokenizer.py
├── 50 abstractions                 ├── attention.py
├── 20 factories                    ├── moe.py
├── 10 registries                   ├── mlp.py
└── 5000 dòng framework             ├── block.py
                                    ├── model.py
                                    ├── train.py
                                    └── generate.py
```

**Mỗi file = một ý tưởng. Mỗi file trả lời một câu hỏi.**
