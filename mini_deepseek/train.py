"""
train.py — Làm sao AI học?

    Đưa cho nó một câu:    "Hôm nay trời rất"
    Nó đoán chữ tiếp theo: "đẹp"
    Đoán sai  -> bị phạt   (loss cao)
    Đoán đúng -> được thưởng (loss thấp)
    Lặp lại hàng nghìn lần -> nó giỏi dần

Cách chạy:
    python train.py          # 1000 bước (chạy vài phút)
    python train.py 200      # 200 bước, để thử cho nhanh

Lúc đầu loss khoảng 4.5 (đoán bừa trong ~90 chữ).
Khi loss xuống dưới 0.5 là nó đã thuộc bài.
"""

import sys

import torch
import torch.nn.functional as F

from config import Config
from model import DeepSeekMini
from tokenizer import Tokenizer

# Để in được tiếng Việt trên Windows.
sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------
# 1. Đọc sách
# ---------------------------------------------------------------------
with open("data.txt", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(text)
tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

if len(tokens) < 64:
    raise SystemExit("data.txt quá ngắn. Hãy viết thêm vài câu nhé!")

print(f"Sách có {len(text)} ký tự, gồm {tokenizer.vocab_size} ký tự khác nhau.")


# ---------------------------------------------------------------------
# 2. Dựng model
# ---------------------------------------------------------------------
config = Config(vocab_size=tokenizer.vocab_size)
model = DeepSeekMini(config)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model có {n_params:,} thông số.")


# ---------------------------------------------------------------------
# 3. Cách lấy bài để học
# ---------------------------------------------------------------------
BLOCK = min(config.max_seq_len, len(tokens) - 2)


def get_batch(batch_size=8):
    """Bốc ngẫu nhiên vài đoạn trong sách.

    x = đoạn văn
    y = cũng đoạn đó nhưng dịch lên 1 chữ
        -> đó chính là "đáp án" mà model phải đoán.
    """
    starts = torch.randint(0, len(tokens) - BLOCK - 1, (batch_size,))
    x = torch.stack([tokens[s : s + BLOCK] for s in starts])
    y = torch.stack([tokens[s + 1 : s + BLOCK + 1] for s in starts])
    return x, y


# ---------------------------------------------------------------------
# 4. Vòng lặp học
# ---------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

model.train()

# Ghi lại loss sau mỗi bước, để studio/ vẽ biểu đồ loss theo thời gian.
history = []

for step in range(steps + 1):
    x, y = get_batch()

    logits = model(x)  # model đoán: [B, T, vocab_size]

    # So đáp án với điều model đoán. Đây là "điểm phạt".
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    # Cộng thêm điểm phạt nếu model dồn hết việc cho một chuyên gia
    # (xem phần cuối file moe.py).
    total = loss + 0.01 * model.aux_loss()

    optimizer.zero_grad()   # xoá lỗi cũ
    total.backward()        # tính xem nên sửa thông số nào
    optimizer.step()        # sửa thông số

    if step % 100 == 0:
        print(f"bước {step:5d}   loss = {loss.item():.3f}")

    history.append([step, round(loss.item(), 4), round(model.aux_loss().item(), 4)])


# ---------------------------------------------------------------------
# 5. Lưu lại để generate.py dùng
# ---------------------------------------------------------------------
torch.save(
    {
        "config": config.__dict__,
        "chars": tokenizer.chars,
        "model": model.state_dict(),
        "history": history,
    },
    "mini_deepseek.pt",
)
print("Đã lưu model vào mini_deepseek.pt")
