"""
main.py — Chạy thử xem mọi thứ có hoạt động không.

Cách chạy:
    python main.py
"""

import os
import sys

import torch

from config import Config
from generate import generate
from model import DeepSeekMini
from tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 60)
print("MINI DEEPSEEK — một LLM nhỏ để học")
print("=" * 60)


# ---------------------------------------------------------------------
# 1. Dựng model và xem nó to cỡ nào
# ---------------------------------------------------------------------
config = Config()
model = DeepSeekMini(config)

n_params = sum(p.numel() for p in model.parameters())
print(f"\n1. Model có {n_params:,} thông số.")
print("   (đây là từ điển mặc định 256 chữ; học tiếng Việt thì ít hơn.)")
print(f"   {config.n_layers} tầng, mỗi tầng {config.n_experts} chuyên gia,")
print(f"   mỗi token chỉ hỏi {config.top_k} chuyên gia.")
print("   (Model thật của DeepSeek có hơn 500.000.000.000 thông số.)")


# ---------------------------------------------------------------------
# 2. Cho nó đọc thử vài token xem hình dạng có đúng không
# ---------------------------------------------------------------------
tokens = torch.randint(0, config.vocab_size, (2, 5))  # 2 câu, mỗi câu 5 token
logits = model(tokens)

print("\n2. Cho nó đọc thử:")
print(f"   đầu vào  {tuple(tokens.shape)}  (2 câu, 5 token)")
print(f"   đầu ra   {tuple(logits.shape)}  (2 câu, 5 token, {config.vocab_size} chữ)")
print("   Mỗi token có điểm cho MỌI chữ trong từ điển.")


# ---------------------------------------------------------------------
# 3. Chưa học thì nó nói gì?
# ---------------------------------------------------------------------
prompt = torch.randint(0, config.vocab_size, (1, 4))
out = generate(model, prompt, max_new_tokens=12, temperature=1.0)

print("\n3. Khi CHƯA học, nó chỉ nói linh tinh:")
print("  ", out[0].tolist())
print("   (toàn số, vì nó chưa biết chữ nào đi với chữ nào.)")


# ---------------------------------------------------------------------
# 4. Nếu đã học xong thì cho nó nói thử
# ---------------------------------------------------------------------
if os.path.exists("mini_deepseek.pt"):
    checkpoint = torch.load("mini_deepseek.pt", map_location="cpu")

    tokenizer = Tokenizer.load(checkpoint["chars"])
    trained = DeepSeekMini(Config(**checkpoint["config"]))
    trained.load_state_dict(checkpoint["model"])
    trained.eval()

    start = torch.tensor([tokenizer.encode("Học mãi thì")], dtype=torch.long)
    text = tokenizer.decode(generate(trained, start).tolist()[0])

    print("\n4. Sau khi học, nó nói được thế này:")
    print("  ", text.replace("\n", "\n   "))
else:
    print("\n4. Chưa có model học xong.")
    print("   Chạy  python train.py  rồi chạy lại file này nhé!")

print("\n" + "=" * 60)
