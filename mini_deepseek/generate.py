"""
generate.py — Làm sao AI nói?

Sau khi học xong, model đoán từng chữ MỘT.
Đoán xong chữ này, nó ghép chữ đó vào câu, rồi đoán chữ tiếp theo.
Cứ như vậy cho đến khi hết.

    "Học mãi thì"  -> đoán " "
    "Học mãi thì " -> đoán "g"
    "Học mãi thì g" -> đoán "i"
    ...

Cách chạy (nhớ chạy train.py trước):
    python generate.py
    python generate.py "Học mãi thì"
    python generate.py "Học mãi thì" 1.3     # đổi độ sáng tạo
"""

import os
import sys

import torch

from config import Config
from model import DeepSeekMini
from tokenizer import Tokenizer


@torch.no_grad()  # không cần học nữa, chỉ cần đoán
def generate(model, tokens, max_new_tokens=200, temperature=0.8):
    """
    tokens: [1, T] — câu mở đầu, đã đổi thành số
    Trả về: [1, T + max_new_tokens] — câu đã viết thêm

    temperature nhỏ (0.2) -> nói chắc chắn, hay lặp lại
    temperature to (1.5)  -> nói sáng tạo, hay linh tinh
    """
    for _ in range(max_new_tokens):
        # 1. Model nhìn cả câu và cho điểm cho MỌI chữ có thể đứng tiếp.
        logits = model(tokens)[:, -1, :]

        # 2. Chia cho temperature rồi đổi thành phần trăm.
        probs = torch.softmax(logits / temperature, dim=-1)

        # 3. Bốc thăm một chữ theo đúng phần trăm đó.
        next_token = torch.multinomial(probs, num_samples=1)

        # 4. Ghép chữ vừa đoán vào câu, rồi lặp lại.
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if not os.path.exists("mini_deepseek.pt"):
        raise SystemExit("Chưa có model. Hãy chạy  python train.py  trước nhé!")

    # Đọc lại đúng model đã học (cùng cấu hình, cùng từ điển).
    checkpoint = torch.load("mini_deepseek.pt", map_location="cpu")

    tokenizer = Tokenizer.load(checkpoint["chars"])
    config = Config(**checkpoint["config"])
    model = DeepSeekMini(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Học mãi thì"
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    result = generate(model, tokens, temperature=temperature).tolist()[0]

    print(tokenizer.decode(result))
