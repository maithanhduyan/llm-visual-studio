"""
tiny_gpt.py — LLM nhỏ nhất có thể, để hiểu LLM là gì.

Đây là CẤP 1 trong lộ trình. Cả file chỉ có một ý tưởng:

    Chữ -> số -> Embedding -> [Attention -> MLP] x N -> LM Head -> chữ tiếp theo

So với mini_deepseek/, file này CỐ TÌNH thiếu những thứ sau:

    - không có MoE          (chỉ một MLP chung cho mọi token)
    - không có RoPE         (dùng bảng vị trí học được)
    - không có RMSNorm      (dùng LayerNorm có sẵn của PyTorch)
    - không có SwiGLU       (dùng MLP thường)
    - không có mHC

Đổi lại, em đọc một mạch từ trên xuống dưới là hiểu hết, không phải mở
file nào khác. Bảy mục dưới đây chính là bảy câu hỏi trong README.

Chạy:
    python tiny_gpt.py          # học 1000 bước rồi tự nói thử
    python tiny_gpt.py 200      # học 200 bước cho nhanh
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows


# =====================================================================
# 1. CHỮ BIẾN THÀNH SỐ THẾ NÀO?
# =====================================================================
# Model KHÔNG nhìn thấy chữ. Nó chỉ nhìn thấy số.
# Muốn học bài khác thì sửa đoạn chữ trong dấu ba ngoặc kép này.

text = """Tôi là TinyGPT. Tôi là một mô hình ngôn ngữ rất nhỏ.
Tôi không hiểu tiếng Việt. Tôi chỉ đoán chữ tiếp theo.
Khi đọc câu "Hôm nay trời rất", tôi đoán chữ tiếp theo là "đẹp".
Đoán sai thì tôi tự sửa. Đoán đúng thì tôi nhớ.
Tôi có ba tầng. Mỗi tầng làm hai việc.
Việc thứ nhất là attention: tôi nhìn lại các chữ đứng trước.
Việc thứ hai là mlp: tôi tự suy nghĩ một mình.
Attention giúp tôi biết chữ nào đi với chữ nào.
Mlp giúp tôi nhớ những điều đã học.
Học mãi thì giỏi. Đoán mãi thì quen.
Tôi không thông minh. Tôi chỉ chăm chỉ.
Bé cũng làm được. Chỉ cần đọc từng dòng một.
"""

chars = sorted(set(text))                      # các ký tự có trong bài
stoi = {ch: i for i, ch in enumerate(chars)}   # chữ -> số
itos = {i: ch for ch, i in stoi.items()}       # số -> chữ

vocab_size = len(chars)                                            # từ điển
encode = lambda s: [stoi[ch] for ch in s if ch in stoi]            # chữ -> số
decode = lambda ids: "".join(itos[i] for i in ids)                 # số -> chữ

data = torch.tensor(encode(text), dtype=torch.long)


# =====================================================================
# 2. CÁC CON SỐ
# =====================================================================
# Vặn to lên thì thông minh hơn nhưng chạy chậm hơn.

block_size = 64    # một câu dài tối đa bao nhiêu token
n_embd = 96        # "bộ não" của mỗi token rộng bao nhiêu ô
n_head = 4         # chia bộ não thành mấy đầu để nhìn nhiều kiểu khác nhau
n_layer = 3        # có mấy tầng suy nghĩ
dropout = 0.0      # bỏ ngẫu nhiên vài nơ-ron (0.0 = tắt)
batch_size = 16    # mỗi bước học mấy đoạn văn
lr = 3e-3          # học nhanh hay chậm


# =====================================================================
# 3. AI NHÌN VÀO TỪ NÀO?  (Attention)
# =====================================================================
# Mỗi token tự hỏi: "Trong các token đứng trước, token nào giống mình nhất?"
#
#     Q = Question (tôi đang cần tìm gì?)
#     K = Key      (tôi có thông tin gì?)
#     V = Value    (nội dung của tôi là gì?)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)  # trộn các đầu lại thành một
        self.dropout = nn.Dropout(dropout)

        # Mặt nạ hình tam giác: token chỉ được nhìn về phía trước.
        # Nếu nhìn được cả tương lai thì lúc học nó đã "chép bài".
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        # Bật lên để lưu lại bản đồ attention (trang web studio/ dùng).
        self.capture = False
        self.attention_map = None

    def forward(self, x):
        B, T, C = x.shape                      # B câu, T token, C ô
        head_dim = C // n_head

        # Tách mỗi cái thành n_head đầu: [B, T, C] -> [B, n_head, T, head_dim]
        q = self.q(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, n_head, head_dim).transpose(1, 2)

        # Đây chính là công thức attention: softmax(Q·Kᵀ / √d)
        score = q @ k.transpose(-1, -2) / head_dim**0.5
        score = score.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weight = F.softmax(score, dim=-1)      # mỗi hàng cộng lại = 1
        weight = self.dropout(weight)

        if self.capture:
            self.attention_map = weight.detach()

        # Lấy thông tin của các token theo đúng tỉ lệ vừa tính.
        y = weight @ v
        y = y.transpose(1, 2).reshape(B, T, C)

        return self.dropout(self.proj(y))


# =====================================================================
# 4. AI SUY NGHĨ BÊN TRONG THẾ NÀO?  (MLP)
# =====================================================================
# Attention lo việc các token trao đổi với nhau.
# MLP là chỗ model thật sự "tính toán" một mình: mở rộng ra 4 lần rồi thu lại.


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# =====================================================================
# 5. MỘT TẦNG GỒM NHỮNG GÌ?  (Block)
# =====================================================================
# Một tầng chỉ có hai việc, và mỗi việc đều có ĐƯỜNG TẮT (residual):
#
#     x = x + <việc vừa làm>
#
# Đường tắt là bí quyết giúp model nhiều tầng vẫn học được: mỗi tầng chỉ
# cần học "ghi thêm một chút" chứ không phải học lại từ đầu.
# LayerNorm giữ cho các con số không quá to cũng không quá nhỏ.


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # các token nói chuyện với nhau
        x = x + self.mlp(self.norm2(x))    # suy nghĩ riêng
        return x


# =====================================================================
# 6. GHÉP TẤT CẢ LẠI  (Model)
# =====================================================================


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()

        # Mỗi chữ có một vector riêng, và mỗi VỊ TRÍ cũng có một vector riêng.
        # Cộng hai thứ lại: model biết "chữ gì" và "ở chỗ nào".
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)

        # LM Head: biến vector thành điểm cho MỌI chữ trong từ điển.
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: dùng chung một bảng tra ở hai chỗ cho đỡ tốn.
        self.lm_head.weight = self.token_embedding.weight

        # Khởi tạo trọng số thật nhỏ, để lúc đầu model "đoán bừa nhẹ nhàng".
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.lm_head(x)               # [B, T, vocab_size]

        if targets is None:
            return logits, None

        # So điều model đoán với đáp án. Đây là "điểm phạt" (loss).
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss


# =====================================================================
# 7. LÀM SAO AI HỌC, VÀ LÀM SAO AI NÓI?
# =====================================================================


def get_batch():
    """Bốc ngẫu nhiên vài đoạn văn trong bài.

    x = đoạn văn
    y = cũng đoạn đó nhưng dịch lên 1 chữ -> đó là "đáp án" phải đoán.
    """
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


@torch.no_grad()  # không cần học nữa, chỉ cần đoán
def generate(model, idx, max_new_tokens=200, temperature=0.8):
    """Đoán từng chữ một. Đoán xong chữ này thì ghép vào rồi đoán chữ tiếp."""
    for _ in range(max_new_tokens):
        # Chỉ nhìn được block_size token gần nhất, vì bảng vị trí chỉ có thế.
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, idx_next], dim=1)

    return idx


if __name__ == "__main__":
    torch.manual_seed(0)

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    model = TinyGPT()
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 62)
    print("TinyGPT — LLM nhỏ nhất có thể")
    print("=" * 62)
    print(f"Bài học : {len(text)} ký tự, gồm {vocab_size} ký tự khác nhau")
    print(f"Model   : {n_params:,} thông số | {n_layer} tầng | {n_head} đầu | não rộng {n_embd}")

    # --- Trước khi học thì nó nói gì? ---
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    before = decode(generate(model, start, 70).tolist()[0])

    print("\nTrước khi học, nó chỉ nói linh tinh:")
    print("  " + before.replace("\n", " "))

    # --- Học ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\nBắt đầu học {steps} bước...")
    model.train()

    history = []

    for step in range(steps + 1):
        x, y = get_batch()
        logits, loss = model(x, y)

        optimizer.zero_grad()   # xoá lỗi cũ
        loss.backward()         # tính xem nên sửa thông số nào
        optimizer.step()        # sửa thông số

        history.append([step, round(loss.item(), 4)])

        if step % 100 == 0:
            print(f"  bước {step:5d}   loss = {loss.item():.3f}")

    # --- Sau khi học thì nó nói gì? ---
    model.eval()
    start = torch.tensor([encode("Tôi")], dtype=torch.long)
    after = decode(generate(model, start, 400).tolist()[0])

    print("\nSau khi học, nó viết được thế này:")
    print("  " + after.replace("\n", "\n  "))

    # Lưu lại để studio/ mở ra xem được, và để lần sau khỏi học lại.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_gpt.pt")
    torch.save(
        {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "chars": chars,
            "model": model.state_dict(),
            "history": history,
        },
        out,
    )
    print(f"\nĐã lưu model vào {out}")

    print("\n" + "=" * 62)
    print("Muốn nó giỏi hơn: tăng n_layer, n_embd, hoặc học nhiều bước hơn.")
    print("Muốn hiểu sâu hơn:  mở  mini_deepseek/  (Cấp 2) xem MoE và RoPE.")
    print("=" * 62)
