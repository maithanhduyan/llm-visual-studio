"""
tokenizer.py — Chữ biến thành số thế nào?

Đây là ý tưởng quan trọng nhất của cả project:

    Model KHÔNG nhìn thấy chữ. Nó chỉ nhìn thấy số.

    "hello"  ->  [7, 4, 11, 11, 14]

Bản này là tokenizer "theo từng ký tự" (character tokenizer):
mỗi ký tự là một số. Đơn giản nhất để hiểu.

Bản thật của DeepSeek dùng BPE, mỗi mảnh chữ là một số,
nên 129.000 token thay vì vài trăm ký tự. Ý tưởng vẫn y hệt.
"""


class Tokenizer:
    def __init__(self, text):
        # Lấy ra tất cả ký tự có trong sách, xếp theo thứ tự.
        chars = sorted(set(text))

        # Bảng dịch: chữ -> số
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Bảng dịch ngược: số -> chữ
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # Giữ lại danh sách ký tự để lưu vào file model.
        self.chars = chars

    def encode(self, text):
        """Chữ -> số. Ký tự lạ (không có trong sách) thì bỏ qua."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens):
        """Số -> chữ."""
        return "".join(self.itos[t] for t in tokens)

    @property
    def vocab_size(self):
        """Từ điển có bao nhiêu chữ."""
        return len(self.stoi)

    @classmethod
    def load(cls, chars):
        """Dựng lại tokenizer đã lưu trong file model."""
        return cls("".join(chars))
