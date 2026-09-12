"""
config.py — Các con số của model.

Mỗi con số ở đây giống như một "núm vặn".
Vặn to lên thì model thông minh hơn, nhưng chạy chậm hơn.
Muốn model to hơn? Chỉ cần đổi số, KHÔNG cần sửa kiến trúc.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Từ điển: model biết bao nhiêu "chữ" khác nhau.
    # Con số này sẽ được tokenizer tự điền lại cho đúng.
    vocab_size: int = 256

    # "Bộ não" của mỗi token rộng bao nhiêu ô.
    d_model: int = 128

    # Số "con mắt" để nhìn các token khác cùng lúc.
    # d_model phải chia hết cho n_heads.
    n_heads: int = 4

    # Model có bao nhiêu tầng suy nghĩ. Càng nhiều tầng càng "sâu".
    n_layers: int = 4

    # Có bao nhiêu chuyên gia, và mỗi lần hỏi mấy chuyên gia.
    n_experts: int = 4
    top_k: int = 2

    # Một câu dài tối đa bao nhiêu token.
    max_seq_len: int = 256

    # Bỏ ngẫu nhiên vài nơ-ron để model không học vẹt.
    # 0.0 = tắt (dễ hiểu hơn khi mới học).
    dropout: float = 0.0

    # Số "dòng suy nghĩ" chạy song song (ý tưởng mHC trong hình).
    # 1 = bình thường. 4 = bật bản mini của mHC trong file mhc.py.
    n_streams: int = 1

    # Bật lên để lưu lại bản đồ attention và đường đi của chuyên gia.
    # Trang web trong thư mục studio/ dùng cờ này để vẽ hình.
    # Lúc học bình thường thì để tắt cho nhanh.
    capture: bool = False
