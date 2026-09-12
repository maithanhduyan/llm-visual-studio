"""
config.py — Các con số của model (Cấp 3).

So với Cấp 2, ở đây có thêm mấy núm mới:

    n_kv_heads          GQA: số đầu dùng cho K và V, ít hơn số đầu Q
    window              cửa sổ trượt: mỗi token chỉ nhìn lại một đoạn gần nhất
    n_shared            chuyên gia dùng chung: ai cũng hỏi, không cần router
    router_bias_speed   tốc độ chỉnh điểm thiên vị của router
    n_streams           số dòng suy nghĩ chạy song song (hyper-connections)
"""

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- Từ điển -------------------------------------------------------
    vocab_size: int = 256  # tokenizer sẽ tự điền lại cho đúng

    # --- Kích thước ----------------------------------------------------
    d_model: int = 192
    n_layers: int = 3
    max_seq_len: int = 512
    dropout: float = 0.0

    # --- Attention -----------------------------------------------------
    # GQA: nhiều đầu Q, ít đầu K/V.
    # Token nào cũng cần "hỏi" (Q), nhưng K và V của các đầu giống nhau
    # nên chia sẻ được -> đệm K/V nhỏ đi n_kv_heads/n_heads lần.
    n_heads: int = 6
    n_kv_heads: int = 2

    # Cửa sổ trượt: token chỉ nhìn lại `window` token gần nhất.
    # 0 = nhìn hết cả câu (không dùng cửa sổ).
    #
    # Phải NHỎ HƠN block_size lúc học, nếu không nó chẳng có tác dụng gì:
    # câu học dài 128 mà cửa sổ cũng 128 thì chưa bao giờ chạm tới cửa sổ.
    window: int = 64

    # --- Chuyên gia ----------------------------------------------------
    # Nhiều chuyên gia NHỎ (fine-grained) tốt hơn ít chuyên gia to.
    n_experts: int = 8
    top_k: int = 2
    n_shared: int = 1  # chuyên gia dùng chung, token nào cũng hỏi
    expert_hidden: int = 96  # độ rộng bên trong mỗi chuyên gia (SwiGLU)

    # Mỗi bước học, điểm thiên vị được cộng/trừ bao nhiêu.
    # To quá thì router rung lắc, nhỏ quá thì chia việc mãi không đều.
    router_bias_speed: float = 0.01

    # --- Dòng suy nghĩ -------------------------------------------------
    # 1 = đường tắt bình thường. 4 = hyper-connections (xem mhc.py).
    n_streams: int = 4

    # --- Khác ----------------------------------------------------------
    # Bật để lưu bản đồ attention + đường đi chuyên gia (dùng cho studio/).
    capture: bool = False

    def __post_init__(self):
        """Kiểm tra các con số có hợp lý không, báo lỗi sớm cho dễ sửa."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) phải chia hết cho n_heads ({self.n_heads})."
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) phải chia hết cho n_kv_heads ({self.n_kv_heads}) "
                "— vì mỗi đầu K/V được chia sẻ cho một nhóm đầu Q."
            )
        if self.top_k > self.n_experts:
            raise ValueError(
                f"top_k ({self.top_k}) không được lớn hơn n_experts ({self.n_experts})."
            )
        if self.n_streams < 1:
            raise ValueError(f"n_streams ({self.n_streams}) phải >= 1.")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len ({self.max_seq_len}) phải >= 1.")

    def to_dict(self) -> dict:
        """Đổi thành dict để lưu vào file model."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Dựng lại từ dict đã lưu."""
        return cls(**data)
