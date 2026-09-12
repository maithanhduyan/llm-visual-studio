"""
paged_cache.py — Đệm K/V chia thành khối (PagedAttention).

Cách thường: mỗi câu xin một vùng nhớ dài bằng `max_seq_len`. Câu dài 20
token vẫn chiếm chỗ của 512 token. Phục vụ 8 người là hết nhớ, trong khi
thực dùng chưa tới 4%.

Cách chia khối: cắt bộ nhớ thành nhiều khối nhỏ (16 token mỗi khối), câu nào
cần thì phát khối, dài bao nhiêu dùng bấy nhiêu.

    Cách thường              Cách chia khối
    ┌──────────────┐         ┌──┬──┬──┬──┬──┬──┬──┬──┐
    │ câu 1 (20)   │ 512     │1 │1 │  │2 │  │3 │3 │  │  ← khối cấp phát
    ├──────────────┤ chỗ     └──┴──┴──┴──┴──┴──┴──┴──┘
    │ câu 2 (20)   │ 512     mỗi khối 16 token, khối nào chưa dùng thì
    ├──────────────┤ chỗ     để đó cho câu khác
    │ ...          │
    └──────────────┘

Đổi lại phải có một "bảng khối" cho mỗi câu: khối thứ 3 của câu 1 nằm ở
ngăn nào của bộ nhớ. Bảng đó gọi là block table.

Ở đây khi đọc ra thì ta GOM các khối lại thành một tensor liền mạch rồi mới
cho attention dùng. PagedAttention thật không gom — nó viết kernel đọc thẳng
từ khối để đỡ phải copy. Bộ nhớ tiết kiệm thì giống hệt nhau, chỉ khác là
bản thật nhanh hơn.
"""

from __future__ import annotations

import torch

from .base_model import Config


class PagedKVCache:
    def __init__(self, config: Config, num_blocks: int = 64, block_size: int = 16) -> None:
        self.config = config
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.max_len = config.max_seq_len

        shape = (num_blocks, block_size, self.n_kv_heads, self.head_dim)
        self.key_pool = [torch.zeros(shape) for _ in range(config.n_layers)]
        self.value_pool = [torch.zeros(shape) for _ in range(config.n_layers)]

        # Khối nào còn trống.
        self.free_blocks = list(range(num_blocks))

        # Mỗi câu một bảng khối: bảng[3] = ngăn chứa khối thứ 3 của câu đó.
        self.tables: dict[int, list[int]] = {}
        self.committed_lengths: dict[int, int] = {}

        self.slots: list[int] = []  # các câu đang có trong lô này
        self._lengths = torch.zeros(0, dtype=torch.long)

    # ------------------------------------------------------------------
    # Cấp phát
    # ------------------------------------------------------------------

    def add_sequence(self, seq_id: int) -> bool:
        """Nhận một câu mới. Trả về False nếu hết chỗ."""
        if seq_id in self.tables:
            return True
        if not self.free_blocks:
            return False

        self.tables[seq_id] = []
        self.committed_lengths[seq_id] = 0
        return True

    def free_sequence(self, seq_id: int) -> None:
        """Trả hết khối của một câu về cho bộ nhớ chung."""
        for block in self.tables.pop(seq_id, []):
            self.free_blocks.append(block)
        self.committed_lengths.pop(seq_id, None)

    def begin_batch(self, seq_ids: list[int]) -> None:
        """Cho biết lô sắp chạy gồm những câu nào, theo thứ tự nào."""
        self.slots = list(seq_ids)

    def _take_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError(
                "Hết khối K/V. Giảm số câu phục vụ cùng lúc, hoặc tăng num_blocks."
            )
        return self.free_blocks.pop()

    def can_admit(self, seq_id: int) -> bool:
        return seq_id in self.tables or bool(self.free_blocks)

    # ------------------------------------------------------------------
    # Giao diện giống KVCache của Cấp 3
    # ------------------------------------------------------------------

    @property
    def lengths(self) -> torch.Tensor:
        """Số token hợp lệ của từng câu trong lô (đã tính cả token vừa thêm)."""
        return self._lengths

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Cất K/V của token mới vào ĐÚNG KHỐI của từng câu, rồi gom ra."""
        T = k.shape[-2]
        lengths: list[int] = []

        for batch_index, seq_id in enumerate(self.slots):
            start = self.committed_lengths[seq_id]

            for offset in range(T):
                position = start + offset
                block_index = position // self.block_size
                slot = position % self.block_size

                table = self.tables[seq_id]
                while len(table) <= block_index:
                    table.append(self._take_block())

                block = table[block_index]
                self.key_pool[layer][block, slot] = k[batch_index, :, offset, :]
                self.value_pool[layer][block, slot] = v[batch_index, :, offset, :]

            lengths.append(start + T)

        self._lengths = torch.tensor(lengths, dtype=torch.long)
        return self._gather(layer, lengths)

    def advance(self, T: int) -> None:
        """Sau khi mọi tầng đã cất xong thì mới tính là đã thêm token."""
        for seq_id in self.slots:
            self.committed_lengths[seq_id] += T

    def reset(self) -> None:
        self.slots = []
        self._lengths = torch.zeros(0, dtype=torch.long)

    # ------------------------------------------------------------------
    # Đọc ra
    # ------------------------------------------------------------------

    def _read_sequence(self, pool, layer: int, seq_id: int, length: int) -> torch.Tensor:
        """Ghép các khối của một câu lại thành một dãy liền mạch."""
        if length == 0:
            return torch.zeros(0, self.n_kv_heads, self.head_dim)

        table = self.tables[seq_id]
        needed = (length + self.block_size - 1) // self.block_size
        blocks = torch.stack([pool[layer][block] for block in table[:needed]])

        return blocks.reshape(-1, self.n_kv_heads, self.head_dim)[:length]

    def _gather(self, layer: int, lengths: list[int]):
        """Gom K/V của cả lô thành tensor chữ nhật, đệm cho bằng nhau."""
        batch = len(self.slots)
        longest = max(lengths) if lengths else 1

        k_out = torch.zeros(batch, self.n_kv_heads, longest, self.head_dim)
        v_out = torch.zeros_like(k_out)

        for batch_index, seq_id in enumerate(self.slots):
            length = lengths[batch_index]
            if length == 0:
                continue

            k_seq = self._read_sequence(self.key_pool, layer, seq_id, length)
            v_seq = self._read_sequence(self.value_pool, layer, seq_id, length)

            k_out[batch_index, :, :length, :] = k_seq.transpose(0, 1)
            v_out[batch_index, :, :length, :] = v_seq.transpose(0, 1)

        return k_out, v_out

    # ------------------------------------------------------------------
    # Đo đạc
    # ------------------------------------------------------------------

    @property
    def blocks_in_use(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def used_bytes(self) -> int:
        """Bộ nhớ thật sự đang dùng, tính theo token đã cất."""
        tokens = sum(self.committed_lengths.values())
        per_token = 2 * self.n_kv_heads * self.head_dim * 4 * self.config.n_layers
        return tokens * per_token

    @property
    def reserved_bytes(self) -> int:
        """Bộ nhớ đã cấp phát (theo khối)."""
        per_block = 2 * self.block_size * self.n_kv_heads * self.head_dim * 4 * self.config.n_layers
        return self.blocks_in_use * per_block

    @property
    def waste_bytes(self) -> int:
        """Chỗ trống nằm trong khối đã cấp phát nhưng chưa dùng."""
        return max(self.reserved_bytes - self.used_bytes, 0)

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(khối {self.blocks_in_use}/{self.num_blocks}, "
            f"câu {len(self.tables)}, lô {len(self.slots)})"
        )


class ContiguousKVCache:
    """Cách thường, để so sánh: mỗi câu xin sẵn `max_seq_len` chỗ.

    Không dùng trong máy phục vụ — chỉ để đo xem chia khối tiết kiệm bao nhiêu.
    """

    def __init__(self, config: Config, capacity: int) -> None:
        self.capacity = capacity
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.n_kv_heads = config.n_kv_heads

    @property
    def reserved_bytes(self) -> int:
        """Mỗi câu chiếm trọn `max_seq_len`, dù có dùng hết hay không."""
        per_token = 2 * self.n_kv_heads * self.head_dim * 4 * self.config.n_layers
        return self.capacity * self.config.max_seq_len * per_token
