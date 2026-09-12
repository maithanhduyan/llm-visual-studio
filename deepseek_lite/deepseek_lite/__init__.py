"""
deepseek_lite — một LLM nhỏ kiểu DeepSeek, để học.

Cấp 3 trong lộ trình: thêm GQA, cửa sổ trượt, KV cache, chuyên gia dùng
chung, chia việc không cần hàm phạt, và nhiều dòng suy nghĩ song song.

    Cấp 1  tiny_gpt/         Attention + MLP, một file
    Cấp 2  mini_deepseek/    thêm RoPE, SwiGLU, MoE
    Cấp 3  deepseek_lite/    thêm GQA, cửa sổ trượt, KV cache, mHC   <-- đây

Ví dụ dùng như thư viện:

    from deepseek_lite import Config, DeepSeekLite, Tokenizer

    config = Config(vocab_size=100)
    model = DeepSeekLite(config)
    logits, loss = model(tokens, targets)
"""

from .cache import KVCache
from .config import Config
from .model import DeepSeekLite
from .tokenizer import Tokenizer

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DeepSeekLite",
    "KVCache",
    "Tokenizer",
    "__version__",
]
