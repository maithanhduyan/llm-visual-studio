"""
deepseek_math — dạy model làm toán, từng bước một.

Bốn cấp trước hỏi: **model hoạt động thế nào?**
Dự án này hỏi câu khác: **model học được kỹ năng gì, và học theo trình tự nào?**

Năm bậc:

    1. Cộng một chữ số, không nhớ        học thuộc được
    2. Cộng một chữ số, có nhớ           học thuộc được
    3. Cộng hai chữ số, trả lời thẳng    phải tính
    4. Cộng hai chữ số, viết từng bước   phải tính, nhưng có chỗ để tính
    5. Cộng ba chữ số, viết từng bước    chưa học bao giờ

Thí nghiệm chính: bậc 3 và bậc 4 là **cùng một tập bài**, chỉ khác chỗ có bắt
model viết ra từng bước hay không. Rồi đem cả hai đi thử với số ba chữ số.

    python -m deepseek_math steps        xem năm bậc
    python -m deepseek_math demo         xem vài bài mẫu
    python -m deepseek_math teach 1      dạy bậc 1
    python -m deepseek_math experiment   chạy thí nghiệm chính
"""

from .curriculum import LEVELS, Level, build_text
from .problems import Problem, direct, read_answer, scratchpad

__version__ = "0.1.0"

__all__ = [
    "LEVELS",
    "Level",
    "Problem",
    "build_text",
    "direct",
    "read_answer",
    "scratchpad",
    "__version__",
]
