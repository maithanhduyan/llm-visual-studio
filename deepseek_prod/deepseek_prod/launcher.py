"""
launcher.py — Chạy nhiều tiến trình.

Trên Windows, `torchrun --standalone` không tạo được TCP store, còn
`multiprocessing.spawn` thì treo. Cách chạy được (đã thử) là:

    multiprocessing.Process  +  init_method="file:///..."

nên file này tự lo việc đó, khỏi phải nhớ.

Cách dùng:

    from .launcher import run_distributed

    def worker(rank, world, queue):
        ...  # ở đây đã init_process_group xong
        queue.put({"rank": rank, "loss": 1.23})

    results = run_distributed(worker, world_size=2, timeout=300)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import threading
import traceback
from pathlib import Path

import torch.distributed as dist


def _store_url() -> str:
    """Địa chỉ file dùng làm nơi các tiến trình gặp nhau."""
    folder = tempfile.mkdtemp(prefix="deepseek-prod-store-")
    return "file:///" + Path(folder, "store").as_posix()


def _entry(worker, rank: int, world_size: int, store: str, queue, args, timeout: float):
    """Thân của một tiến trình con: vào nhóm, chạy việc, rồi thoát cho sạch."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=store,
            rank=rank,
            world_size=world_size,
            timeout=__import__("datetime").timedelta(seconds=timeout),
        )
        try:
            worker(rank, world_size, queue, *args)
        finally:
            try:
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        os._exit(1)


def run_distributed(worker, world_size: int, *args, timeout: float = 300.0) -> list:
    """Chạy `worker` trên `world_size` tiến trình, gom kết quả trả về.

    `worker` phải là hàm ở cấp module (pickle được), có chữ ký:
        worker(rank, world_size, queue, *args)
    và nên `queue.put(...)` kết quả cần kiểm tra.

    Chú ý về cái bẫy đã sập một lần rồi: KHÔNG được `join()` trước khi đọc
    queue. Nếu kết quả lớn hơn bộ đệm của đường ống (~64 KB), tiến trình con
    sẽ đứng chặn ở `queue.put` và không bao giờ thoát — cha chờ con, con chờ
    cha đọc. Nên phải có một luồng đọc queue song song với lúc chờ.
    """
    if world_size < 1:
        raise ValueError(f"world_size phải >= 1, đang là {world_size}.")

    if world_size == 1:
        # Chạy một tiến trình thì không cần khởi động nhóm làm gì cho mệt.
        queue = mp.get_context("spawn").Queue()
        worker(0, 1, queue, *args)
        return _drain(queue, 1)

    context = mp.get_context("spawn")
    queue = context.Queue()
    store = _store_url()

    processes = [
        context.Process(
            target=_entry,
            args=(worker, rank, world_size, store, queue, args, timeout),
        )
        for rank in range(world_size)
    ]

    # Luồng này đọc queue LIÊN TỤC trong lúc các tiến trình còn chạy.
    collected: list = []

    def collect() -> None:
        for _ in range(world_size):
            try:
                collected.append(queue.get())
            except Exception:  # noqa: BLE001
                return

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=timeout)

    stuck = [p for p in processes if p.is_alive()]
    if stuck:
        for process in stuck:
            process.kill()
        raise TimeoutError(
            f"{len(stuck)}/{world_size} tiến trình không xong sau {timeout:.0f} giây. "
            "Nhiều khả năng có tiến trình chết giữa chừng nên những tiến trình "
            "còn lại đứng chờ ở một phép giao tiếp tập thể."
        )

    reader.join(timeout=10)

    if len(collected) != world_size:
        raise RuntimeError(
            f"Chỉ {len(collected)}/{world_size} tiến trình gửi kết quả về. "
            "Có tiến trình đã chết trước khi báo cáo."
        )

    return sorted(collected, key=lambda item: item.get("rank", 0))


def _drain(queue, world_size: int) -> list:
    """Lấy hết kết quả các tiến trình gửi về."""
    results = []
    while not queue.empty():
        results.append(queue.get())

    if len(results) != world_size:
        raise RuntimeError(
            f"Chỉ {len(results)}/{world_size} tiến trình gửi kết quả về. "
            "Có tiến trình đã chết trước khi báo cáo."
        )

    return sorted(results, key=lambda item: item.get("rank", 0))
