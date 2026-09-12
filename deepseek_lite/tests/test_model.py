"""
Test model.

Bài test quan trọng nhất ở đây là `test_cache_gives_identical_results`.
Đó là điều kiện sống còn của KV cache: nếu sinh từng chữ mà cho ra kết quả
khác với chạy cả câu một lần, thì đệm K/V vô nghĩa dù nhanh đến đâu.
"""

from dataclasses import replace

import pytest
import torch

from deepseek_lite import DeepSeekLite, KVCache


def test_forward_shapes(small_config):
    model = DeepSeekLite(small_config).eval()
    ids = torch.randint(0, small_config.vocab_size, (2, 7))

    logits, loss = model(ids)

    assert logits.shape == (2, 7, small_config.vocab_size)
    assert loss is None, "không truyền đáp án thì không được tính loss"


def test_loss_needs_targets(small_config):
    model = DeepSeekLite(small_config).eval()
    ids = torch.randint(0, small_config.vocab_size, (2, 7))

    _, loss = model(ids, ids)

    assert loss is not None
    assert loss.item() > 0


def test_moe_makes_active_params_smaller_than_total(small_config):
    """Mỗi token chỉ hỏi top_k chuyên gia, nên không dùng hết thông số."""
    model = DeepSeekLite(small_config)

    total, active = model.parameter_counts()

    assert 0 < active < total


def test_single_stream_matches_plain_residual(small_config):
    """n_streams = 1 thì phải chạy được như đường tắt bình thường."""
    config = replace(small_config, n_streams=1)
    model = DeepSeekLite(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 7))

    with torch.no_grad():
        logits, _ = model(ids)

    assert logits.shape == (2, 7, config.vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("window", [0, 4])
def test_cache_gives_identical_results(small_config, window):
    """Sinh từng chữ phải cho ra y hệt chạy cả câu một lần.

    Thử cả hai trường hợp: không cửa sổ, và có cửa sổ (để kiểm tra cả
    đoạn cắt bớt K/V cũ trong attention.py).
    """
    config = replace(small_config, window=window)
    torch.manual_seed(0)
    model = DeepSeekLite(config).eval()

    ids = torch.randint(0, config.vocab_size, (1, 20))
    split = 8

    with torch.no_grad():
        full, _ = model(ids)  # chạy cả câu một lần

        cache = KVCache(config, batch_size=1)
        prefill, _ = model(ids[:, :split], cache=cache, start_pos=0)

        assert torch.allclose(prefill, full[:, :split], atol=1e-4), "phần đọc đầu bị lệch"

        # Rồi sinh từng chữ một, đúng như lúc viết văn.
        for t in range(split, ids.shape[1]):
            step, _ = model(ids[:, t : t + 1], cache=cache, start_pos=t)
            assert torch.allclose(step[:, 0], full[:, t], atol=1e-4), f"lệch ở token {t}"


def test_cache_length_grows(small_config):
    model = DeepSeekLite(small_config).eval()
    cache = KVCache(small_config, batch_size=1)
    ids = torch.randint(0, small_config.vocab_size, (1, 5))

    with torch.no_grad():
        model(ids, cache=cache, start_pos=0)
    assert cache.length == 5

    with torch.no_grad():
        model(ids[:, :1], cache=cache, start_pos=5)
    assert cache.length == 6


def test_gradients_reach_every_parameter(small_config):
    """Không được có thông số nào đứng ngoài đồ thị — kể cả chuyên gia hiếm dùng."""
    torch.manual_seed(0)
    model = DeepSeekLite(small_config)

    # Lô to để mọi chuyên gia đều được chọn ít nhất một lần.
    ids = torch.randint(0, small_config.vocab_size, (16, 64))

    _, loss = model(ids, ids)
    loss.backward()

    missing = [
        name
        for name, param in model.named_parameters()
        if param.grad is None or param.grad.abs().sum() == 0
    ]

    assert not missing, f"thông số không nhận gradient: {missing[:6]}"


def test_update_router_bias_runs(small_config):
    model = DeepSeekLite(small_config)
    ids = torch.randint(0, small_config.vocab_size, (4, 16))

    model.train()
    model(ids, ids)
    model.update_router_bias()

    # Sau khi chỉnh thì bộ đếm phải được xoá sạch.
    assert all(block.moe.load_count.sum() == 0 for block in model.blocks)


def test_no_nan_with_long_sequence(small_config):
    """Câu dài chạm đúng giới hạn max_seq_len thì không được sinh ra NaN."""
    config = replace(small_config, window=8)
    model = DeepSeekLite(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))

    with torch.no_grad():
        logits, _ = model(ids)

    assert torch.isfinite(logits).all()
