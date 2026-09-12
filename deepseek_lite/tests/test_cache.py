"""Test đệm K/V: cất, lấy ra, đếm độ dài, và báo lỗi khi tràn."""

import pytest
import torch

from deepseek_lite import KVCache


def test_append_returns_everything_so_far(small_config):
    cache = KVCache(small_config, batch_size=1)
    shape = (1, small_config.n_kv_heads, 3, small_config.d_model // small_config.n_heads)

    k1, v1 = cache.append(0, torch.randn(shape), torch.randn(shape))
    assert k1.shape[-2] == 3  # mới cất 3 token

    k2, _ = cache.append(0, torch.randn(shape), torch.randn(shape))
    assert k2.shape[-2] == 3  # chưa advance() thì vẫn ghi đè chỗ cũ

    cache.advance(3)

    k3, _ = cache.append(0, torch.randn(shape), torch.randn(shape))
    assert k3.shape[-2] == 6  # giờ mới thành 6


def test_layers_are_independent(small_config):
    cache = KVCache(small_config, batch_size=1)
    head_dim = small_config.d_model // small_config.n_heads
    shape = (1, small_config.n_kv_heads, 2, head_dim)

    cache.append(0, torch.ones(shape), torch.ones(shape))
    cache.append(1, torch.zeros(shape), torch.zeros(shape))
    cache.advance(2)

    # Mỗi tầng có đệm riêng: tầng 0 toàn số 1, tầng 1 toàn số 0.
    so_phan_tu = 1 * small_config.n_kv_heads * 2 * head_dim

    assert cache.keys[0][:, :, :2].sum() == so_phan_tu
    assert cache.keys[1][:, :, :2].sum() == 0


def test_reset(small_config):
    cache = KVCache(small_config, batch_size=1)
    cache.advance(7)

    cache.reset()

    assert cache.length == 0


def test_overflow_raises(small_config):
    cache = KVCache(small_config, batch_size=1)
    shape = (1, small_config.n_kv_heads, small_config.max_seq_len + 1,
             small_config.d_model // small_config.n_heads)

    with pytest.raises(ValueError, match="max_seq_len"):
        cache.append(0, torch.randn(shape), torch.randn(shape))


def test_memory_is_smaller_with_gqa(small_config):
    cache = KVCache(small_config)

    # 2 đầu K/V, mỗi đầu 2 tensor (K và V), 4 byte mỗi số.
    head_dim = small_config.d_model // small_config.n_heads
    expected = 2 * 2 * small_config.max_seq_len * head_dim * 4 * small_config.n_layers

    assert cache.memory_bytes == expected
