"""Test attention: không nhìn tương lai, có cửa sổ trượt, và GQA đúng kích thước."""

from dataclasses import replace

import pytest
import torch

from deepseek_lite.attention import Attention


def _run(config, T):
    """Chạy attention với capture bật, trả về bản đồ attention [H, T, T]."""
    attention = Attention(config, layer_index=0).eval()
    x = torch.randn(1, T, config.d_model)

    with torch.no_grad():
        attention(x)

    return attention.attention_map[0]


def test_is_causal(small_config):
    """Token không được nhìn vào tương lai."""
    T = 8
    config = replace(small_config, capture=True)
    weights = _run(config, T)

    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    assert (weights[:, future] == 0).all(), "có token đang nhìn vào tương lai"


def test_rows_sum_to_one(small_config):
    """Mỗi token chia sự chú ý của mình thành các phần cộng lại bằng 1."""
    config = replace(small_config, capture=True)
    weights = _run(config, 8)

    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_first_token_only_sees_itself(small_config):
    config = replace(small_config, capture=True)
    weights = _run(config, 6)

    # Token đầu tiên không có ai đứng trước, nên chỉ nhìn chính nó.
    assert torch.allclose(weights[:, 0, 0], torch.ones_like(weights[:, 0, 0]), atol=1e-5)


def test_sliding_window_hides_the_past(small_config):
    T = 10
    config = replace(small_config, capture=True, window=3)
    weights = _run(config, T)

    # Token thứ 9 chỉ được nhìn các token 7, 8, 9.
    assert (weights[:, 9, :7] == 0).all(), "cửa sổ trượt không che token cũ"
    assert weights[:, 9, 7:].sum() > 0, "cửa sổ trượt che nhầm cả token gần"


def test_no_window_means_see_everything(small_config):
    T = 10
    config = replace(small_config, capture=True, window=0)
    weights = _run(config, T)

    # Không cửa sổ thì token cuối nhìn được tất cả các token trước.
    assert (weights[:, 9, :10] > 0).all()


def test_gqa_shrinks_kv(small_config):
    """K và V phải hẹp hơn Q đúng theo tỉ lệ n_heads / n_kv_heads."""
    attention = Attention(small_config, layer_index=0)
    head_dim = small_config.d_model // small_config.n_heads

    assert attention.q_proj.out_features == small_config.n_heads * head_dim
    assert attention.k_proj.out_features == small_config.n_kv_heads * head_dim
    assert attention.v_proj.out_features == small_config.n_kv_heads * head_dim


def test_rejects_heads_not_divisible():
    from deepseek_lite import Config

    with pytest.raises(ValueError, match="chia hết"):
        Config(d_model=32, n_heads=5, n_kv_heads=2)
