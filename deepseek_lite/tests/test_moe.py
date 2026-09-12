"""Test MoE: chọn đúng top_k, chuyên gia dùng chung luôn chạy, và chia việc công bằng."""

from dataclasses import replace

import torch
import torch.nn as nn

from deepseek_lite.moe import MoE


def test_output_shape(small_config):
    moe = MoE(small_config).eval()
    x = torch.randn(2, 5, small_config.d_model)

    with torch.no_grad():
        out = moe(x)

    assert out.shape == x.shape


def test_picks_exactly_top_k_distinct_experts(small_config):
    # Bật capture để moe.routing ghi lại lựa chọn của router.
    moe = MoE(replace(small_config, capture=True)).eval()
    x = torch.randn(2, 5, small_config.d_model)

    with torch.no_grad():
        moe(x)

    indices, weights = moe.routing

    assert indices.shape[-1] == small_config.top_k

    # Trong một token, không được chọn cùng một chuyên gia hai lần.
    for row in indices.reshape(-1, small_config.top_k):
        assert len(set(row.tolist())) == small_config.top_k

    # Trọng số của mỗi token cộng lại = 1.
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5)


def test_shared_expert_always_contributes(small_config):
    """Tắt hết chuyên gia riêng thì vẫn phải có đầu ra, nhờ chuyên gia dùng chung."""
    moe = MoE(small_config).eval()
    x = torch.randn(2, 5, small_config.d_model)

    with torch.no_grad():
        full = moe(x).abs().sum()

        for expert in moe.experts:
            nn.init.zeros_(expert.out.weight)
            nn.init.zeros_(expert.out.bias)

        only_shared = moe(x).abs().sum()

    assert full > 0
    assert only_shared > 0, "chuyên gia dùng chung không đóng góp gì"


def test_routed_experts_contribute(small_config):
    """Tắt chuyên gia dùng chung thì đầu ra phải khác đi."""
    moe = MoE(small_config).eval()
    x = torch.randn(2, 5, small_config.d_model)

    with torch.no_grad():
        before = moe(x).clone()

        for expert in moe.shared_experts:
            nn.init.zeros_(expert.out.weight)
            nn.init.zeros_(expert.out.bias)

        after = moe(x)

    assert not torch.allclose(before, after)


def test_no_shared_experts_still_works(small_config):
    config = replace(small_config, n_shared=0)
    moe = MoE(config).eval()
    x = torch.randn(2, 5, config.d_model)

    with torch.no_grad():
        out = moe(x)

    assert out.shape == x.shape
    assert out.abs().sum() > 0


def test_counting_only_happens_while_training(small_config):
    """Lúc đánh giá thì không được đếm, nếu không bảng thống kê bị lẫn."""
    moe = MoE(small_config)
    x = torch.randn(2, 5, small_config.d_model)

    moe.eval()
    with torch.no_grad():
        moe(x)
    assert moe.load_count.sum() == 0, "đang đánh giá mà vẫn đếm"

    moe.train()
    moe(x)
    assert moe.load_count.sum() > 0, "đang học mà không đếm"


def test_bias_punishes_overused_and_rewards_underused(small_config):
    """Đây là toàn bộ thuật toán chia đều — ba dòng ở moe.py."""
    moe = MoE(small_config)

    # Giả sử chuyên gia 0 được chọn quá nhiều, hai chuyên gia cuối quá ít.
    with torch.no_grad():
        moe.load_count.copy_(torch.tensor([100.0, 40.0, 5.0, 5.0]))

    moe.update_bias()

    assert moe.expert_bias[0] < 0, "chuyên gia quá tải phải bị trừ điểm"
    assert moe.expert_bias[3] > 0, "chuyên gia ít việc phải được cộng điểm"
    assert (moe.load_count == 0).all(), "phải xoá bộ đếm sau mỗi bước"


def test_balanced_load_gives_no_bias_change(small_config):
    """Chia đều hoàn hảo rồi thì đừng chỉnh gì nữa."""
    moe = MoE(small_config)

    with torch.no_grad():
        moe.load_count.fill_(25.0)

    moe.update_bias()

    assert (moe.expert_bias == 0).all()


def test_balance_improves_over_time(small_config):
    """Chạy nhiều bước thì việc chia phải đều lên, không được tệ đi."""
    torch.manual_seed(0)

    moe = MoE(small_config)
    x = torch.randn(8, 32, small_config.d_model)

    # Ép router thiên vị chuyên gia 0 một cách nhẹ nhàng (không bão hoà).
    with torch.no_grad():
        nn.init.zeros_(moe.router.weight)
        moe.router.weight[0, :] = 0.05

    moe.train()

    deviations = []
    for _ in range(40):
        with torch.no_grad():
            moe(x)
        share = moe.load_share()
        deviations.append((share - 1.0 / small_config.n_experts).abs().sum().item())
        moe.update_bias()

    early = sum(deviations[:10]) / 10
    late = sum(deviations[-10:]) / 10

    assert late <= early, f"chia việc tệ đi: {early:.3f} -> {late:.3f}"
