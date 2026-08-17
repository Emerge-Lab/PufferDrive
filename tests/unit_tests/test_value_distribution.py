"""Tests for the binned distributional value support (pufferlib/value_distribution.py).

Both modes share the bin support, the expectation and the cross-entropy loss; only
`target_probs` differs, so most cases are parametrized over both modes and the
mode-specific behaviour is isolated in the `two_hot_*` / `hl_gauss_*` cases.
"""

import math

import pytest
import torch

from pufferlib.value_distribution import (
    VALUE_DISTRIBUTION_HL_GAUSS,
    VALUE_DISTRIBUTION_MODES,
    VALUE_DISTRIBUTION_OFF,
    VALUE_DISTRIBUTION_TWO_HOT,
    make_binned_value_distribution,
)

NUM_BINS = 101
SUPPORT_MIN = -10.0
SUPPORT_MAX = 10.0
SIGMA_RATIO = 0.75
BINNED_MODES = (VALUE_DISTRIBUTION_TWO_HOT, VALUE_DISTRIBUTION_HL_GAUSS)
# Spans both supports plus targets past either end, so clamping is always exercised.
PROBE_TARGETS = (-12.0, -10.0, -7.0, -1.18, 0.0, 0.1, 6.17, 7.0, 9.999, 10.0, 25.0)
# log(0) would make the p*log(p) product NaN for the zero bins, so the reference
# entropy uses the xlogy convention and the logits floor the zeros instead.
LOG_PROB_FLOOR = 1e-30
# bf16 logits perturb the softmax enough to move the expectation over a 20-wide
# support by up to 0.019 (measured worst case over 500 seeds); the dtype assert
# is what pins the float32 accumulation.
BFLOAT16_EXPECTATION_ATOL = 0.05


def make_distribution(mode, num_bins=NUM_BINS, support_min=SUPPORT_MIN, support_max=SUPPORT_MAX):
    return make_binned_value_distribution(mode, num_bins, support_min, support_max, SIGMA_RATIO)


def test_off_mode_returns_none():
    assert make_distribution(VALUE_DISTRIBUTION_OFF) is None


@pytest.mark.parametrize("mode", ["twohot", "two-hot", "", "TWO_HOT", None])
def test_invalid_mode_raises(mode):
    with pytest.raises(ValueError) as excinfo:
        make_distribution(mode)
    for valid_mode in VALUE_DISTRIBUTION_MODES:
        assert valid_mode in str(excinfo.value)


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_invalid_num_bins_raises(mode):
    with pytest.raises(ValueError, match="critic_num_bins"):
        make_distribution(mode, num_bins=2)


@pytest.mark.parametrize("mode", BINNED_MODES)
@pytest.mark.parametrize("support_min,support_max", [(10.0, 10.0), (5.0, -5.0)])
def test_invalid_support_raises(mode, support_min, support_max):
    with pytest.raises(ValueError, match="critic_support_max"):
        make_distribution(mode, support_min=support_min, support_max=support_max)


@pytest.mark.parametrize("mode", BINNED_MODES)
@pytest.mark.parametrize("sigma_ratio", [0.0, -1.0])
def test_invalid_sigma_ratio_raises(mode, sigma_ratio):
    with pytest.raises(ValueError, match="critic_hl_gauss_sigma_ratio"):
        make_binned_value_distribution(mode, NUM_BINS, SUPPORT_MIN, SUPPORT_MAX, sigma_ratio)


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_support_geometry(mode):
    dist = make_distribution(mode)
    assert dist.bin_edges.shape == (NUM_BINS + 1,)
    assert dist.bin_centers.shape == (NUM_BINS,)
    assert dist.bin_width == pytest.approx((SUPPORT_MAX - SUPPORT_MIN) / NUM_BINS)
    assert dist.bin_centers[NUM_BINS // 2].item() == 0.0
    assert dist.bin_centers.mean().item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_target_probs_sum_to_one(mode):
    dist = make_distribution(mode)
    probs = dist.target_probs(torch.tensor(PROBE_TARGETS))
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(len(PROBE_TARGETS)), atol=1e-6, rtol=0)


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_target_probs_shape_flattens_input(mode):
    dist = make_distribution(mode)
    assert dist.target_probs(torch.zeros(7)).shape == (7, NUM_BINS)
    assert dist.target_probs(torch.zeros(4, 8)).shape == (32, NUM_BINS)


def test_two_hot_has_exactly_two_nonzero_bins():
    dist = make_distribution(VALUE_DISTRIBUTION_TWO_HOT)
    for target in (-9.0, -1.18, 0.0, 0.1, 3.3, 6.17, 9.0):
        probs = dist.target_probs(torch.tensor([target]))[0]
        nonzero = probs.nonzero().flatten()
        assert nonzero.numel() <= 2
        if nonzero.numel() == 2:
            assert nonzero[1].item() == nonzero[0].item() + 1


def test_two_hot_expectation_matches_target():
    dist = make_distribution(VALUE_DISTRIBUTION_TWO_HOT)
    targets = torch.linspace(dist.target_min, dist.target_max, 37)
    expectation = (dist.target_probs(targets) * dist.bin_centers).sum(dim=-1)
    torch.testing.assert_close(expectation, targets, atol=1e-5, rtol=0)


def test_hl_gauss_expectation_matches_target():
    dist = make_distribution(VALUE_DISTRIBUTION_HL_GAUSS)
    targets = torch.linspace(-9.0, 9.0, 37)
    expectation = (dist.target_probs(targets) * dist.bin_centers).sum(dim=-1)
    torch.testing.assert_close(expectation, targets, atol=1e-4, rtol=0)


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_out_of_support_targets_are_clamped_not_nan(mode):
    dist = make_distribution(mode)
    targets = torch.tensor([1e6, -1e6])
    probs = dist.target_probs(targets)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0)
    expectation = (probs * dist.bin_centers).sum(dim=-1)
    assert abs(expectation[0].item() - dist.target_max) < dist.bin_width
    assert abs(expectation[1].item() - dist.target_min) < dist.bin_width


def test_two_hot_edge_targets_stay_in_range():
    dist = make_distribution(VALUE_DISTRIBUTION_TWO_HOT)
    probs = dist.target_probs(torch.tensor([dist.target_min, dist.target_max]))
    # float32 rounding of the edge position leaks ~1e-5 into the neighbouring bin
    assert probs[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert probs[1, NUM_BINS - 1].item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "mode,expected_fraction",
    [(VALUE_DISTRIBUTION_TWO_HOT, 4 / 6), (VALUE_DISTRIBUTION_HL_GAUSS, 2 / 6)],
)
def test_clipped_fraction(mode, expected_fraction):
    dist = make_distribution(mode)
    # 9.95 sits inside [-10, 10] but outside [bin_centers[0], bin_centers[-1]].
    targets = torch.tensor([0.0, 5.0, 9.95, -9.95, 10.5, -10.5])
    assert dist.clipped_fraction(targets).item() == pytest.approx(expected_fraction)


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_expectation_of_uniform_logits_is_zero(mode):
    dist = make_distribution(mode)
    assert dist.expectation(torch.zeros(4, NUM_BINS)).abs().max().item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_expectation_upcasts_bfloat16_logits(mode):
    dist = make_distribution(mode)
    torch.manual_seed(0)
    logits = torch.randn(6, NUM_BINS)
    expectation = dist.expectation(logits.to(torch.bfloat16))
    assert expectation.dtype == torch.float32
    torch.testing.assert_close(expectation, dist.expectation(logits), atol=BFLOAT16_EXPECTATION_ATOL, rtol=0)


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_cross_entropy_minimum_is_target_entropy(mode):
    dist = make_distribution(mode)
    targets = torch.tensor([-1.18, 0.0, 3.3, 6.17])
    probs = dist.target_probs(targets)
    target_entropy = -torch.special.xlogy(probs, probs).sum(dim=-1).mean()
    at_target = dist.cross_entropy(probs.clamp_min(LOG_PROB_FLOOR).log(), targets)
    torch.testing.assert_close(at_target, target_entropy, atol=1e-6, rtol=0)

    at_uniform = dist.cross_entropy(torch.zeros(targets.shape[0], NUM_BINS), targets)
    assert at_uniform.item() == pytest.approx(math.log(NUM_BINS), abs=1e-5)
    assert at_uniform.item() > target_entropy.item()


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_cross_entropy_gradient_moves_expectation_toward_target(mode):
    dist = make_distribution(mode)
    target = torch.tensor([3.3])
    logits = torch.zeros(1, NUM_BINS, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.05)
    for _ in range(400):
        optimizer.zero_grad()
        dist.cross_entropy(logits, target).backward()
        optimizer.step()
    assert abs(dist.expectation(logits).item() - target.item()) < 0.05
