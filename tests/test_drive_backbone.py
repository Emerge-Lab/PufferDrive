"""Unit tests for DriveBackbone.forward with variable lane/boundary counts.

Verifies that the backbone correctly handles observations produced by a
training env (obs_dropout > 0 → fewer lane slots) and an eval env
(obs_dropout = 0 → full lane slots), which have different flat obs sizes.

The backbone uses max-pooling over the lane dimension, so its Linear layers
are unaffected by lane count. The bug tested here was that forward() used
the training-time self.obs_slots_lane_kept for slicing instead of inferring
from the actual obs size, causing misaligned slices for eval observations.
"""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch import DriveBackbone

LANE_N = 10
BOUNDARY_N = 10

BACKBONE_KWARGS = dict(
    input_size=32,
    backbone_hidden_size=64,
    backbone_num_layers=1,
    encoder_gigaflow=False,
    dropout=0.0,
)


def _make_env(lane_kept, boundary_kept, lane_n=LANE_N, boundary_n=BOUNDARY_N):
    """Minimal env-like namespace with all attributes DriveBackbone.__init__ reads."""
    return types.SimpleNamespace(
        obs_slots_lane_n=lane_n,
        obs_slots_boundary_n=boundary_n,
        obs_slots_lane_kept=lane_kept,
        obs_slots_boundary_kept=boundary_kept,
        obs_slots_partners_n=4,
        partner_features=binding.PARTNER_FEATURES,
        road_features=binding.ROAD_FEATURES,
        obs_slots_traffic_controls_n=2,
        traffic_control_features=binding.TRAFFIC_CONTROL_FEATURES,
        num_reward_coefs=0,
        target_dim=0,
    )


def _lane_start(env):
    return (
        binding.EGO_FEATURES
        + env.num_reward_coefs
        + env.target_dim
        + env.obs_slots_partners_n * env.partner_features
    )


def _tc_start(env):
    return (
        _lane_start(env)
        + env.obs_slots_lane_kept * env.road_features
        + env.obs_slots_boundary_kept * env.road_features
    )


def _obs_size(env):
    return _tc_start(env) + env.obs_slots_traffic_controls_n * env.traffic_control_features


def _make_obs(env, batch=2, seed=None):
    """Build a valid flat observation tensor.

    Lane/boundary slots are filled with random values; traffic-control type
    and state fields are set to valid enum indices (0) so F.one_hot doesn't
    assert out-of-bounds.
    """
    if seed is not None:
        torch.manual_seed(seed)
    obs = torch.randn(batch, _obs_size(env))

    tc_continuous = env.traffic_control_features - 2
    base = _tc_start(env)
    for slot in range(env.obs_slots_traffic_controls_n):
        slot_base = base + slot * env.traffic_control_features
        obs[:, slot_base + tc_continuous] = 0.0      # type  = NONE (0)
        obs[:, slot_base + tc_continuous + 1] = 0.0  # state = UNKNOWN (0)

    return obs


@pytest.fixture
def train_backbone():
    """Backbone built from a symmetric training env: dropout=0.4, kept=6/6."""
    return DriveBackbone(env=_make_env(6, 6), ego_dim=binding.EGO_FEATURES, **BACKBONE_KWARGS)


# ---------------------------------------------------------------------------
# Basic shape / no-crash tests
# ---------------------------------------------------------------------------


def test_forward_train_obs(train_backbone):
    """Backbone processes training observations (6 lane / 6 boundary slots)."""
    obs = _make_obs(_make_env(6, 6), seed=0)
    out = train_backbone(obs, binding.EGO_FEATURES)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_forward_eval_obs(train_backbone):
    """Same backbone processes eval observations (10 lane / 10 boundary slots).

    Regression: before the fix the traffic-control slice was shifted onto
    boundary data, causing F.one_hot to receive out-of-range indices.
    """
    obs = _make_obs(_make_env(10, 10), seed=0)
    out = train_backbone(obs, binding.EGO_FEATURES)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Asymmetric dropout  (lane_n=BOUNDARY_N=10, dropout_lane=0.5, dropout_boundary=0.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def asymmetric_backbone():
    """Backbone for dropout_lane=0.5 / dropout_boundary=0.4 on 10-slot envs."""
    # kept = int(10 * 0.5) = 5 lanes, int(10 * 0.4) = 6 boundaries  (wrong direction but
    # we just need asymmetry; actual formula rounds down)
    return DriveBackbone(env=_make_env(5, 6), ego_dim=binding.EGO_FEATURES, **BACKBONE_KWARGS)


def test_forward_asymmetric_train_obs(asymmetric_backbone):
    """Asymmetric-dropout backbone handles its own training obs (5 lane / 6 boundary)."""
    obs = _make_obs(_make_env(5, 6), seed=1)
    out = asymmetric_backbone(obs, binding.EGO_FEATURES)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_forward_asymmetric_eval_obs(asymmetric_backbone):
    """Asymmetric-dropout backbone handles eval obs (10 lane / 10 boundary, dropout=0).

    With the N-ratio split: road_slots=20, lane_n=boundary_n=10 → 10/10 ✓.
    """
    obs = _make_obs(_make_env(10, 10), seed=1)
    out = asymmetric_backbone(obs, binding.EGO_FEATURES)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Output values differ when lane content differs
# ---------------------------------------------------------------------------


def test_eval_obs_uses_extra_lanes(train_backbone):
    """Extra lanes in eval obs influence the output (max-pool sees more candidates)."""
    eval_env = _make_env(10, 10)
    lane_start = _lane_start(eval_env)
    lane_bytes = 10 * binding.ROAD_FEATURES

    # obs_a: all-zero lane slots
    obs_a = _make_obs(eval_env, seed=0)
    obs_a[:, lane_start : lane_start + lane_bytes] = 0.0

    # obs_b: slots 7-9 (beyond the 6 the old code would have read) set to large values
    obs_b = obs_a.clone()
    obs_b[:, lane_start + 6 * binding.ROAD_FEATURES : lane_start + lane_bytes] = 10.0

    out_a = train_backbone(obs_a, binding.EGO_FEATURES)
    out_b = train_backbone(obs_b, binding.EGO_FEATURES)
    assert not torch.equal(out_a, out_b), "extra eval lanes should affect the output"


# ---------------------------------------------------------------------------
# Out-of-range traffic-control type documents the pre-fix failure mode
# ---------------------------------------------------------------------------


def test_invalid_traffic_control_type_raises(train_backbone):
    """An out-of-range traffic-control type index triggers an error."""
    eval_env = _make_env(10, 10)
    obs = _make_obs(eval_env, seed=0)

    tc_continuous = eval_env.traffic_control_features - 2
    base = _tc_start(eval_env)
    obs[:, base + tc_continuous] = float(binding.NUM_TRAFFIC_CONTROL_TYPES)  # OOB

    with pytest.raises((RuntimeError, IndexError)):
        train_backbone(obs, binding.EGO_FEATURES)
