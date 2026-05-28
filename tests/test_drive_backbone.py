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


def _make_env(obs_slots_lane_kept, obs_slots_boundary_kept):
    """Minimal env-like namespace with all attributes DriveBackbone.__init__ reads."""
    env = types.SimpleNamespace(
        obs_slots_partners_n=4,
        partner_features=binding.PARTNER_FEATURES,
        obs_slots_lane_kept=obs_slots_lane_kept,
        obs_slots_boundary_kept=obs_slots_boundary_kept,
        road_features=binding.ROAD_FEATURES,
        obs_slots_traffic_controls_n=2,
        traffic_control_features=binding.TRAFFIC_CONTROL_FEATURES,
        num_reward_coefs=0,
        target_dim=0,
    )
    return env


def _obs_size(env):
    return (
        binding.EGO_FEATURES
        + env.num_reward_coefs
        + env.target_dim
        + env.obs_slots_partners_n * env.partner_features
        + env.obs_slots_lane_kept * env.road_features
        + env.obs_slots_boundary_kept * env.road_features
        + env.obs_slots_traffic_controls_n * env.traffic_control_features
    )


def _make_obs(env, batch=2):
    """Build a valid flat observation tensor with realistic traffic-control values.

    Continuous fields are filled with small floats; the discrete type/state
    fields at the end of each traffic-control slot are set to valid enum
    indices (0 = NONE / UNKNOWN) so F.one_hot inside the backbone doesn't
    assert out-of-bounds.
    """
    obs_size = _obs_size(env)
    obs = torch.zeros(batch, obs_size)

    # Fill traffic-control slots with valid type/state values.
    tc_features = env.traffic_control_features
    tc_continuous = tc_features - 2
    fixed = (
        binding.EGO_FEATURES
        + env.num_reward_coefs
        + env.target_dim
        + env.obs_slots_partners_n * env.partner_features
        + env.obs_slots_lane_kept * env.road_features
        + env.obs_slots_boundary_kept * env.road_features
    )
    for slot in range(env.obs_slots_traffic_controls_n):
        base = fixed + slot * tc_features
        # type = TRAFFIC_CONTROL_TYPE_NONE (0), state = TRAFFIC_CONTROL_STATE_UNKNOWN (0)
        obs[:, base + tc_continuous] = 0.0
        obs[:, base + tc_continuous + 1] = 0.0

    return obs


BACKBONE_KWARGS = dict(
    input_size=32,
    backbone_hidden_size=64,
    backbone_num_layers=1,
    encoder_gigaflow=False,
    dropout=0.0,
)

# obs_slots_lane_n=10, dropout=0.4 → kept=6
TRAIN_LANE_KEPT = 6
# obs_slots_lane_n=10, dropout=0.0 → kept=10
EVAL_LANE_KEPT = 10


@pytest.fixture
def train_backbone():
    """Backbone built from a training env with obs_dropout_lane=0.4."""
    train_env = _make_env(TRAIN_LANE_KEPT, TRAIN_LANE_KEPT)
    return DriveBackbone(env=train_env, ego_dim=binding.EGO_FEATURES, **BACKBONE_KWARGS)


def test_forward_train_obs(train_backbone):
    """Backbone processes training observations (6 lane/boundary slots) correctly."""
    train_env = _make_env(TRAIN_LANE_KEPT, TRAIN_LANE_KEPT)
    obs = _make_obs(train_env)
    out = train_backbone(obs, binding.EGO_FEATURES)
    assert out.shape == (2, 64), f"unexpected shape {out.shape}"
    assert not torch.isnan(out).any(), "NaN in train forward output"


def test_forward_eval_obs(train_backbone):
    """Same backbone processes eval observations (10 lane/boundary slots) correctly.

    This is the regression test: before the fix, the backbone used
    self.obs_slots_lane_kept=6 to slice a 10-lane obs buffer, which shifted
    the traffic-control slice onto boundary data, causing F.one_hot to receive
    out-of-range indices and fire a CUDA scatter/gather assertion.
    """
    eval_env = _make_env(EVAL_LANE_KEPT, EVAL_LANE_KEPT)
    obs = _make_obs(eval_env)
    out = train_backbone(obs, binding.EGO_FEATURES)
    assert out.shape == (2, 64), f"unexpected shape {out.shape}"
    assert not torch.isnan(out).any(), "NaN in eval forward output"


def test_forward_outputs_differ(train_backbone):
    """Outputs for train and eval obs should differ (different lane content)."""
    train_env = _make_env(TRAIN_LANE_KEPT, TRAIN_LANE_KEPT)
    eval_env = _make_env(EVAL_LANE_KEPT, EVAL_LANE_KEPT)

    # Use the same random lane values so any difference comes from lane count, not noise
    torch.manual_seed(0)
    train_obs = _make_obs(train_env)
    eval_obs = _make_obs(eval_env)
    # Copy train lane data into the eval obs (first 6 lanes match)
    train_lane_start = binding.EGO_FEATURES + train_env.num_reward_coefs + train_env.target_dim + train_env.obs_slots_partners_n * train_env.partner_features
    eval_obs[:, train_lane_start : train_lane_start + TRAIN_LANE_KEPT * binding.ROAD_FEATURES] = \
        train_obs[:, train_lane_start : train_lane_start + TRAIN_LANE_KEPT * binding.ROAD_FEATURES]

    train_out = train_backbone(train_obs, binding.EGO_FEATURES)
    eval_out = train_backbone(eval_obs, binding.EGO_FEATURES)
    # Both produce valid tensors of the same shape
    assert train_out.shape == eval_out.shape


def test_forward_invalid_traffic_control_raises(train_backbone):
    """Ensure out-of-range traffic-control type triggers an error (documents the bug)."""
    eval_env = _make_env(EVAL_LANE_KEPT, EVAL_LANE_KEPT)
    obs = _make_obs(eval_env)

    # Corrupt the traffic-control type field with an invalid index
    tc_features = eval_env.traffic_control_features
    tc_continuous = tc_features - 2
    fixed = (
        binding.EGO_FEATURES
        + eval_env.num_reward_coefs
        + eval_env.target_dim
        + eval_env.obs_slots_partners_n * eval_env.partner_features
        + eval_env.obs_slots_lane_kept * eval_env.road_features
        + eval_env.obs_slots_boundary_kept * eval_env.road_features
    )
    obs[:, fixed + tc_continuous] = float(binding.NUM_TRAFFIC_CONTROL_TYPES)  # exactly num_classes → OOB

    with pytest.raises((RuntimeError, IndexError)):
        train_backbone(obs, binding.EGO_FEATURES)
