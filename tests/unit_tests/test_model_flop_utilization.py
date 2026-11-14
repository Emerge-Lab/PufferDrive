"""Model FLOP utilization: the policy's analytic FLOP count, the measured peak and the pooling of per-rank windows."""

import math
from types import SimpleNamespace

import pytest
import torch

import pufferlib.ocean.torch as ocean_torch
from pufferlib import pufferl
from pufferlib.pufferl import PuffeRL


EGO_FEATURES = 4
LANE_SLOTS = 3
LANE_FEATURES = 5
PARTNER_SLOTS = 2
PARTNER_FEATURES = 6
ENCODER_WIDTH = 8
BACKBONE_WIDTH = 16
HEAD_WIDTH = 8


def matmul_flops(in_features, out_features):
    return 2.0 * in_features * out_features


def test_policy_flops_count_every_slot_and_skip_observation_gradients():
    env = SimpleNamespace(
        ego_features=EGO_FEATURES,
        obs_slots_lane_kept=LANE_SLOTS,
        lane_features=LANE_FEATURES,
        obs_slots_partners_n=PARTNER_SLOTS,
        partner_features=PARTNER_FEATURES,
        obs_slots_boundary_kept=0,
        boundary_features=5,
        obs_slots_traffic_controls_n=0,
        traffic_control_features=4,
        num_reward_coefs=0,
        goal_dim=0,
    )
    policy = ocean_torch.Drive(
        env,
        ego_input_size=ENCODER_WIDTH,
        partner_input_size=ENCODER_WIDTH,
        lane_input_size=ENCODER_WIDTH,
        boundary_input_size=ENCODER_WIDTH,
        traffic_control_input_size=ENCODER_WIDTH,
        context_input_size=ENCODER_WIDTH,
        backbone_hidden_size=BACKBONE_WIDTH,
        backbone_num_layers=1,
        actor_hidden_size=HEAD_WIDTH,
        actor_num_layers=1,
        critic_hidden_size=HEAD_WIDTH,
        critic_num_layers=1,
        encoder_activation="relu",
        encoder_layer_norm=True,
        backbone_activation="gelu",
        backbone_layer_norm=False,
        action_type="discrete",
    )

    first_encoder_layers = (
        matmul_flops(EGO_FEATURES, ENCODER_WIDTH)
        + LANE_SLOTS * matmul_flops(LANE_FEATURES, ENCODER_WIDTH)
        + PARTNER_SLOTS * matmul_flops(PARTNER_FEATURES, ENCODER_WIDTH)
    )
    second_encoder_layers = (1 + LANE_SLOTS + PARTNER_SLOTS) * matmul_flops(ENCODER_WIDTH, ENCODER_WIDTH)
    backbone_layer = matmul_flops(3 * ENCODER_WIDTH, BACKBONE_WIDTH)
    heads = (
        matmul_flops(BACKBONE_WIDTH, HEAD_WIDTH)
        + matmul_flops(HEAD_WIDTH, policy.action_dim)
        + matmul_flops(BACKBONE_WIDTH, HEAD_WIDTH)
        + matmul_flops(HEAD_WIDTH, 1)
    )
    # actor and critic each own a backbone; only layers behind an encoder's first one pay for an input gradient
    forward_flops = 2 * (first_encoder_layers + second_encoder_layers + backbone_layer) + heads
    backward_flops = 2 * (first_encoder_layers + 2 * (second_encoder_layers + backbone_layer)) + 2 * heads

    assert policy.matmul_flops_per_transition() == (forward_flops, forward_flops + backward_flops)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to run the tensor-core kernel")
def test_tensor_core_peak_is_measured_without_touching_the_training_rng():
    from pufferlib.tensor_core_peak import measure_tensor_core_peak_flops_per_second

    torch.manual_seed(7)
    rng_state_before = torch.cuda.get_rng_state()

    peak_flops_per_second = measure_tensor_core_peak_flops_per_second("cuda")

    assert math.isfinite(peak_flops_per_second)
    assert peak_flops_per_second > 0.0
    assert torch.equal(rng_state_before, torch.cuda.get_rng_state())


def test_flop_windows_pool_over_ranks_by_peak_rate(monkeypatch):
    trainer = PuffeRL.__new__(PuffeRL)
    trainer.peak_flops_per_second = 100.0
    trainer.forward_flops_per_transition = 2.0
    trainer.update_flops_per_transition = 6.0
    trainer.last_log_step = 10
    trainer.global_step = 30
    trainer.trained_transition_count = 5
    trainer.rollout_seconds = 1.0
    trainer.update_seconds = 0.5
    trainer.last_log_time = 1000.0
    monkeypatch.setattr(pufferl.time, "time", lambda: 1002.0)

    window = trainer.flop_window()

    assert window == {"MFU": (70.0, 200.0), "MFU_rollout": (40.0, 100.0), "MFU_train": (30.0, 50.0)}
    # a second rank that did the same work on a GPU that spent longer in its rollout
    slower_rank_window = {"MFU": (70.0, 200.0), "MFU_rollout": (40.0, 150.0), "MFU_train": (30.0, 50.0)}
    assert pufferl.pooled_flop_utilization_percent([window, slower_rank_window]) == {
        "MFU": 35.0,
        "MFU_rollout": 32.0,
        "MFU_train": 60.0,
    }


def test_no_flop_window_without_a_gpu_peak_or_new_agent_steps():
    trainer = PuffeRL.__new__(PuffeRL)
    trainer.peak_flops_per_second = None
    assert trainer.flop_window() == {}

    trainer.peak_flops_per_second = 100.0
    trainer.global_step = 30
    trainer.last_log_step = 30
    assert trainer.flop_window() == {}
    assert pufferl.pooled_flop_utilization_percent([{}]) == {}
