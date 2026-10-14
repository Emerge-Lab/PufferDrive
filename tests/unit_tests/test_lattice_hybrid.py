"""Hybrid lattice action: comfort-level speed cells (env.lattice_lon_urgency) and the structured head
(policy.lattice_structured_head) that scores the env's cells from factor logits and bell curves."""

import math
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import pufferlib
import pufferlib.pytorch
from pufferlib.config_schema import normalize_puffer_drive_config, validate_puffer_drive_config
from pufferlib.ocean.torch import (
    LATTICE_KIND_LANE_LEFT,
    LATTICE_KIND_LANE_RIGHT,
    LATTICE_KIND_OFFSET,
    LATTICE_KIND_ONCOMING,
    LATTICE_MASKED_LOGIT,
    LATTICE_MODE_BACKUP,
    LATTICE_MODE_CRUISE,
    LATTICE_MODE_EMERGENCY,
    LATTICE_MODE_STOP,
    LATTICE_MODE_STOP_LINE,
    LATTICE_MODE_TURN,
    LATTICE_SPECIAL_CELL_LOGIT_BIAS,
    LatticeStructuredHead,
)
from pufferlib.pufferl import load_config

HYBRID_SPEEDS = [float(speed) for speed in range(21)]
HYBRID_OFFSETS = [-0.9, -0.45, 0.0, 0.45, 0.9]
HYBRID_STOPS = [3.0, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0]


def _menu_env(oncoming, turnaround, urgency, offsets=HYBRID_OFFSETS, speeds=HYBRID_SPEEDS, stops=HYBRID_STOPS):
    """Stand-in for drive.Drive's lattice attributes; nvec mirrors drive.py / lattice.h init_lattice_config."""
    lat_durations, lon_durations, fracs, backups = (
        [2.4, 3.6, 4.8, 6.0],
        [1.2, 2.4, 3.6, 4.8, 6.0],
        [0.4, 0.7, 1.0],
        [1.0, 2.0, 5.0, 10.0],
    )
    slot_count = len(fracs) if urgency else len(lon_durations)
    nvec = [
        2,
        (len(offsets) + 2 + int(oncoming)) * len(lat_durations),
        2,
        len(speeds) * slot_count + len(stops) + 2 + len(backups) + int(turnaround),
        5,
    ]
    return SimpleNamespace(
        lattice_lat_offsets_m=offsets,
        lattice_lat_durations_s=lat_durations,
        lattice_lon_speeds_mps=speeds,
        lattice_lon_durations_s=lon_durations,
        lattice_lon_urgency=urgency,
        lattice_lon_urgency_fracs=fracs,
        lattice_lon_slot_count=slot_count,
        lattice_stop_distances_m=stops,
        lattice_backup_distances_m=backups,
        lattice_oncoming_overtake=oncoming,
        lattice_turnaround=turnaround,
        lattice_nvec=nvec,
    )


MENUS = [
    _menu_env(oncoming, turnaround, urgency)
    for oncoming in (False, True)
    for turnaround in (False, True)
    for urgency in (False, True)
] + [
    _menu_env(
        True,
        True,
        False,
        offsets=[-0.9, 0.0, 0.9],
        speeds=[0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0],
        stops=[5.0, 10.0, 20.0, 40.0],
    )
]


def _groups(head, cells):
    nvec = head.nvec
    return cells[:, nvec[0] : nvec[0] + nvec[1]], cells[:, sum(nvec[:3]) : sum(nvec[:4])]


def _random_masks(env, batch, seed, keep=0.5):
    """Random per-cell masks with index 0 of every factor offered (lattice_mask_index0_only)."""
    gen = torch.Generator().manual_seed(seed)
    masks = (torch.rand(batch, sum(env.lattice_nvec), generator=gen) < keep).float()
    offset = 0
    for count in env.lattice_nvec:
        masks[:, offset] = 1.0
        offset += count
    return masks


def _all_allowed(env, batch=1):
    return torch.ones(batch, sum(env.lattice_nvec))


@pytest.mark.parametrize("env", MENUS)
@pytest.mark.parametrize("keep", [1.0, 0.5, 0.1])
def test_each_group_is_an_exact_distribution_over_the_allowed_cells(env, keep):
    head = LatticeStructuredHead(env)
    params = 3.0 * torch.randn(64, head.param_count, generator=torch.Generator().manual_seed(0))
    masks = _random_masks(env, 64, seed=1, keep=keep)
    cells = head(params, masks)
    assert cells.shape == (64, sum(env.lattice_nvec))
    probs = cells.masked_fill(masks < 0.5, LATTICE_MASKED_LOGIT).exp()
    lat, lon = _groups(head, probs)
    assert torch.allclose(lat.sum(1), torch.ones(64), atol=1e-5)
    assert torch.allclose(lon.sum(1), torch.ones(64), atol=1e-5)
    assert torch.isfinite(cells).all()


def _part_start(head, name):
    return sum(size for part, size in head.layout[: [part for part, _ in head.layout].index(name)])


def test_cell_maps_follow_the_c_layout():
    """Steering one factor makes the cell that lattice.h decodes to that meaning the most likely one."""
    env = _menu_env(oncoming=True, turnaround=True, urgency=True)
    head = LatticeStructuredHead(env)
    assert head.choice_kind.tolist() == [LATTICE_KIND_LANE_RIGHT] + [LATTICE_KIND_OFFSET] * 5 + [
        LATTICE_KIND_LANE_LEFT,
        LATTICE_KIND_ONCOMING,
    ]
    nvec = env.lattice_nvec
    lat_start, lon_start = nvec[0], sum(nvec[:3])

    def best_cell(edits, start, size):
        params = torch.zeros(1, head.param_count)
        for name, index, value in edits:
            params[0, _part_start(head, name) + index] = value
        return head(params, _all_allowed(env))[0, start : start + size].argmax().item()

    choice_count = 8  # lane right, 5 offsets, lane left, oncoming: cell = 8 * duration + choice
    assert (
        best_cell(
            [
                ("lat_kind", LATTICE_KIND_OFFSET, 9.0),
                ("offset_bell", 0, 9.0),
                ("offset_bell", 1, -9.0),
                ("lat_duration", 2, 9.0),
            ],
            lat_start,
            nvec[1],
        )
        == 8 * 2 + 5
    )
    assert (
        best_cell([("lat_kind", LATTICE_KIND_LANE_LEFT, 9.0), ("lat_duration", 1, 9.0)], lat_start, nvec[1])
        == choice_count * 1 + 6
    )
    assert (
        best_cell([("lat_kind", LATTICE_KIND_LANE_RIGHT, 9.0), ("lat_duration", 3, 9.0)], lat_start, nvec[1])
        == choice_count * 3 + 0
    )
    assert best_cell([("lat_kind", LATTICE_KIND_ONCOMING, 9.0)], lat_start, nvec[1]) % choice_count == 7
    centre_8 = math.log((8.0 + 2.0) / (24.0 - (8.0 + 2.0)))  # sigmoid^-1 over the centre range [-2, 22]
    assert (
        best_cell([("speed_bell", 0, centre_8), ("speed_bell", 1, -9.0), ("lon_slot", 2, 9.0)], lon_start, nvec[3])
        == 2 * 21 + 8
    )
    centre_ln10 = math.log(10.0)
    low, high = head.stop_bell.centre_low, head.stop_bell.centre_high
    stop_raw = math.log((centre_ln10 - low) / (high - centre_ln10))
    assert (
        best_cell(
            [("lon_mode", LATTICE_MODE_STOP, 20.0), ("stop_bell", 0, stop_raw), ("stop_bell", 1, -9.0)],
            lon_start,
            nvec[3],
        )
        == 63 + 3
    )
    assert best_cell([("lon_mode", LATTICE_MODE_STOP_LINE, 20.0)], lon_start, nvec[3]) == 63 + 8
    assert best_cell([("lon_mode", LATTICE_MODE_EMERGENCY, 20.0)], lon_start, nvec[3]) == 63 + 9
    assert best_cell([("lon_mode", LATTICE_MODE_BACKUP, 20.0), ("backup", 2, 9.0)], lon_start, nvec[3]) == 63 + 10 + 2
    assert best_cell([("lon_mode", LATTICE_MODE_TURN, 20.0)], lon_start, nvec[3]) == nvec[3] - 1


def test_masked_choices_never_shift_mass_between_modes():
    """Mode probabilities depend only on which modes have an allowed cell, never on where the speed bell sits."""
    env = _menu_env(oncoming=False, turnaround=True, urgency=True)
    head = LatticeStructuredHead(env)
    nvec, lon_start = env.lattice_nvec, sum(env.lattice_nvec[:3])
    params = torch.zeros(2, head.param_count)
    speed_bell = _part_start(head, "speed_bell")
    params[0, speed_bell] = 8.0  # centre near 22 m/s
    params[0, speed_bell + 1] = -8.0  # narrowest spread
    params[1, speed_bell] = -8.0  # centre near -2 m/s
    masks = _all_allowed(env, 2)
    for slot in range(3):
        masks[:, lon_start + slot * 21 + 8 : lon_start + slot * 21 + 21] = 0.0  # speeds above 7 m/s masked
    lon = head(params, masks)[:, lon_start : lon_start + nvec[3]].exp() * masks[:, lon_start : lon_start + nvec[3]]
    cruise = lon[:, :63].sum(1)
    assert cruise[0].item() == pytest.approx(cruise[1].item(), rel=1e-5)
    mode_probs = torch.softmax(head.split(params[:1])["lon_mode"], 1)
    assert cruise[0].item() == pytest.approx(mode_probs[0, LATTICE_MODE_CRUISE].item(), rel=1e-5)
    fastest_allowed = lon[0, :63].view(3, 21).sum(0)
    assert fastest_allowed.argmax().item() == 7  # the bell's preference lands on the fastest allowed speed


@pytest.mark.parametrize("env", MENUS)
def test_initial_mode_and_kind_marginals_match_the_linear_head(env):
    head = LatticeStructuredHead(env)
    cells = head(torch.zeros(1, head.param_count), _all_allowed(env))
    lat, lon = _groups(head, cells)
    nvec = env.lattice_nvec
    speed_cells = len(env.lattice_lon_speeds_mps) * env.lattice_lon_slot_count
    linear = torch.zeros(nvec[3])
    linear[speed_cells:] = LATTICE_SPECIAL_CELL_LOGIT_BIAS
    linear_probs = torch.softmax(linear, 0)
    lon_modes = [LATTICE_MODE_CRUISE] * speed_cells + [LATTICE_MODE_STOP] * len(env.lattice_stop_distances_m)
    lon_modes += (
        [LATTICE_MODE_STOP_LINE, LATTICE_MODE_EMERGENCY]
        + [LATTICE_MODE_BACKUP] * 4
        + [LATTICE_MODE_TURN] * int(env.lattice_turnaround)
    )
    lon_modes = torch.tensor(lon_modes)
    for mode in lon_modes.unique().tolist():
        in_mode = lon_modes == mode
        assert lon[0, in_mode].exp().sum().item() == pytest.approx(linear_probs[in_mode].sum().item(), rel=1e-5)
    lat_kinds = head.choice_kind.repeat(len(env.lattice_lat_durations_s))
    for kind in lat_kinds.unique().tolist():
        in_kind = lat_kinds == kind
        assert lat[0, in_kind].exp().sum().item() == pytest.approx(in_kind.float().mean().item(), rel=1e-5)
    assert torch.equal(cells[0, : nvec[0]], torch.zeros(nvec[0]))


def test_initial_speed_marginal_is_close_to_the_linear_head():
    env = _menu_env(oncoming=True, turnaround=True, urgency=True)
    head = LatticeStructuredHead(env)
    cells = head(torch.zeros(1, head.param_count), _all_allowed(env))
    lon_start = sum(env.lattice_nvec[:3])
    speed_probs = cells[0, lon_start : lon_start + 63].exp().view(3, 21).sum(0)
    speed_probs = speed_probs / speed_probs.sum()
    speeds = torch.tensor(HYBRID_SPEEDS)
    mean = (speed_probs * speeds).sum().item()
    assert mean == pytest.approx(10.0, abs=0.05)  # the base head starts uniform over its speeds (mean 9.1 m/s)
    assert speed_probs[speeds >= 10.0].sum().item() > 0.5
    assert speed_probs.min().item() > 0.4 * speed_probs.max().item()


def test_bell_bounds_and_their_gradients():
    env = _menu_env(oncoming=True, turnaround=True, urgency=True)
    head = LatticeStructuredHead(env)
    bell = head.speed_bell
    for raw in (-30.0, 30.0):
        centre, sigma = bell.centre_and_spread(torch.tensor([raw]), torch.tensor([raw]))
        assert -2.0 - 1e-4 <= centre.item() <= 22.0 + 1e-4
        assert 0.35 - 1e-5 <= sigma.item() <= 40.0 + 1e-3
    spread_raw = torch.tensor([-6.0], requires_grad=True)  # sigma close to its floor
    log_probs = torch.log_softmax(bell.logits(head.speed_values, torch.tensor([0.0]), spread_raw), 1)
    entropy = -(log_probs.exp() * log_probs).sum()
    entropy.backward()
    assert spread_raw.grad.item() > 0.0  # the entropy bonus can still widen a collapsed bell
    assert math.exp(head.offset_bell.log_sigma_min) == pytest.approx(0.35 * 0.45)
    assert math.exp(head.stop_bell.log_sigma_max) == pytest.approx(2.0 * math.log(40.0 / 3.0))


def test_masked_cells_get_zero_probability_and_sampling_stays_valid():
    env = _menu_env(oncoming=True, turnaround=True, urgency=True)
    head = LatticeStructuredHead(env)
    params = torch.randn(256, head.param_count, generator=torch.Generator().manual_seed(3))
    masks = _random_masks(env, 256, seed=4)
    masked = head(params, masks).masked_fill(masks < 0.5, LATTICE_MASKED_LOGIT)
    logits = pufferlib.pytorch.LatticeLogits(*torch.split(masked, env.lattice_nvec, dim=1))
    action, logprob, entropy, _ = pufferlib.pytorch.sample_lattice_logits(logits)
    assert torch.isfinite(logprob).all() and torch.isfinite(entropy).all()
    offset = 0
    for factor, count in enumerate(env.lattice_nvec):
        assert (masks[torch.arange(256), offset + action[:, factor]] > 0.5).all()
        offset += count


def test_rewarding_a_faster_speed_moves_the_speed_centre_up():
    env = _menu_env(oncoming=False, turnaround=False, urgency=True)
    head = LatticeStructuredHead(env)
    params = torch.zeros(1, head.param_count, requires_grad=True)
    lon_start = sum(env.lattice_nvec[:3])
    cells = head(params, _all_allowed(env))
    cells[0, lon_start + 2 * 21 + 14].backward()  # 14 m/s at the firm level
    assert (
        params.grad[0, _part_start(head, "speed_bell")].item() > 0.0
    )  # the centre (10 m/s at init) moves toward 14 m/s


def test_structured_head_rejects_unordered_menus():
    with pytest.raises(ValueError):
        LatticeStructuredHead(_menu_env(False, False, True, speeds=[0.0, 5.0, 2.5]))
    with pytest.raises(ValueError):
        LatticeStructuredHead(_menu_env(False, False, True, stops=[10.0, 5.0]))


def _args(config_name):
    with patch.object(sys, "argv", ["pufferl.py"]):
        return load_config(config_name)


def _validate(args):
    normalized = normalize_puffer_drive_config(args, "test")
    validate_puffer_drive_config(normalized, "test")
    return normalized


def test_hybrid_config_loads_with_its_menus():
    args = _validate(_args("puffer_drive_spline_werling_hybrid"))
    env = args["env"]
    assert env["lattice_lon_urgency"] is True and env["lattice_lon_urgency_fracs"] == [0.4, 0.7, 1.0]
    assert env["lattice_lon_speeds_mps"] == HYBRID_SPEEDS
    assert env["lattice_lat_offsets_m"] == HYBRID_OFFSETS and env["lattice_stop_distances_m"] == HYBRID_STOPS
    assert args["policy"]["lattice_structured_head"] is True
    base = _validate(_args("puffer_drive_spline_werling"))
    assert base["env"]["lattice_lon_urgency"] is False and base["policy"]["lattice_structured_head"] is False


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("env", "lattice_lon_urgency_fracs", [0.0, 0.5, 1.0]),
        ("env", "lattice_lon_urgency_fracs", [0.4, 0.7, 1.5]),
        ("env", "lattice_lon_urgency_fracs", [0.7, 0.4, 1.0]),
        ("env", "lattice_lon_urgency_fracs", [0.1] * 7),
        ("env", "lattice_lon_speeds_mps", [0.0, 5.0, 2.0]),
        ("env", "lattice_stop_distances_m", [10.0, 5.0]),
    ],
)
def test_invalid_hybrid_settings_are_rejected(section, key, value):
    args = _args("puffer_drive_spline_werling_hybrid")
    args[section][key] = value
    with pytest.raises(pufferlib.APIUsageError):
        _validate(args)


def test_structured_head_needs_the_lattice():
    args = _args("puffer_drive")
    args["policy"]["lattice_structured_head"] = True
    with pytest.raises(pufferlib.APIUsageError):
        _validate(args)
