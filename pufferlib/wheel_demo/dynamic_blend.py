"""Dynamic blend: assist as needed. The Blend share follows how the driver has been doing.

Blend steers toward the policy's smoothed look-ahead angle (the mean over the look-ahead window, not its step at this
instant). For the first DYNAMIC_WARMUP_SECONDS it gives no help: the policy's own imagined drive over that window is
scored every step to give the target. After that, every policy step:

    gap      = clip((target - R_last30s) / DYNAMIC_GAP_SCALE, 0, 1)       target = baseline - allowed off-road margin
    disagree = clip(mean |wheel - policy angle| over the last few s / DYNAMIC_DISAGREE_FULL_RAD, 0, 1)
    desired  = share_min + (share_max - share_min) * max(gap, disagree if DYNAMIC_USE_DISAGREEMENT else 0)
    share   += clip(desired - share, -DOWN_RATE * dt, +UP_RATE * dt)

The score is the reward without its goal part, for the target as well as for R_last30s. Help comes quickly and is
withdrawn slowly, so doing well with assist does not immediately take the assist away.
"""

import math
from collections import deque

import numpy as np

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo.coadapt import human_step_score

DYNAMIC_WARMUP_SECONDS = 30.0
# Fewer clean look-ahead scores than this over the warm-up and the baseline is not trusted.
DYNAMIC_BASELINE_MIN_STEPS = 50
DYNAMIC_BASELINE_FALLBACK = -0.005
DYNAMIC_RECENT_SECONDS = 30.0
# Before this much driving the 30 s average says too little; only the disagreement acts.
DYNAMIC_RECENT_MIN_SECONDS = 5.0
DYNAMIC_DISAGREE_SECONDS = 3.0
# Each step off the road costs about this; the allowed fraction of time off the road lowers the target by it.
DYNAMIC_OFFROAD_STEP_PENALTY = 1.2
DYNAMIC_ALLOWED_OFFROAD_FRACTION = 0.0
# Falling this far below the target (per step) asks for the most help; 0.24 below asks for 80%.
DYNAMIC_GAP_SCALE = 0.30
# A gap this large between the wheel and the policy's angle (90 of 450 degrees of wheel) asks for the most help.
DYNAMIC_DISAGREE_FULL_RAD = 90.0 / 450.0 * binding.HUMAN_STEERING_LIMIT_RAD
# Off: only the reward gap asks for help; the disagreement is still measured and shown.
DYNAMIC_USE_DISAGREEMENT = False
DYNAMIC_SHARE_MIN = 0.0
DYNAMIC_SHARE_MAX = 1.0
DYNAMIC_START_SHARE = 0.0
DYNAMIC_UP_RATE_PER_S = 0.10
DYNAMIC_DOWN_RATE_PER_S = 0.02


class DynamicBlend:
    """Call after_step once per policy step while active; it sets the env's Blend share itself."""

    def __init__(self, loop):
        self.loop = loop
        self.dt_s = loop.dt
        self.warmup_steps = max(1, round(DYNAMIC_WARMUP_SECONDS / self.dt_s))
        self.recent_min_steps = max(1, round(DYNAMIC_RECENT_MIN_SECONDS / self.dt_s))
        self._recent_scores = deque(maxlen=max(1, round(DYNAMIC_RECENT_SECONDS / self.dt_s)))
        self._recent_disagreement_rad = deque(maxlen=max(1, round(DYNAMIC_DISAGREE_SECONDS / self.dt_s)))
        self.active = False
        self.baseline = None  # measured once per session; kept across restarts
        self.share = DYNAMIC_START_SHARE
        self.gap = 0.0
        self.disagree = 0.0
        self._warmup_step_count = 0
        self._policy_clean_scores = []

    @property
    def measuring_baseline(self):
        return self.active and self.baseline is None

    @property
    def target(self):
        if self.baseline is None:
            return None
        return self.baseline - DYNAMIC_OFFROAD_STEP_PENALTY * DYNAMIC_ALLOWED_OFFROAD_FRACTION

    @property
    def recent_score(self):
        return float(np.mean(self._recent_scores)) if self._recent_scores else None

    @property
    def baseline_seconds_left(self):
        return (self.warmup_steps - self._warmup_step_count) * self.dt_s

    def start(self):
        self.active = True
        self._recent_scores.clear()
        self._recent_disagreement_rad.clear()
        self.share = DYNAMIC_START_SHARE
        self.gap = 0.0
        self.disagree = 0.0
        if self.baseline is None:
            self._warmup_step_count = 0
            self._policy_clean_scores = []
        self.loop.set_blend_follows_lookahead(True)
        self.loop.env.set_assist_mode(binding.ASSIST_MODE_BLEND, self.share)

    def stop(self):
        """Leaving drops a baseline measurement in progress; a measured baseline is kept for next time."""
        self.active = False
        self.loop.set_blend_follows_lookahead(False)

    def restart(self):
        """A restarted drive keeps the baseline but not the recent windows, which belong to the old drive."""
        if self.active:
            self.start()

    def after_step(self):
        if not self.active:
            return
        assist = self.loop.env.get_assist_state()
        self._recent_scores.append(human_step_score(self.loop))
        if assist["reference_valid"]:
            self._recent_disagreement_rad.append(
                abs(assist["human_steering_rad"] - assist["blend_reference_steering_rad"])
            )
        if self.baseline is None:
            self._measure_baseline()
        self._update_share()

    def _measure_baseline(self):
        policy_score = self.loop.lookahead_policy_score()
        if policy_score is not None and policy_score[1]:
            self._policy_clean_scores.append(policy_score[0])
        self._warmup_step_count += 1
        if self._warmup_step_count < self.warmup_steps:
            return
        trusted = len(self._policy_clean_scores) >= DYNAMIC_BASELINE_MIN_STEPS
        self.baseline = float(np.mean(self._policy_clean_scores)) if trusted else DYNAMIC_BASELINE_FALLBACK

    def _update_share(self):
        if self.baseline is not None and len(self._recent_scores) >= self.recent_min_steps:
            self.gap = float(np.clip((self.target - self.recent_score) / DYNAMIC_GAP_SCALE, 0.0, 1.0))
        else:
            self.gap = 0.0
        if self._recent_disagreement_rad:
            mean_rad = float(np.mean(self._recent_disagreement_rad))
            self.disagree = float(np.clip(mean_rad / DYNAMIC_DISAGREE_FULL_RAD, 0.0, 1.0))
        else:
            self.disagree = 0.0
        need = max(self.gap, self.disagree) if DYNAMIC_USE_DISAGREEMENT else self.gap
        desired = DYNAMIC_SHARE_MIN + (DYNAMIC_SHARE_MAX - DYNAMIC_SHARE_MIN) * need
        change = float(
            np.clip(desired - self.share, -DYNAMIC_DOWN_RATE_PER_S * self.dt_s, DYNAMIC_UP_RATE_PER_S * self.dt_s)
        )
        self.share = float(np.clip(self.share + change, DYNAMIC_SHARE_MIN, DYNAMIC_SHARE_MAX))
        self.loop.env.set_assist_mode(binding.ASSIST_MODE_BLEND, self.share)

    def hud_line(self):
        if not self.active:
            return ""
        if self.baseline is None:
            return (
                f"{round(100 * self.share)}% policy (no help yet: learning the target, "
                f"{math.ceil(self.baseline_seconds_left)} s left)"
            )
        recent = "-" if self.recent_score is None else f"{self.recent_score:+.3f}"
        return (
            f"{round(100 * self.share)}% policy (recent {recent} vs target {self.target:+.3f}, "
            f"gap {round(100 * self.gap)}%, disagree {round(100 * self.disagree)}%"
            f"{'' if DYNAMIC_USE_DISAGREEMENT else ' (off)'})"
        )
