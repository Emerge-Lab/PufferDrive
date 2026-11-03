"""Co-adapt: Blend whose policy share is learned for the driver, one trial at a time.

Each trial drives for COADAPT_TRIAL_SECONDS at one share. Its score is the human car's return over the trial
without the goal reward, so a trial is not decided by whether a goal happened to fall inside it. A UCB1 bandit
picks the next share from the scores so far.
"""

import math

from pufferlib.ocean.drive import binding

COADAPT_SHARES = (0.2, 0.4, 0.6, 0.8)
COADAPT_TRIAL_SECONDS = 20.0


def human_step_score(loop):
    """The human car's reward on the latest policy step without its goal part: driving quality, not where goals fall."""
    return float(loop.last_rewards[loop.human_agent_idx]) - loop.human_state()["step_goal_reward"]


class UcbBandit:
    """UCB1 over a fixed set of arms. Scores have no known range, so means are rescaled to [0, 1] by the lowest
    and highest trial score seen so far before the exploration bonus sqrt(2 ln N / n) is added."""

    def __init__(self, arm_count):
        self.counts = [0] * arm_count
        self.score_sums = [0.0] * arm_count
        self.lowest_score = math.inf
        self.highest_score = -math.inf
        self.score_square_sum = 0.0

    def select(self):
        """Every arm once, in order; then the highest upper confidence bound (the lowest index on a tie)."""
        untried = [arm for arm, count in enumerate(self.counts) if count == 0]
        if untried:
            return untried[0]
        total = sum(self.counts)
        span = self.highest_score - self.lowest_score
        bounds = []
        for count, score_sum in zip(self.counts, self.score_sums):
            scaled_mean = (score_sum / count - self.lowest_score) / span if span > 0.0 else 0.5
            bounds.append(scaled_mean + math.sqrt(2.0 * math.log(total) / count))
        return max(range(len(bounds)), key=lambda arm: (bounds[arm], -arm))

    def record(self, arm, score):
        self.counts[arm] += 1
        self.score_sums[arm] += score
        self.score_square_sum += score * score
        self.lowest_score = min(self.lowest_score, score)
        self.highest_score = max(self.highest_score, score)

    def mean(self, arm):
        return self.score_sums[arm] / self.counts[arm] if self.counts[arm] else None

    def best_arm_belief(self):
        """Gibbs (logit) belief over which arm is best: softmax of each arm's mean score over a temperature equal to
        the spread of all trial scores so far, since scores have no natural units. An untried arm stands at the
        overall mean; with no spread yet (fewer than two different scores) every arm is equally likely."""
        arm_count = len(self.counts)
        trial_count = sum(self.counts)
        if trial_count == 0:
            return [1.0 / arm_count] * arm_count
        overall_mean = sum(self.score_sums) / trial_count
        spread = math.sqrt(max(self.score_square_sum / trial_count - overall_mean * overall_mean, 0.0))
        if spread <= 0.0:
            return [1.0 / arm_count] * arm_count
        means = [
            overall_mean if self.counts[arm] == 0 else self.score_sums[arm] / self.counts[arm]
            for arm in range(arm_count)
        ]
        top = max(means)
        weights = [math.exp((mean - top) / spread) for mean in means]
        total = sum(weights)
        return [weight / total for weight in weights]


class CoAdapt:
    """Drives the Blend share from the bandit. Call after_step once per policy step while it is active."""

    def __init__(self, dt_s, shares=COADAPT_SHARES, trial_seconds=COADAPT_TRIAL_SECONDS):
        self.shares = shares
        self.trial_steps = max(1, round(trial_seconds / dt_s))
        self.dt_s = dt_s
        self.bandit = UcbBandit(len(shares))
        self.active = False
        self.arm = None
        self.trial_count = 0
        self._trial_step_count = 0
        self._trial_score = 0.0

    def start(self, env):
        self.active = True
        self._begin_trial(env)

    def stop(self):
        """Leaving co-adapt drops the trial in progress: part of a trial would score a different stretch of road."""
        self.active = False
        self.arm = None

    def restart(self, env):
        """A restarted drive starts a fresh trial; what the bandit has learned about the driver is kept."""
        if self.active:
            self._begin_trial(env)

    def after_step(self, loop):
        if not self.active:
            return None
        self._trial_score += human_step_score(loop)
        self._trial_step_count += 1
        if self._trial_step_count < self.trial_steps:
            return None
        finished_arm, score = self.arm, self._trial_score
        self.bandit.record(finished_arm, score)
        self.trial_count += 1
        self._begin_trial(loop.env)
        return finished_arm, score

    def _begin_trial(self, env):
        self.arm = self.bandit.select()
        self._trial_step_count = 0
        self._trial_score = 0.0
        env.set_assist_mode(binding.ASSIST_MODE_BLEND, self.shares[self.arm])

    @property
    def share(self):
        return None if self.arm is None else self.shares[self.arm]

    @property
    def seconds_left(self):
        return (self.trial_steps - self._trial_step_count) * self.dt_s

    def summary(self):
        """Each share's mean score and trial count, e.g. '20% -1.2 (2)  40% -0.8 (3)'."""
        parts = []
        for arm, share in enumerate(self.shares):
            mean = self.bandit.mean(arm)
            shown = "  -  " if mean is None else f"{mean:+.2f}"
            parts.append(f"{round(100 * share)}% {shown} ({self.bandit.counts[arm]})")
        return "  ".join(parts)
