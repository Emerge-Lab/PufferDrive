"""Bandit-driven policy comparison over WOMD maps.

Selects maps using a UCB1 bandit to maximise signed regret (score_p1 - score_p2),
concentrating rollouts on maps that best discriminate the two policies.

Each bandit arm = one scenario (.bin file).
Each pull = N rollouts on that map with control_wosac:
  all tracks_to_predict agents are swapped with the policy for 81 policy steps
  (10 open-loop init steps + 81 = 91 total, matching the WOMD scenario length).
Score = goals_reached / goals_attempted, averaged over all controlled agents.

Usage
-----
python -m pufferlib.ocean.drive.bandit_compare \\
    --policy1 weights1.pt --policy2 weights2.pt \\
    --map-dir resources/drive/binaries/training \\
    [--num-rounds 50] [--rollouts-per-map 3] [--batch-size 8] \\
    [--exploration-coeff 2.0] [--device cuda] \\
    [--policy-input-size 64] [--policy-hidden-size 256] [--rnn-hidden-size 256]
"""

import os
import math
import tempfile
import argparse

import numpy as np
import torch

import pufferlib
import pufferlib.pytorch
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.torch import Drive as DrivePolicy, Recurrent


# ──────────────────────────────────────────────────────────────────────────────
# UCB1 Bandit
# ──────────────────────────────────────────────────────────────────────────────


class MapBandit:
    """UCB1 bandit over map arms.

    Exploitation term : |mean_regret[m]|
    Exploration bonus : coeff * sqrt( log(t+1) / (counts[m]+1) )

    Welford online variance is maintained for confidence intervals.
    Regret is stored signed so the caller can see which policy wins where.
    """

    def __init__(self, num_maps: int, exploration_coeff: float = 2.0):
        self.num_maps = num_maps
        self.exploration_coeff = exploration_coeff
        self.counts = np.zeros(num_maps, dtype=np.float64)
        self.mean_regret = np.zeros(num_maps, dtype=np.float64)
        self._M2 = np.zeros(num_maps, dtype=np.float64)  # Welford
        self.t = 0  # total map selections made

    def select(self, k: int) -> np.ndarray:
        """Return indices of k maps with highest UCB score."""
        bonus = self.exploration_coeff * np.sqrt(np.log(self.t + 1) / (self.counts + 1))
        ucb = np.abs(self.mean_regret) + bonus
        # Small random tiebreak so unseen arms don't always come out in
        # alphabetical order when they share the same UCB value.
        ucb += np.random.uniform(0, 1e-9, size=self.num_maps)
        return np.argsort(ucb)[::-1][:k]

    def update(self, map_indices: np.ndarray, regrets: np.ndarray):
        """Welford update for each (map_index, regret) pair."""
        for idx, r in zip(map_indices, regrets):
            self.counts[idx] += 1
            delta = r - self.mean_regret[idx]
            self.mean_regret[idx] += delta / self.counts[idx]
            delta2 = r - self.mean_regret[idx]
            self._M2[idx] += delta * delta2
        self.t += len(map_indices)

    def ci_halfwidth(self, z: float = 1.96) -> np.ndarray:
        """95 % CI half-width per arm (nan for arms with <2 pulls)."""
        var = np.full(self.num_maps, np.nan)
        m = self.counts >= 2
        var[m] = self._M2[m] / (self.counts[m] - 1)
        stderr = np.sqrt(var / np.maximum(self.counts, 1))
        return z * stderr


# ──────────────────────────────────────────────────────────────────────────────
# Policy loading
# ──────────────────────────────────────────────────────────────────────────────


def load_policy_weights(
    weights_path: str,
    env: Drive,
    input_size: int,
    hidden_size: int,
    rnn_hidden_size: int,
    device: str,
    use_rnn: bool,
) -> torch.nn.Module:
    """Instantiate Drive + optional Recurrent, load weights from disk."""
    base = DrivePolicy(env, input_size=input_size, hidden_size=hidden_size)
    if use_rnn:
        policy = Recurrent(env, base, input_size=hidden_size, hidden_size=rnn_hidden_size)
    else:
        policy = base
    policy = policy.to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


# ──────────────────────────────────────────────────────────────────────────────
# Single-map evaluation
# ──────────────────────────────────────────────────────────────────────────────

_WOMD_EPISODE_STEPS = 91
_WOMD_INIT_STEPS = 10
_POLICY_STEPS = _WOMD_EPISODE_STEPS - _WOMD_INIT_STEPS  # 81


def _make_eval_env(map_path: str, num_agents: int) -> Drive:
    """Create a Drive env pinned to a single map via a tmpdir symlink."""
    tmpdir = tempfile.mkdtemp(prefix="bandit_map_")
    link = os.path.join(tmpdir, "map.bin")
    os.symlink(os.path.abspath(map_path), link)
    env = Drive(
        map_dir=tmpdir,
        num_maps=1,
        use_all_maps=True,
        resample_frequency=0,
        control_mode="control_wosac",
        init_mode="create_all_valid",
        init_steps=_WOMD_INIT_STEPS,
        episode_length=_WOMD_EPISODE_STEPS,
        goal_behavior=2,
        goal_radius=2.0,
        num_agents=num_agents,
        allow_fewer_maps=True,
        report_interval=1,
    )
    env._bandit_tmpdir = tmpdir
    env._bandit_link = link
    return env


def _cleanup_env(env: Drive):
    try:
        env.close()
    except Exception:
        pass
    try:
        os.unlink(env._bandit_link)
        os.rmdir(env._bandit_tmpdir)
    except Exception:
        pass


def evaluate_map(
    policy: torch.nn.Module,
    map_path: str,
    n_rollouts: int,
    device: str,
    use_rnn: bool,
    num_agents: int,
) -> float:
    """Run policy on one WOMD scenario for n_rollouts; return mean score.

    Returns 0.0 if the map produces no valid controlled agents or the log
    never fires (can happen on degenerate scenarios).
    """
    try:
        env = _make_eval_env(map_path, num_agents=num_agents)
    except Exception as exc:
        print(f"    [skip] could not load {os.path.basename(map_path)}: {exc}")
        return 0.0

    # Save the real agent count for buffer sizing, then lower the vec_log
    # threshold to 1 so the log fires as soon as any agent completes their
    # episode — regardless of how many controlled agents the scenario has.
    na = env.num_agents
    env.num_agents = 1
    scores = []

    try:
        for _ in range(n_rollouts):
            obs, _ = env.reset()
            state = {}
            if use_rnn and hasattr(policy, "hidden_size"):
                state = dict(
                    lstm_h=torch.zeros(na, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(na, policy.hidden_size, device=device),
                )

            has_forward_eval = hasattr(policy, "forward_eval")
            episode_score = None
            for _ in range(_POLICY_STEPS):
                with torch.no_grad():
                    ob_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
                    if has_forward_eval:
                        logits, _ = policy.forward_eval(ob_t, state)
                    else:
                        logits, _ = policy(ob_t, state)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits)
                    action_np = action.cpu().numpy().reshape(env.action_space.shape)

                obs, _, _, _, info_list = env.step(action_np)

                if info_list:
                    episode_score = info_list[0]["score"]
                    break

            if episode_score is not None:
                scores.append(episode_score)
    finally:
        _cleanup_env(env)

    return float(np.mean(scores)) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────


def print_rankings(map_files: list, bandit: MapBandit, top_n: int = 20):
    ci = bandit.ci_halfwidth()
    rows = [
        (i, bandit.mean_regret[i], ci[i], int(bandit.counts[i])) for i in range(len(map_files)) if bandit.counts[i] > 0
    ]
    rows.sort(key=lambda r: abs(r[1]), reverse=True)

    print(f"\n{'─' * 74}")
    print(f"{'Rank':<5} {'Map':<37} {'Regret':>8} {'95%CI':>8} {'Pulls':>6}")
    print(f"{'─' * 74}")
    for rank, (i, mean_r, half_ci, n) in enumerate(rows[:top_n], 1):
        name = os.path.basename(map_files[i])
        ci_str = f"±{half_ci:.3f}" if not np.isnan(half_ci) else "   n/a"
        print(f"{rank:<5} {name:<37} {mean_r:>+8.4f} {ci_str:>8} {n:>6}")
    print(f"{'─' * 74}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="UCB1 bandit comparison of two policies over WOMD maps")
    parser.add_argument("--policy1", required=True, help="Policy 1 weights (.pt)")
    parser.add_argument("--policy2", required=True, help="Policy 2 weights (.pt)")
    parser.add_argument("--map-dir", required=True, help="Directory of .bin WOMD maps")
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument(
        "--rollouts-per-map", type=int, default=3, help="Rollouts per policy per selected map per round"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Number of maps selected per bandit round")
    parser.add_argument("--exploration-coeff", type=float, default=2.0, help="UCB1 exploration coefficient")
    parser.add_argument(
        "--num-agents", type=int, default=64, help="Agent buffer size passed to Drive (>= tracks_to_predict in any map)"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy-input-size", type=int, default=64)
    parser.add_argument("--policy-hidden-size", type=int, default=256)
    parser.add_argument("--rnn-hidden-size", type=int, default=256)
    parser.add_argument("--no-rnn", action="store_true", help="Disable RNN wrapper")
    parser.add_argument("--top-n", type=int, default=20, help="How many maps to show in the rankings table")
    parser.add_argument("--print-every", type=int, default=5, help="Print rankings every N rounds")
    args = parser.parse_args()

    use_rnn = not args.no_rnn

    # ── Discover maps ────────────────────────────────────────────────────────
    map_files = sorted([os.path.join(args.map_dir, f) for f in os.listdir(args.map_dir) if f.endswith(".bin")])
    if not map_files:
        raise FileNotFoundError(f"No .bin files found in {args.map_dir}")
    print(f"Found {len(map_files)} maps in {args.map_dir}")

    # ── Reference env for policy architecture ────────────────────────────────
    print("Building reference env to infer policy architecture...")
    ref_env = _make_eval_env(map_files[0], num_agents=args.num_agents)

    print(f"Loading policies onto {args.device}...")
    policy1 = load_policy_weights(
        args.policy1,
        ref_env,
        args.policy_input_size,
        args.policy_hidden_size,
        args.rnn_hidden_size,
        args.device,
        use_rnn,
    )
    policy2 = load_policy_weights(
        args.policy2,
        ref_env,
        args.policy_input_size,
        args.policy_hidden_size,
        args.rnn_hidden_size,
        args.device,
        use_rnn,
    )
    _cleanup_env(ref_env)
    print(f"  policy1: {args.policy1}")
    print(f"  policy2: {args.policy2}")

    bandit = MapBandit(len(map_files), exploration_coeff=args.exploration_coeff)

    print(
        f"\nStarting {args.num_rounds} rounds | "
        f"batch={args.batch_size} | "
        f"rollouts/map={args.rollouts_per_map} | "
        f"total_map_evals≤{args.num_rounds * args.batch_size * 2}\n"
    )

    # ── Bandit loop ──────────────────────────────────────────────────────────
    for round_idx in range(args.num_rounds):
        selected = bandit.select(args.batch_size)
        regrets = []

        for map_idx in selected:
            map_path = map_files[map_idx]
            name = os.path.basename(map_path)

            s1 = evaluate_map(policy1, map_path, args.rollouts_per_map, args.device, use_rnn, args.num_agents)
            s2 = evaluate_map(policy2, map_path, args.rollouts_per_map, args.device, use_rnn, args.num_agents)
            regret = s1 - s2
            regrets.append(regret)

            print(f"  [{round_idx + 1:3d}/{args.num_rounds}] {name:<37} p1={s1:.4f}  p2={s2:.4f}  regret={regret:+.4f}")

        bandit.update(selected, np.array(regrets))

        if (round_idx + 1) % args.print_every == 0:
            print_rankings(map_files, bandit, top_n=args.top_n)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "═" * 74)
    print("FINAL RANKINGS")
    print_rankings(map_files, bandit, top_n=args.top_n)

    seen = bandit.counts > 0
    n_seen = int(seen.sum())
    print(f"Maps evaluated : {n_seen} / {len(map_files)}")
    print(f"Total rounds   : {args.num_rounds}")

    if n_seen > 0:
        top_idx = int(np.argmax(np.abs(bandit.mean_regret) * seen))
        top_name = os.path.basename(map_files[top_idx])
        top_regret = bandit.mean_regret[top_idx]
        winner = "policy1" if top_regret > 0 else "policy2"
        print(f"Most discriminating map : {top_name} (regret={top_regret:+.4f}, {winner} wins here)")


if __name__ == "__main__":
    main()
