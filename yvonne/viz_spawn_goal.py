"""
Spawn / Goal visualization — saves one figure per RL step containing 3 subplots:
  • Success   — agents that reached ≥ 1 goal (outcome == 1)
  • DNF       — no infractions AND no waypoints reached (outcome == 2)
  • Infraction — collided / offroad / red-light without reaching a goal (outcome == 0)

Each subplot draws a line from spawn (●) to next goal (★) for every agent.

Usage
-----
In training (pufferl.py or wherever completed-episode summaries are drained):

    from yvonne.viz_spawn_goal import SpawnGoalLogger

    viz = SpawnGoalLogger(save_dir="runs/spawn_goal", flush_every=50)

    # inside the summary-drain loop:
    for summary in completed_summaries:
        viz.add(summary)

    # at each RL step:
    viz.maybe_save(global_step)

The env must be created with emit_completed_episodes=True.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# outcome codes emitted by binding.c
OUTCOME_INFRACTION = 0
OUTCOME_SUCCESS = 1
OUTCOME_DNF = 2

_CMAPS = {
    OUTCOME_SUCCESS: "Greens",
    OUTCOME_DNF: "Oranges",
    OUTCOME_INFRACTION: "Reds",
}
_OUTCOME_LABELS = {
    OUTCOME_SUCCESS: "Success",
    OUTCOME_DNF: "DNF",
    OUTCOME_INFRACTION: "Infraction",
}


class SpawnGoalLogger:
    """Accumulates per-agent spatial data from completed-episode summaries
    and periodically saves a 3-panel figure to disk."""

    def __init__(self, save_dir: str = "runs/spawn_goal", flush_every: int = 50):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.flush_every = flush_every
        self._last_saved_step = -1
        self._buf: dict[int, dict] = {
            OUTCOME_SUCCESS: {"sx": [], "sy": [], "gx": [], "gy": []},
            OUTCOME_DNF: {"sx": [], "sy": [], "gx": [], "gy": []},
            OUTCOME_INFRACTION: {"sx": [], "sy": [], "gx": [], "gy": []},
        }

    def add(self, summary: dict) -> None:
        """Ingest one completed-episode summary dict (from vec_pop_completed_episodes)."""
        sx = summary.get("agent_spawn_x", [])
        sy = summary.get("agent_spawn_y", [])
        gx = summary.get("agent_final_goal_x", [])
        gy = summary.get("agent_final_goal_y", [])
        oc = summary.get("agent_outcome", [])
        if not sx:
            return
        sx = np.asarray(sx, dtype=np.float32)
        sy = np.asarray(sy, dtype=np.float32)
        gx = np.asarray(gx, dtype=np.float32)
        gy = np.asarray(gy, dtype=np.float32)
        oc = np.asarray(oc, dtype=np.int8)
        for outcome in (OUTCOME_SUCCESS, OUTCOME_DNF, OUTCOME_INFRACTION):
            mask = oc == outcome
            if not mask.any():
                continue
            b = self._buf[outcome]
            b["sx"].append(sx[mask])
            b["sy"].append(sy[mask])
            b["gx"].append(gx[mask])
            b["gy"].append(gy[mask])

    def maybe_save(self, global_step: int) -> str | None:
        """Save a figure if flush_every steps have passed. Returns the saved path or None."""
        if global_step - self._last_saved_step < self.flush_every:
            return None
        path = self.save(global_step)
        self._last_saved_step = global_step
        return path

    def save(self, global_step: int) -> str:
        """Always save a figure for global_step. Returns the file path."""
        outcomes = (OUTCOME_SUCCESS, OUTCOME_DNF, OUTCOME_INFRACTION)
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Spawn & goal density  (step {global_step:,})", fontsize=13)

        # Collect all positions to compute shared axis limits per row
        all_sx, all_sy, all_gx, all_gy = [], [], [], []
        data = {}
        for outcome in outcomes:
            b = self._buf[outcome]
            if b["sx"]:
                sx = np.concatenate(b["sx"])
                sy = np.concatenate(b["sy"])
                gx = np.concatenate(b["gx"])
                gy = np.concatenate(b["gy"])
                data[outcome] = (sx, sy, gx, gy)
                all_sx.append(sx); all_sy.append(sy)
                all_gx.append(gx); all_gy.append(gy)

        def _extent(arrays, pad=20.0):
            if not arrays:
                return -100, 100
            v = np.concatenate(arrays)
            return v.min() - pad, v.max() + pad

        sx_lo, sx_hi = _extent(all_sx)
        sy_lo, sy_hi = _extent(all_sy)
        gx_lo, gx_hi = _extent(all_gx)
        gy_lo, gy_hi = _extent(all_gy)

        BINS = 60
        row_labels = ["Spawn (init)", "Goal (next)"]

        for col, outcome in enumerate(outcomes):
            label = _OUTCOME_LABELS[outcome]
            cmap = _CMAPS[outcome]

            for row, (xs, ys, xlim, ylim, kind) in enumerate([
                (data.get(outcome, (None,))[0] if outcome in data else None,
                 data.get(outcome, (None, None))[1] if outcome in data else None,
                 (sx_lo, sx_hi), (sy_lo, sy_hi), "spawn"),
                (data.get(outcome, (None, None, None))[2] if outcome in data else None,
                 data.get(outcome, (None, None, None, None))[3] if outcome in data else None,
                 (gx_lo, gx_hi), (gy_lo, gy_hi), "goal"),
            ]):
                ax = axes[row, col]
                ax.set_title(f"{label} — {row_labels[row]}", fontsize=9)
                ax.set_aspect("equal")
                ax.set_xlabel("x (m)", fontsize=7)
                ax.set_ylabel("y (m)", fontsize=7)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

                if xs is not None and len(xs) >= 2:
                    n = len(xs)
                    h, xedges, yedges, img = ax.hist2d(
                        xs, ys,
                        bins=BINS,
                        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
                        cmap=cmap,
                        norm=LogNorm(vmin=1),
                    )
                    fig.colorbar(img, ax=ax, shrink=0.7, label="count")
                    ax.text(0.02, 0.98, f"n={n:,}", transform=ax.transAxes,
                            fontsize=7, va="top", ha="left", color="black",
                            bbox=dict(fc="white", alpha=0.6, pad=1, ec="none"))
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", color="gray", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"step_{global_step:08d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        for b in self._buf.values():
            for lst in b.values():
                lst.clear()

        return path
