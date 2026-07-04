"""Rebuild the per-seed "trend" runs in each nightly wandb project.

The nightly launchers stamp every run's wandb group with the launch date.
This script folds those into one small trend run per seed in the dedicated
TREND_PROJECT (named <source-project>-seed<N>): each logs the final value of every metric with the row
_timestamp overridden to the night's midnight (and step = days since
NIGHT_ZERO for monotonicity). A wall-time line panel grouped over the trend
runs then plots the mean across seeds with stderr bands on a real date
axis — see the "PufferDrive Nightlies" report
(scripts/make_nightly_report.py).

Re-running deletes and rebuilds the trend runs, so in-progress nights update
and new nights append. The nightly launchers invoke this before submitting,
which keeps the report current without any other scheduler.

    python scripts/update_nightly_trends.py
"""

import datetime
import os

# The Greene cluster's wandb settings default to the TORC instance; the
# nightly projects live on wandb.ai. Explicit env still wins.
os.environ.setdefault("WANDB_BASE_URL", "https://api.wandb.ai")

import wandb

ENTITY = "emerge_"
PROJECTS = ["nightly-multi", "nightly-single"]
TREND_PROJECT = "nightly-trends"
NIGHT_ZERO = datetime.date(2026, 7, 4)
METRICS = [
    "environment/score",
    "environment/episode_return",
    "environment/collision_rate",
    "environment/offroad_rate",
    "environment/num_goals_reached",
    "environment/avg_speed_per_agent",
    "environment/avg_distance_per_infraction",
    "SPS",
    "validation_gigaflow/score",
    "validation_gigaflow/episode_return",
    "validation_gigaflow/collision_rate",
    "validation_gigaflow/offroad_rate",
]


def night_of(run):
    if not run.group:
        return None
    try:
        return datetime.datetime.strptime(run.group, "%Y-%m-%d").date()
    except ValueError:
        return None


def main():
    api = wandb.Api()
    for project in PROJECTS:
        # newest first, so the first run seen per (night, seed) wins below
        by_seed = {}
        for run in api.runs(f"{ENTITY}/{project}", order="-created_at"):
            night = night_of(run)
            if night is None:
                continue
            seed = run.config.get("train", {}).get("seed")
            if seed is None:
                continue
            by_seed.setdefault(seed, {}).setdefault(night, run)

        try:
            stale = list(api.runs(f"{ENTITY}/{TREND_PROJECT}", filters={"group": project}))
        except ValueError:
            stale = []  # trend project not created yet; wandb.init below creates it
        for run in stale:
            run.delete()

        for seed in sorted(by_seed):
            trend = wandb.init(
                entity=ENTITY,
                project=TREND_PROJECT,
                name=f"{project}-seed{seed}",
                group=project,
                tags=["trend"],
            )
            for night in sorted(by_seed[seed]):
                run = by_seed[seed][night]
                night_index = (night - NIGHT_ZERO).days
                row = {
                    "_timestamp": datetime.datetime.combine(
                        night, datetime.time()
                    ).timestamp()
                }
                for metric in METRICS:
                    if metric in run.summary:
                        row[metric] = run.summary[metric]
                trend.log(row, step=night_index)
                print(f"{project} seed{seed} night {night_index} ({night}): {len(row) - 1} metrics")
            trend.finish()


if __name__ == "__main__":
    main()
