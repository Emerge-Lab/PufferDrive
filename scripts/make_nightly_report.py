"""Create or rewrite the "PufferDrive Nightlies" wandb report.

Layout, per nightly project (nightly-multi, nightly-single):
  1. Nightly trend — line panels over the per-seed trend runs that
     scripts/update_nightly_trends.py maintains: x = the night as a real
     date axis, y = mean across seeds with stderr bands.
  2. Nightly finals — native bar charts grouped by run group (the launch
     date): one bar per night, mean over seeds; the bar labels carry the
     actual dates.
  3. Training curves — one mean-over-seeds line per night (run group) with
     stderr bands, so nights overlay for regression comparison.

Panels are live wandb queries; the launchers refresh the trend runs each
night. Requires the wandb-workspaces package (pip install wandb-workspaces).

    python scripts/make_nightly_report.py             # edit the existing report
    python scripts/make_nightly_report.py --create    # mint a fresh report
"""

import os
import sys

# The Greene cluster's wandb settings default to the TORC instance; the
# nightly projects live on wandb.ai. Explicit env still wins.
os.environ.setdefault("WANDB_BASE_URL", "https://api.wandb.ai")

import wandb_workspaces.reports.v2 as wr

ENTITY = "emerge_"
PROJECTS = ["nightly-multi", "nightly-single"]
# 2026-07-03 midnight: one day before the first night, so the first point is visible
NIGHT_ZERO_TS = 1783051200.0
REPORT_URL = "https://wandb.ai/emerge_/nightly-multi/reports/PufferDrive-Nightlies--VmlldzoxNzQxNzI4NQ=="

FINALS_METRICS = [
    "environment/score",
    "environment/episode_return",
    "environment/collision_rate",
    "environment/offroad_rate",
    "environment/num_goals_reached",
    "SPS",
    "validation_gigaflow/score",
    "validation_gigaflow/collision_rate",
]
CURVE_METRICS = [
    "environment/score",
    "environment/episode_return",
    "environment/collision_rate",
    "environment/offroad_rate",
    "SPS",
    "losses/entropy",
]


def trend_section(project):
    runset = wr.Runset(entity=ENTITY, project=project, name="trend runs", query="trend")
    panels = [
        wr.LinePlot(
            x="_timestamp",
            y=[m],
            groupby="group",
            groupby_aggfunc="mean",
            groupby_rangefunc="stderr",
            # Anchor the axis at night zero: with a single night logged the
            # autorange spans [0, 2x], which renders as the year 1969.
            range_x=(NIGHT_ZERO_TS, None),
            title=m,
        )
        for m in FINALS_METRICS
    ]
    return [
        wr.H1(f"{project}: nightly trend (mean over seeds, stderr bands)"),
        wr.P(
            "x = the night (wall-time axis; each trend row is stamped with "
            "its night\'s midnight). Trend runs are rebuilt by "
            "scripts/update_nightly_trends.py at each nightly launch."
        ),
        wr.PanelGrid(runsets=[runset], panels=panels),
    ]


def finals_section(project):
    runset = wr.Runset(
        entity=ENTITY,
        project=project,
        name=f"{project} (all nights)",
        filters='group != "trend"',
    )
    bar_panels = [
        wr.BarPlot(
            metrics=[m],
            groupby="group",
            groupby_aggfunc="mean",
            groupby_rangefunc="stderr",
            title=f"{m} - final per night (mean over seeds)",
        )
        for m in FINALS_METRICS
    ]
    return [
        wr.H1(f"{project}: nightly finals"),
        wr.P(
            "One bar per night (run group): mean over seeds of the final "
            "logged value, with stderr."
        ),
        wr.PanelGrid(runsets=[runset], panels=bar_panels),
    ]


def curves_section(project):
    runset = wr.Runset(
        entity=ENTITY,
        project=project,
        name=f"{project} (all nights)",
        filters='group != "trend"',
    )
    panels = [
        wr.LinePlot(
            x="agent_steps",
            y=[m],
            groupby="group",
            groupby_aggfunc="mean",
            groupby_rangefunc="stderr",
            title=m,
        )
        for m in CURVE_METRICS
    ]
    return [
        wr.H1(f"{project}: training curves (one line per night)"),
        wr.PanelGrid(runsets=[runset], panels=panels),
    ]


def main():
    if "--create" in sys.argv:
        report = wr.Report(
            entity=ENTITY,
            project=PROJECTS[0],
            title="PufferDrive Nightlies",
            description=(
                "Nightly multi-agent and single-agent training: date-indexed "
                "final-value bars plus per-night training curves, mean over "
                "seeds. Auto-updates as new nights land."
            ),
        )
    else:
        report = wr.Report.from_url(REPORT_URL)

    blocks = []
    for project in PROJECTS:
        blocks += trend_section(project)
    for project in PROJECTS:
        blocks += finals_section(project)
    for project in PROJECTS:
        blocks += curves_section(project)
    report.blocks = blocks
    report.save()
    print("report:", report.url)


if __name__ == "__main__":
    main()
