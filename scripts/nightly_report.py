"""Nightly wandb trend runs + the "PufferDrive Nightlies" report.

The nightly launchers stamp every run's wandb group with the launch date.
This tool folds those into a nightly regression view:

  update (default)  Rebuild the per-seed trend runs in the nightly-trends
                    project (one run per source project and seed, named
                    <project>-seed<N>, grouped by source project). Each logs
                    the final value of every metric with the row _timestamp
                    overridden to the night's midnight (and step = days since
                    NIGHT_ZERO for monotonicity). Re-running deletes and
                    rebuilds them, so in-progress nights update and new
                    nights append. The launchers invoke this before
                    submitting, which keeps the report current without any
                    other scheduler.

  report [--create] Create or rewrite the report. Layout, per nightly
                    project: (1) trend line panels over the trend runs — x =
                    the night on a real date axis, y = mean across seeds
                    with stderr bands; (2) native bar charts of finals
                    grouped by run group (the launch date); (3) per-night
                    mean training curves. Panels are live queries, so new
                    nights appear without edits. Requires wandb-workspaces
                    (pip install wandb-workspaces).

    python scripts/nightly_report.py            # rebuild trend runs
    python scripts/nightly_report.py report     # rewrite the report in place
"""

import argparse
import datetime
import os

# The nightly projects live on wandb.ai regardless of any locally configured
# wandb host. Explicit env still wins.
os.environ.setdefault("WANDB_BASE_URL", "https://api.wandb.ai")

import wandb

ENTITY = "emerge_"
PROJECTS = ["nightly-multi", "nightly-single"]
TREND_PROJECT = "nightly-trends"
NIGHT_ZERO = datetime.date(2026, 7, 4)
# One day before the first night, so the first point is visible: with a
# single night logged the wall-time autorange spans [0, 2x], which renders
# as the year 1969.
NIGHT_ZERO_TS = 1783051200.0
REPORT_URL = "https://wandb.ai/emerge_/nightly-multi/reports/PufferDrive-Nightlies--VmlldzoxNzQxNzI4NQ=="

TREND_METRICS = [
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
FINALS_METRICS = [
    "environment/score",
    "environment/episode_return",
    "environment/collision_rate",
    "environment/offroad_rate",
    "environment/num_goals_reached",
    "environment/avg_distance_per_infraction",
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


def night_of(run):
    if not run.group:
        return None
    try:
        return datetime.datetime.strptime(run.group, "%Y-%m-%d").date()
    except ValueError:
        return None


def update_trends():
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
                row = {"_timestamp": datetime.datetime.combine(night, datetime.time()).timestamp()}
                for metric in TREND_METRICS:
                    if metric in run.summary:
                        row[metric] = run.summary[metric]
                trend.log(row, step=night_index)
                print(f"{project} seed{seed} night {night_index} ({night}): {len(row) - 1} metrics")
            trend.finish()


def trend_section(wr, project):
    runset = wr.Runset(
        entity=ENTITY,
        project=TREND_PROJECT,
        name=f"{project} trend runs",
        filters=f'group == "{project}"',
    )
    panels = [
        wr.LinePlot(
            x="_timestamp",
            y=[m],
            groupby="group",
            groupby_aggfunc="mean",
            groupby_rangefunc="stderr",
            range_x=(NIGHT_ZERO_TS, None),
            title=m,
        )
        for m in FINALS_METRICS
    ]
    return [
        wr.H1(f"{project}: nightly trend (mean over seeds, stderr bands)"),
        wr.P(
            "x = the night (wall-time axis; each trend row is stamped with "
            "its night's midnight). Trend runs live in the nightly-trends "
            "project and are rebuilt by this script at each nightly launch."
        ),
        wr.PanelGrid(runsets=[runset], panels=panels),
    ]


def finals_section(wr, project):
    runset = wr.Runset(entity=ENTITY, project=project, name=f"{project} (all nights)")
    panels = [
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
        wr.P("One bar per night (run group): mean over seeds of the final logged value, with stderr."),
        wr.PanelGrid(runsets=[runset], panels=panels),
    ]


def curves_section(wr, project):
    runset = wr.Runset(entity=ENTITY, project=project, name=f"{project} (all nights)")
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


def make_report(create):
    import wandb_workspaces.reports.v2 as wr

    if create:
        report = wr.Report(
            entity=ENTITY,
            project=PROJECTS[0],
            title="PufferDrive Nightlies",
            description=(
                "Nightly multi-agent and single-agent training: date-indexed "
                "final-value trends plus per-night training curves, mean over "
                "seeds. Auto-updates as new nights land."
            ),
        )
    else:
        report = wr.Report.from_url(REPORT_URL)

    blocks = []
    for project in PROJECTS:
        blocks += trend_section(wr, project)
    for project in PROJECTS:
        blocks += finals_section(wr, project)
    for project in PROJECTS:
        blocks += curves_section(wr, project)
    report.blocks = blocks
    report.save()
    print("report:", report.url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="update", choices=["update", "report"])
    parser.add_argument("--create", action="store_true", help="mint a fresh report instead of editing in place")
    args = parser.parse_args()
    if args.command == "update":
        update_trends()
    else:
        make_report(args.create)


if __name__ == "__main__":
    main()
