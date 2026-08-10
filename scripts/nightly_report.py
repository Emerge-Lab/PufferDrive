"""Nightly wandb trend runs + the "PufferDrive Nightlies" report.

The nightly launchers stamp every run's wandb group with the launch date.
This tool folds those into a nightly regression view:

  update (default)  Refresh the report's data: rebuild the per-seed trend
                    runs in the nightly-trends project from the nightly
                    runs' final metric values, except SPS, which is
                    averaged over the run. The launchers run this at
                    every submission, before that night's runs have
                    reported anything, so a night only lands here on the
                    next launch; run it by hand to pull one in sooner.

  report [--create] Rewrite the report's layout: which panels exist and how
                    they aggregate, per nightly project — (1) trend line
                    panels over the trend runs (x = the night, y = mean
                    across seeds with stderr bands); (2) bar charts of
                    finals grouped by night; (3) per-night mean training
                    curves. Panels are live queries, so new data appears
                    without rerunning this — only rerun it to change the
                    panel set (e.g. after editing TREND_METRICS et al.).

    python scripts/nightly_report.py            # rebuild trend runs
    python scripts/nightly_report.py report     # rewrite the report in place
"""

import argparse
import datetime
import os

# Ensure correct wandb for machines with multiple wandb accounts
os.environ.setdefault("WANDB_BASE_URL", "https://api.wandb.ai")

import wandb

ENTITY = "emerge_"
PROJECTS = ["nightly-multi", "nightly-single", "nightly-multi-long"]
TREND_PROJECT = "nightly-trends"
NIGHT_ZERO = datetime.date(2026, 7, 4)
# The multi-agent nightlies started logging eval_carla_fast on this night; earlier ones have no eval keys.
EVAL_START = datetime.date(2026, 8, 3)
# The single-agent nightly runs no evaluation, so it gets no eval panels.
EVAL_PROJECTS = ("nightly-multi", "nightly-multi-long")
SPS_SAMPLES = 500
REPORT_URL = "https://wandb.ai/emerge_/nightly-multi/reports/PufferDrive-Nightlies--VmlldzoxNzQxNzI4NQ=="


def axis_start(night):
    # One day before the first plotted night: a lone point makes the wall-time autorange render as 1969.
    start = night - datetime.timedelta(days=1)
    return datetime.datetime.combine(start, datetime.time()).timestamp()


TRAIN_TREND_METRICS = [
    "environment/episode_return",
    "environment/collision_rate",
    "environment/offroad_rate",
    "environment/avg_speed_per_agent",
    "environment/avg_distance_per_infraction",
    "SPS",
]
EVAL_TREND_METRICS = [
    "eval_carla_fast/episode_return",
    "eval_carla_fast/collision_rate",
    "eval_carla_fast/offroad_rate",
    "eval_carla_fast/avg_distance_per_infraction",
]
TREND_METRICS = TRAIN_TREND_METRICS + EVAL_TREND_METRICS
TRAIN_FINALS_METRICS = [
    "environment/episode_return",
    "environment/collision_rate",
    "environment/offroad_rate",
    "environment/avg_distance_per_infraction",
    "SPS",
]
EVAL_FINALS_METRICS = [
    "eval_carla_fast/collision_rate",
    "eval_carla_fast/avg_distance_per_infraction",
]
CURVE_METRICS = [
    "environment/episode_return",
    "environment/collision_rate",
    "environment/offroad_rate",
    "SPS",
    "losses/entropy",
]


WIDE_METRIC = "avg_distance_per_infraction"
GRID_WIDTH = 24
PANEL_WIDTH = 8
PANEL_HEIGHT = 6
WIDE_PANEL_HEIGHT = 10


def layout_panels(wr, panels):
    # Panels carry a default layout that stacks them 3-up; assign explicit slots so the
    # wide metric gets a full-width row of its own above the rest.
    wide = [panel for panel in panels if WIDE_METRIC in panel.title]
    rest = [panel for panel in panels if WIDE_METRIC not in panel.title]
    next_y = 0
    for panel in wide:
        panel.layout = wr.Layout(x=0, y=next_y, w=GRID_WIDTH, h=WIDE_PANEL_HEIGHT)
        next_y += WIDE_PANEL_HEIGHT
    per_row = GRID_WIDTH // PANEL_WIDTH
    for index, panel in enumerate(rest):
        panel.layout = wr.Layout(
            x=(index % per_row) * PANEL_WIDTH,
            y=next_y + (index // per_row) * PANEL_HEIGHT,
            w=PANEL_WIDTH,
            h=PANEL_HEIGHT,
        )
    return wide + rest


def night_of(run):
    if not run.group:
        return None
    try:
        return datetime.datetime.strptime(run.group, "%Y-%m-%d").date()
    except ValueError:
        return None


def mean_sps(run):
    # Skips the zeros pufferl logs whenever no steps elapsed since the previous log.
    rows = run.history(keys=["SPS"], pandas=False, samples=SPS_SAMPLES)
    values = [row["SPS"] for row in rows if row.get("SPS")]
    if not values:
        return None
    return sum(values) / len(values)


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
                # summary SPS is the last logged value, which is 0 for every run that trained to completion
                sps = mean_sps(run)
                if sps is not None:
                    row["SPS"] = sps
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

    def line(metric, first_night):
        return wr.LinePlot(
            x="_timestamp",
            y=[metric],
            groupby="group",
            groupby_aggfunc="mean",
            groupby_rangefunc="stderr",
            range_x=(axis_start(first_night), None),
            title=metric,
        )

    panels = [line(m, NIGHT_ZERO) for m in TRAIN_FINALS_METRICS]
    if project in EVAL_PROJECTS:
        panels += [line(m, EVAL_START) for m in EVAL_TREND_METRICS]
    panels = layout_panels(wr, panels)
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
    metrics = list(TRAIN_FINALS_METRICS)
    if project in EVAL_PROJECTS:
        metrics += EVAL_FINALS_METRICS
    panels = [
        wr.BarPlot(
            metrics=[m],
            groupby="group",
            groupby_aggfunc="mean",
            groupby_rangefunc="stderr",
            title=f"{m} - final per night (mean over seeds)",
        )
        for m in metrics
    ]
    panels = layout_panels(wr, panels)
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
