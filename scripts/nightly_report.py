"""Nightly wandb trend runs + the "PufferDrive Nightlies" report.

The nightly launchers stamp every run's wandb group with the launch date.
This tool folds those into a nightly regression view:

  update (default)  Refresh the report's data: rebuild the per-seed trend
                    runs in the nightly-trends project from the nightly
                    runs' final metric values. The launchers run this at
                    every submission; run it by hand to pull in a night
                    that finished since.

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
    "validation_gigaflow/avg_distance_per_infraction",
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
    "validation_gigaflow/avg_distance_per_infraction",
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

        # One aggregate trend run per project. Seeds are folded here (mean +
        # stderr per night) rather than by panel grouping, because per-series
        # style overrides (the "points" mark) only address ungrouped series,
        # keyed by run id. The run id changes on every rebuild, so update()
        # ends by rewriting the report, which re-keys the marks.
        nights = sorted({night for seeds in by_seed.values() for night in seeds})
        trend = wandb.init(
            entity=ENTITY,
            project=TREND_PROJECT,
            name=f"{project}-mean",
            group=project,
            tags=["trend"],
        )
        for night in nights:
            night_index = (night - NIGHT_ZERO).days
            row = {"_timestamp": datetime.datetime.combine(night, datetime.time()).timestamp()}
            for metric in TREND_METRICS:
                values = [
                    by_seed[seed][night].summary[metric]
                    for seed in by_seed
                    if night in by_seed[seed] and metric in by_seed[seed][night].summary
                ]
                if not values:
                    continue
                mean = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                    stderr = (variance / len(values)) ** 0.5
                else:
                    stderr = 0.0
                row[metric] = mean
                row[f"{metric}__hi"] = mean + stderr
                row[f"{metric}__lo"] = mean - stderr
            trend.log(row, step=night_index)
            print(f"{project} night {night_index} ({night}): {(len(row) - 1) // 3} metrics")
        trend.finish()


def trend_run_id(project):
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{TREND_PROJECT}", filters={"group": project}))
    if len(runs) != 1:
        raise RuntimeError(f"expected exactly one trend run for {project}, found {len(runs)}; rerun update first")
    return runs[0].id


def trend_section(wr, project):
    run_id = trend_run_id(project)
    runset = wr.Runset(
        entity=ENTITY,
        project=TREND_PROJECT,
        name=f"{project} trend runs",
        filters=f'group == "{project}"',
    )
    stderr_grey = "#b0b0b0"
    panels = [
        wr.LinePlot(
            x="_timestamp",
            y=[m, f"{m}__hi", f"{m}__lo"],
            range_x=(NIGHT_ZERO_TS, None),
            title=m,
            # "points" marks only bind to ungrouped series, keyed by run id;
            # update() rewrites this report after every rebuild so the keys
            # track the recreated trend run.
            line_marks={f"{run_id}:{y}": "points" for y in (m, f"{m}__hi", f"{m}__lo")},
            line_colors={f"{run_id}:{m}__hi": stderr_grey, f"{run_id}:{m}__lo": stderr_grey},
            line_titles={
                f"{run_id}:{m}": "mean",
                f"{run_id}:{m}__hi": "+stderr",
                f"{run_id}:{m}__lo": "-stderr",
            },
        )
        for m in FINALS_METRICS
    ]
    return [
        wr.H1(f"{project}: nightly trend (mean over seeds, ±stderr points)"),
        wr.P(
            "x = the night (wall-time axis; each trend row is stamped with "
            "its night's midnight). Seeds are folded into mean/±stderr rows "
            "in the nightly-trends project, rebuilt by this script at each "
            "nightly launch; unconnected points keep skipped nights visible."
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
        # The rebuild gave the trend runs new ids; rewrite the report so its
        # run-id-keyed point marks bind to the new runs.
        make_report(create=False)
    else:
        make_report(args.create)


if __name__ == "__main__":
    main()
