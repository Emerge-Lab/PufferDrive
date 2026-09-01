"""Evaluator for the CARLA and nuPlan closed-loop co-simulation benchmarks.
"""

import contextlib
import json
import os
import time

import numpy as np
import pandas as pd

import pufferlib
from pufferlib.ocean.evaluation_utils.evaluation_utils import (
    _load_yaml_mapping,
    _positive_int,
    _require_mapping,
)

COSIM_SIMULATION_MODES = ("carla_cosim", "nuplan_cosim")

DEFAULT_NUPLAN_CHALLENGES = (
    "closed_loop_nonreactive_agents_pufferdrive",
    "closed_loop_reactive_agents_pufferdrive",
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(pufferlib.__file__)))
CARLA_LEADERBOARD_SCRIPT = os.path.join(_REPO_ROOT, "pufferlib/ocean/cosim/carla/run_leaderboard.sh")
NUPLAN_PLANNER_SCRIPT = os.path.join(_REPO_ROOT, "pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh")
CARLA_PORT_STRIDE = 20  # distinct CARLA_PORT per route job, guards against two landing on one node


# ─── benchmark config parsing ────────────────────────────────────────────────
def parse_cosim_benchmark(name, benchmark):
    simulation_mode = benchmark.get("simulation_mode")
    if simulation_mode == "carla_cosim":
        return _parse_carla_benchmark(name, benchmark)
    if simulation_mode == "nuplan_cosim":
        return _parse_nuplan_benchmark_config(name, benchmark)
    raise pufferlib.APIUsageError(f"Benchmark {name} has unsupported cosim simulation_mode: {simulation_mode}")


def _parse_carla_benchmark(name, benchmark):
    route_ids = benchmark.get("route_ids")
    if not isinstance(route_ids, list) or not route_ids or any(not isinstance(r, int) for r in route_ids):
        raise pufferlib.APIUsageError(f"Benchmark {name} route_ids must be a non-empty list of integers")

    return {
        "name": name,
        "simulation_mode": "carla_cosim",
        "route_ids": route_ids,
        "routes": benchmark.get("routes"),  # None -> run_leaderboard.sh's own default routes xml
        "base_carla_port": _positive_int(benchmark.get("base_carla_port", 2000), f"Benchmark {name} base_carla_port"),
        "device": benchmark.get("device"),
        "dynamics_source": benchmark.get("dynamics_source"),
        "compute_config": benchmark.get("compute_config"),
    }


def _parse_nuplan_benchmark_config(name, benchmark):
    scenario_filters = benchmark.get("scenario_filters")
    if (
        not isinstance(scenario_filters, list)
        or not scenario_filters
        or any(not isinstance(s, str) or not s for s in scenario_filters)
    ):
        raise pufferlib.APIUsageError(f"Benchmark {name} scenario_filters must be a non-empty list of strings")
    challenges = benchmark.get("challenges", list(DEFAULT_NUPLAN_CHALLENGES))
    if not isinstance(challenges, list) or not challenges:
        raise pufferlib.APIUsageError(f"Benchmark {name} challenges must be a non-empty list of strings")

    return {
        "name": name,
        "simulation_mode": "nuplan_cosim",
        "scenario_filters": scenario_filters,
        "challenges": challenges,
        "worker": benchmark.get("worker"),
        "threads_per_node": benchmark.get("threads_per_node"),
        "limit_total_scenarios": benchmark.get("limit_total_scenarios"),
        "debug_bev": bool(benchmark.get("debug_bev", False)),
        "nuplan_env": _require_mapping(benchmark.get("nuplan_env", {}), f"Benchmark {name} nuplan_env"),
        "compute_config": benchmark.get("compute_config"),
    }


# ─── submitit launcher (mirrors scripts/submit_cluster.py's executor setup) ──
def _load_compute_config(compute_config):
    if compute_config is None:
        return {}
    if isinstance(compute_config, str):
        return _load_yaml_mapping(compute_config, "cosim compute config")
    return _require_mapping(compute_config, "cosim compute config")


@contextlib.contextmanager
def _clean_env_prefixes(*prefixes):
    saved = {key: os.environ.pop(key) for key in list(os.environ) if key.startswith(prefixes)}
    try:
        yield
    finally:
        os.environ.update(saved)


def _clean_slurm_env():
    """Submitting a new sbatch job from inside an already-running SLURM job
    (as the training-loop debug hook does) leaks the parent job's SLURM_* env
    vars -- SLURM_CPU_BIND in particular -- into `sbatch`'s inherited
    environment, and the child job then fails at its own srun step with "CPU
    binding outside of job step allocation" once it lands on a different
    node/allocation. Strip them for just the submission call."""
    return _clean_env_prefixes("SLURM_")


def _clean_slurm_and_wandb_env():
    """Same leak as _clean_slurm_env, but also strips WANDB_* -- the parent
    training process's live WandbLogger sets WANDB_SERVICE (and friends)
    pointing at a local wandb-core socket under its own SLURM job's
    node-local scratch dir. Submitting the orchestrator job (which later
    calls wandb.init itself, possibly on a different node, long after the
    parent's socket is gone) with that inherited would make it try to reuse
    a dead service and silently fall back to logging a disconnected run."""
    return _clean_env_prefixes("SLURM_", "WANDB_")


def _build_executor(compute_config, folder, job_name):
    import submitit

    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=20)
    if compute_config.get("gpu_type") is not None:
        gres = f"gpu:{compute_config['gpu_type']}:{compute_config.get('gpus', 1)}"
    elif compute_config.get("gpus") is not None:
        gres = f"gpu:{compute_config['gpus']}"
    else:
        gres = None
    executor.update_parameters(
        slurm_account=compute_config.get("account"),
        slurm_partition=compute_config.get("partition"),
        cpus_per_task=compute_config.get("cpus", 8),
        nodes=compute_config.get("nodes", 1),
        tasks_per_node=1,
        slurm_gres=gres,
        slurm_exclude=compute_config.get("exclude") or None,
        slurm_mem=compute_config.get("mem", "64gb") or "64gb",
        slurm_time=compute_config.get("time", 120),
        slurm_job_name=job_name,
    )
    return executor


def _run_script(cmd, env, subdir):
    import subprocess

    os.makedirs(subdir, exist_ok=True)
    with open(os.path.join(subdir, "launcher.log"), "w") as log_file:
        result = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    return result.returncode


# ─── CARLA: one SLURM job per route, via run_leaderboard.sh ──────────────────
def _run_carla_route_job(subdir, checkpoint_path, script_path, routes, route_id, carla_port, device, dynamics_source):
    """Runs inside the SLURM job: run_leaderboard.sh starts its own CARLA
    server, runs CaRL's original_leaderboard evaluator against `route_id`
    with leaderboard_agent.py as the driving agent, and writes result.json."""
    out_path = os.path.join(subdir, "result.json")
    env = os.environ.copy()
    env["CKPT"] = checkpoint_path
    env["ROUTES_SUBSET"] = str(route_id)
    env["OUT"] = out_path
    env["CARLA_PORT"] = str(carla_port)
    if routes:
        env["ROUTES"] = routes
    if device:
        env["COSIM_DEVICE"] = device
    if dynamics_source:
        env["COSIM_DYNAMICS_SOURCE"] = dynamics_source

    returncode = _run_script(["bash", script_path], env, subdir)
    if returncode != 0:
        raise RuntimeError(f"run_leaderboard.sh failed (exit {returncode}) for route {route_id}; see {subdir}")
    return _read_leaderboard_result(out_path, route_id)


# Crash statuses CaRL's own evaluate_routes_slurm.py resubmits on (leaderboard
# process/simulator died mid-route) -- as opposed to a legitimate bad-driving
# outcome ("Failed - Agent deviated from the route", "Failed - Collision", ...
# any status not in this set), which is real data, not a failure to retry.
CARLA_CRASH_STATUSES = frozenset(
    {
        "Failed - Agent couldn't be set up",
        "Failed",
        "Failed - Simulation crashed",
        "Failed - Agent crashed",
    }
)
CARLA_MAX_ATTEMPTS = 3  # matches CaRL's PlanT evaluate_routes_slurm.py default; CARLA's own script uses 5


def _read_leaderboard_result(out_path, route_id):
    """CARLA leaderboard's own result.json schema (leaderboard_evaluator.py's
    StatisticsManager): _checkpoint.records[i].scores.{score_composed,
    score_route, score_penalty} (driving score / route completion % /
    infraction-penalty multiplier), .status, .infractions."""
    if not os.path.isfile(out_path):
        raise RuntimeError(f"run_leaderboard.sh did not write a result at {out_path}")
    with open(out_path) as result_file:
        data = json.load(result_file)
    records = data.get("_checkpoint", {}).get("records", [])
    if not records:
        raise RuntimeError(f"run_leaderboard.sh result at {out_path} has no route records")
    crashed = [r for r in records if r.get("status") in CARLA_CRASH_STATUSES]
    if crashed:
        raise RuntimeError(f"CARLA route {route_id} crashed ({crashed[0].get('status')}); see {out_path}")
    rows = []
    for record in records:
        scores = record.get("scores", {})
        infractions = record.get("infractions", {})
        rows.append(
            {
                "route_id": record.get("route_id", route_id),
                "status": record.get("status"),
                "driving_score": scores.get("score_composed"),
                "route_completion": scores.get("score_route"),
                "infraction_penalty": scores.get("score_penalty"),
                "num_infractions": sum(len(v) for v in infractions.values()) if isinstance(infractions, dict) else None,
            }
        )
    return rows


def _submit_carla_route_entry(benchmark, checkpoint_path, output_dir, executor, job_idx, route_id, attempt):
    subdir = os.path.join(output_dir, f"route_{route_id:03d}")
    if attempt > 1:
        subdir += f"_attempt{attempt}"
    # Unique port per (route, attempt) pair -- a crashed attempt's CARLA server
    # may still be tearing down when we resubmit, so never reuse its port.
    carla_port = benchmark["base_carla_port"] + (job_idx * CARLA_MAX_ATTEMPTS + (attempt - 1)) * CARLA_PORT_STRIDE
    with _clean_slurm_env():
        job = executor.submit(
            _run_carla_route_job,
            subdir,
            checkpoint_path,
            CARLA_LEADERBOARD_SCRIPT,
            benchmark["routes"],
            route_id,
            carla_port,
            benchmark["device"],
            benchmark["dynamics_source"],
        )
    return {
        "job": job,
        "route_id": route_id,
        "job_idx": job_idx,
        "attempt": attempt,
        "benchmark": benchmark,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "executor": executor,
    }


def _submit_carla_jobs(benchmark, checkpoint_path, output_dir, executor):
    entries = [
        _submit_carla_route_entry(benchmark, checkpoint_path, output_dir, executor, job_idx, route_id, attempt=1)
        for job_idx, route_id in enumerate(benchmark["route_ids"])
    ]
    print(f"[cosim_eval] submitted {len(entries)} CARLA route jobs (run_leaderboard.sh) for benchmark {benchmark['name']}")
    return entries


def _resolve_carla_entries(entries):
    """Check each route's current attempt; on a genuine crash (see
    CARLA_CRASH_STATUSES / a process-level failure in _run_carla_route_job),
    resubmit a fresh job for that route up to CARLA_MAX_ATTEMPTS times --
    mirrors CaRL's own evaluate_routes_slurm.py resubmit-on-crash loop.
    Returns (rows_from_routes_that_finished_this_call, still_running_entries)."""
    rows = []
    still_running = []
    for entry in entries:
        if not entry["job"].done():
            still_running.append(entry)
            continue
        try:
            rows.extend(entry["job"].result())
        except Exception as exc:
            if entry["attempt"] >= CARLA_MAX_ATTEMPTS:
                print(
                    f"[cosim_eval] CARLA route {entry['route_id']} crashed on all {entry['attempt']} "
                    f"attempt(s), giving up: {exc}"
                )
                continue
            print(
                f"[cosim_eval] CARLA route {entry['route_id']} attempt {entry['attempt']} crashed ({exc}); "
                f"resubmitting as attempt {entry['attempt'] + 1}"
            )
            still_running.append(
                _submit_carla_route_entry(
                    entry["benchmark"],
                    entry["checkpoint_path"],
                    entry["output_dir"],
                    entry["executor"],
                    entry["job_idx"],
                    entry["route_id"],
                    entry["attempt"] + 1,
                )
            )
    return rows, still_running


def _run_carla_benchmark(benchmark, checkpoint_path, output_dir, executor):
    entries = _submit_carla_jobs(benchmark, checkpoint_path, output_dir, executor)
    rows = []
    while entries:
        finished_rows, entries = _resolve_carla_entries(entries)
        rows.extend(finished_rows)
        if entries:
            time.sleep(10)
    return rows


# ─── nuPlan: one SLURM job per scenario_filter shard, via run_nuplan_planner.sh
def _run_nuplan_shard_job(subdir, checkpoint_path, script_path, scenario_filter, challenges, nuplan_env, worker,
                           threads_per_node, limit_total_scenarios, debug_bev):
    """Runs inside the SLURM job: run_nuplan_planner.sh loops the given
    CHALLENGES itself against nuPlan's unmodified run_simulation.py, writing
    aggregator_metric/*.parquet (one file per challenge, from nuPlan's own
    weighted-average metric aggregator) under GROUP=subdir."""
    env = os.environ.copy()
    env.update(nuplan_env)
    env["CKPT"] = checkpoint_path
    env["SPLIT"] = scenario_filter
    env["CHALLENGES"] = " ".join(challenges)
    env["GROUP"] = subdir
    env["COSIM_DEBUG_BEV"] = "1" if debug_bev else "0"
    if worker:
        env["WORKER"] = worker
    if threads_per_node:
        env["THREADS_PER_NODE"] = str(threads_per_node)
    if limit_total_scenarios:
        env["LIMIT_TOTAL_SCENARIOS"] = str(limit_total_scenarios)

    returncode = _run_script(["bash", script_path], env, subdir)
    try:
        rows = _read_nuplan_aggregator_scores(subdir, scenario_filter)
    except RuntimeError:
        if returncode != 0:
            raise RuntimeError(
                f"run_nuplan_planner.sh failed (exit {returncode}) for scenario_filter={scenario_filter}; see {subdir}"
            )
        raise
    if returncode != 0:
        print(
            f"[cosim_eval] run_nuplan_planner.sh exited {returncode} for scenario_filter={scenario_filter}, but "
            f"aggregator_metric data was still produced and will be used (a later, non-scoring step -- e.g. "
            f"carl_nuplan's csv_main_callback -- likely failed after scoring completed); see {subdir}"
        )
    return rows


def _read_nuplan_aggregator_scores(group_dir, scenario_filter):
    """Per-scenario 'score' rows from nuPlan's own aggregator parquet output
    (nuplan/planning/metrics/aggregator/weighted_average_metric_aggregator.py);
    drops the synthetic 'final_score' row -- this module recomputes the mean
    itself so it uses the same reduction as the native benchmark reports.
    run_nuplan_planner.sh writes these under
    group_dir/simulation/<challenge>/<run_timestamp>/aggregator_metric/ (one
    subtree per challenge, per nuPlan's own run_simulation.py layout) --
    NOT directly under group_dir -- so this walks that structure rather
    than assuming a flat aggregator_metric/ next to group_dir."""
    simulation_dir = os.path.join(group_dir, "simulation")
    aggregator_dirs = []
    for challenge_name in sorted(os.listdir(simulation_dir)) if os.path.isdir(simulation_dir) else []:
        challenge_dir = os.path.join(simulation_dir, challenge_name)
        if not os.path.isdir(challenge_dir):
            continue
        for run_timestamp in sorted(os.listdir(challenge_dir)):
            candidate = os.path.join(challenge_dir, run_timestamp, "aggregator_metric")
            if os.path.isdir(candidate):
                aggregator_dirs.append((challenge_name, candidate))
    if not aggregator_dirs:
        raise RuntimeError(f"nuPlan produced no aggregator_metric output under {group_dir}")

    frames = []
    for challenge_name, aggregator_dir in aggregator_dirs:
        for filename in os.listdir(aggregator_dir):
            if not filename.endswith(".parquet"):
                continue
            df = pd.read_parquet(os.path.join(aggregator_dir, filename))
            df = df[df["scenario"] != "final_score"]
            df["scenario_filter"] = scenario_filter
            df["challenge"] = challenge_name
            frames.append(df)
    if not frames:
        raise RuntimeError(f"nuPlan aggregator_metric directories have no parquet files under {group_dir}")
    return pd.concat(frames, ignore_index=True).to_dict(orient="records")


def _submit_nuplan_jobs(benchmark, checkpoint_path, output_dir, executor):
    jobs = []
    with _clean_slurm_env():
        for scenario_filter in benchmark["scenario_filters"]:
            subdir = os.path.join(output_dir, scenario_filter)
            jobs.append(
                executor.submit(
                    _run_nuplan_shard_job,
                    subdir,
                    checkpoint_path,
                    NUPLAN_PLANNER_SCRIPT,
                    scenario_filter,
                    benchmark["challenges"],
                    benchmark["nuplan_env"],
                    benchmark["worker"],
                    benchmark["threads_per_node"],
                    benchmark["limit_total_scenarios"],
                    benchmark["debug_bev"],
                )
            )
    print(
        f"[cosim_eval] submitted {len(jobs)} nuPlan shard jobs (run_nuplan_planner.sh) for benchmark {benchmark['name']}"
    )
    return jobs


def _run_nuplan_benchmark(benchmark, checkpoint_path, output_dir, executor):
    jobs = _submit_nuplan_jobs(benchmark, checkpoint_path, output_dir, executor)
    rows = []
    for job in jobs:
        rows.extend(job.result())
    return rows


# ─── result aggregation + wandb ──────────────────────────────────────────────
def _summarize_rows(rows):
    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include=[np.number])
    metrics_mean = {column: float(numeric[column].dropna().mean()) for column in numeric.columns}
    summary = {"num_scenarios": len(df), "num_episodes": len(df), "metrics_mean": metrics_mean}
    return df, summary


def _write_cosim_report(rows, output_dir):
    if not rows:
        print("No co-sim evaluation episodes were recorded; skipping report.")
        return None
    df, summary = _summarize_rows(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "episode_metrics.csv"), index=False)
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Wrote {len(df)} co-sim episode rows and summary to {output_dir}")
    return summary


def _resolve_wandb_run_id(checkpoint_path):
    """A checkpoint lives at <data_dir>/<env>_<wandb_run_id>/models/*.pt; recover
    the run id from that directory name so co-sim results resume onto the same
    training run instead of a disconnected one. None if it doesn't parse."""
    run_dir_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))))
    if "_" not in run_dir_name:
        return None
    return run_dir_name.rsplit("_", 1)[-1]


def _log_cosim_to_wandb(args, checkpoint_path, benchmark_name, summary):
    if not args.get("wandb") or summary is None:
        return
    import wandb

    metrics = {"num_scenarios": summary["num_scenarios"], "num_episodes": summary["num_episodes"]}
    metrics.update(summary["metrics_mean"])
    metrics = {f"cosim_{benchmark_name}/{key}": value for key, value in metrics.items()}

    run_id = _resolve_wandb_run_id(checkpoint_path)
    try:
        if run_id is None:
            raise RuntimeError("no run id parsed from checkpoint path")
        wandb.init(id=run_id, project=args["wandb_project"], resume="must", settings=wandb.Settings(console="off"))
    except Exception as exc:
        print(f"[cosim_eval] could not resume training run's wandb run ({exc}); logging a new run instead")
        wandb.init(
            project=args["wandb_project"],
            group=args["wandb_group"],
            job_type="cosim_eval",
            name=f"cosim_eval_{benchmark_name}",
            settings=wandb.Settings(console="off"),
        )
    wandb.log(metrics)
    wandb.finish()
    print(f"[cosim_eval] logged {len(metrics)} metrics to wandb run {wandb.run.id if wandb.run else '?'}")


def _prepare_cosim_submission(benchmark, base_args, output_dir):
    checkpoint_path = base_args.get("load_model_path")
    if not isinstance(checkpoint_path, str) or not os.path.isfile(checkpoint_path):
        raise pufferlib.APIUsageError(
            f"Cosim benchmark {benchmark['name']} requires a checkpoint on disk (eval.load_model_path); "
            "co-sim runs launch as separate SLURM jobs that load the policy from a file, not the live "
            "in-process training policy, so use_training_config is not supported for cosim benchmarks"
        )
    compute_config = _load_compute_config(benchmark.get("compute_config"))
    executor = _build_executor(compute_config, os.path.join(output_dir, "submitit"), job_name=benchmark["name"])
    return checkpoint_path, executor


def _run_cosim_benchmark_job(benchmark, base_args, output_dir):
    """Runs inside its own detached SLURM job (an orchestrator -- it submits
    and waits on the real CARLA/nuPlan work, it doesn't do that work itself):
    blocks until every route/shard job finishes (including CARLA's
    retry-on-crash), then logs to wandb. Because this orchestrator is its own
    SLURM job with its own time budget, wandb gets the result whenever the
    real work finishes -- even long after the training process that
    requested it has exited."""
    return run_cosim_benchmark(benchmark, base_args, output_dir, log_to_wandb=True)


def submit_cosim_benchmark_async(benchmark, base_args, output_dir, orchestrator_time_minutes=360):
    """Fire-and-forget for the training-loop debug hook: submit ONE small
    orchestrator job (see _run_cosim_benchmark_job) and return immediately.
    Unlike submit_cosim_benchmark()+collect_cosim_results(), the caller does
    not need to still be running to see the result reach wandb -- the
    orchestrator does that itself, independently. The orchestrator needs no
    GPU (it only submits/waits on the real jobs), just a generous time
    budget to cover CARLA/nuPlan's queue wait + runtime."""
    checkpoint_path = base_args.get("load_model_path")
    if not isinstance(checkpoint_path, str) or not os.path.isfile(checkpoint_path):
        raise pufferlib.APIUsageError(
            f"Cosim benchmark {benchmark['name']} requires a checkpoint on disk (eval.load_model_path)"
        )
    orchestrator_config = dict(_load_compute_config(benchmark.get("compute_config")))
    orchestrator_config.update(cpus=2, mem="8gb", gpus=None, gpu_type=None, time=orchestrator_time_minutes)
    executor = _build_executor(
        orchestrator_config, os.path.join(output_dir, "submitit_orchestrator"), job_name=f"{benchmark['name']}_log"
    )
    with _clean_slurm_and_wandb_env():
        job = executor.submit(_run_cosim_benchmark_job, benchmark, base_args, output_dir)
    print(
        f"[cosim_eval] submitted orchestrator for benchmark {benchmark['name']}: will submit the real CARLA/nuPlan "
        f"jobs and log to wandb once they finish, independent of this process; see {output_dir}"
    )
    return job


def run_cosim_benchmark(benchmark, base_args, output_dir, log_to_wandb=True):
    """Submit, wait for, and aggregate a carla_cosim or nuplan_cosim benchmark.
    Returns {"episodes": [...], "summary": {...}} -- same shape pufferl.py's
    eval() already expects from the native gigaflow/replay benchmarks. This
    blocks until every job finishes -- use submit_cosim_benchmark instead for
    a fire-and-forget launch (e.g. from inside the training loop).

    log_to_wandb=False when the caller already owns an open wandb run and
    will log the summary itself -- this function's own wandb.init/finish
    would otherwise tear down that already-active run."""
    checkpoint_path, executor = _prepare_cosim_submission(benchmark, base_args, output_dir)

    if benchmark["simulation_mode"] == "carla_cosim":
        rows = _run_carla_benchmark(benchmark, checkpoint_path, output_dir, executor)
    else:
        rows = _run_nuplan_benchmark(benchmark, checkpoint_path, output_dir, executor)

    summary = _write_cosim_report(rows, output_dir)
    if log_to_wandb:
        _log_cosim_to_wandb(base_args, checkpoint_path, benchmark["name"], summary)
    return {"episodes": rows, "summary": summary}
