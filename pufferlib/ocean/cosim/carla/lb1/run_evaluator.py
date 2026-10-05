"""Run carla_garage's leaderboard_evaluator_local.py (Leaderboard 1.0, CARLA 0.9.10) for one model directory,
with all outputs under its eval folder; the Leaderboard 1.0 twin of ../run_evaluator.py. Runs in the `garage`
conda env (Python 3.7), so nothing from pufferlib is imported here.

usage: run_evaluator.py --model-dir <dir> [evaluator args...]

<dir> holds final_model.pt + config.yaml (blank -> DEFAULT_MODEL_DIR). The launcher adds --agent-config
<dir>/final_model.pt, --checkpoint <dir>/eval/carla_leaderboard_v1/result.json (the evaluator opens the
result file before any agent code runs) and --scenarios eval_scenarios.json, points COSIM_TELEMETRY /
COSIM_WORLD_LOG / COSIM_OBS_HTML / COSIM_DEBUG_CARLA_VIEW into the eval folder unless already set, and
exports what the garage evaluator reads from the environment: ROUTES (the --routes file, it names the
route records after it) and BENCHMARK=longest6 (all spawn points of background traffic, stop-sign
penalty 1.0). The evaluator is located through GARAGE_WORK_DIR.
"""

import os
import runpy
import sys
from pathlib import Path


DEFAULT_MODEL_DIR = "experiments/k_scaled_0040_1000"
EVAL_SUBDIR = "eval/carla_leaderboard_v1"
DEFAULT_BENCHMARK = "longest6"
DEFAULT_SCENARIOS = "leaderboard/data/scenarios/eval_scenarios.json"
OUTPUT_ENV_DIRS = {
    "COSIM_TELEMETRY": "telemetry",
    "COSIM_WORLD_LOG": "world_log",
    "COSIM_OBS_HTML": "obs_html",
    "COSIM_DEBUG_CARLA_VIEW": "carla_view",
}


def option_value(argv, name):
    """Value of `--name value` or `--name=value` in argv, else None."""
    for k, arg in enumerate(argv):
        if arg == name and k + 1 < len(argv):
            return argv[k + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def main():
    argv = sys.argv[1:]
    model_dir = ""
    if "--model-dir" in argv:
        k = argv.index("--model-dir")
        has_value = k + 1 < len(argv) and not argv[k + 1].startswith("--")
        model_dir = argv[k + 1] if has_value else ""
        del argv[k : k + 2 if has_value else k + 1]
    if not model_dir:
        model_dir = DEFAULT_MODEL_DIR
        print(f"[run_evaluator] --model-dir blank, using {DEFAULT_MODEL_DIR}")
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = Path.cwd() / model_dir
    checkpoint = model_dir / "final_model.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"run_evaluator.py: no checkpoint at {checkpoint}")
    garage_work_dir = os.environ.get("GARAGE_WORK_DIR")
    if not garage_work_dir:
        raise SystemExit("run_evaluator.py: GARAGE_WORK_DIR is not set (the carla_garage checkout)")
    evaluator = Path(garage_work_dir) / "leaderboard" / "leaderboard" / "leaderboard_evaluator_local.py"
    if not evaluator.is_file():
        raise SystemExit(f"run_evaluator.py: evaluator not found at {evaluator}")
    routes = option_value(argv, "--routes")
    if routes is None:
        raise SystemExit("run_evaluator.py: --routes <xml> is required")
    eval_dir = model_dir / EVAL_SUBDIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    if option_value(argv, "--agent-config") is None:
        argv += ["--agent-config", str(checkpoint)]
    if option_value(argv, "--checkpoint") is None:
        argv += ["--checkpoint", str(eval_dir / "result.json")]
    if option_value(argv, "--scenarios") is None:
        argv += ["--scenarios", str(Path(garage_work_dir) / DEFAULT_SCENARIOS)]
    for env_name, subdir in OUTPUT_ENV_DIRS.items():
        if env_name not in os.environ:
            os.environ[env_name] = str(eval_dir / subdir)
    os.environ["ROUTES"] = routes
    os.environ.setdefault("BENCHMARK", DEFAULT_BENCHMARK)
    os.environ.setdefault("SCENARIO_RUNNER_ROOT", str(Path(garage_work_dir) / "scenario_runner"))
    print(f"[run_evaluator] model {model_dir} -> outputs {eval_dir} (BENCHMARK={os.environ['BENCHMARK']})")
    sys.argv = [str(evaluator)] + argv
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
