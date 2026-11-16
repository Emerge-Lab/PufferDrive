"""Run CaRL's leaderboard_evaluator.py for one model directory, with all outputs under its eval folder.

usage: run_evaluator.py --model-dir <dir> [evaluator args...]

<dir> holds final_model.pt + config.yaml (blank -> DEFAULT_MODEL_DIR). The launcher adds
--agent-config <dir>/final_model.pt and --checkpoint <dir>/eval/carla_leaderboard/result.json,
creates that directory (the evaluator opens the result file before any agent code runs), and
points COSIM_TELEMETRY / COSIM_WORLD_LOG / COSIM_OBS_HTML / COSIM_DEBUG_CARLA_VIEW into it unless
they are already set. Remaining arguments go to the evaluator, located through CARL_WORK_DIR.
"""

import os
import runpy
import sys
from pathlib import Path


DEFAULT_MODEL_DIR = "experiments/k_scaled_0036_1000"
EVAL_SUBDIR = "eval/carla_leaderboard"
OUTPUT_ENV_DIRS = {
    "COSIM_TELEMETRY": "telemetry",
    "COSIM_WORLD_LOG": "world_log",
    "COSIM_OBS_HTML": "obs_html",
    "COSIM_DEBUG_CARLA_VIEW": "carla_view",
}


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
    eval_dir = model_dir / EVAL_SUBDIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    if "--agent-config" not in argv:
        argv += ["--agent-config", str(checkpoint)]
    if "--checkpoint" not in argv:
        argv += ["--checkpoint", str(eval_dir / "result.json")]
    for env_name, subdir in OUTPUT_ENV_DIRS.items():
        if env_name not in os.environ:
            os.environ[env_name] = str(eval_dir / subdir)
    carl_work_dir = os.environ.get("CARL_WORK_DIR")
    if not carl_work_dir:
        raise SystemExit("run_evaluator.py: CARL_WORK_DIR is not set")
    evaluator = (
        Path(carl_work_dir) / "original_leaderboard" / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
    )
    if not evaluator.is_file():
        raise SystemExit(f"run_evaluator.py: evaluator not found at {evaluator}")
    print(f"[run_evaluator] model {model_dir} -> outputs {eval_dir}")
    sys.argv = [str(evaluator)] + argv
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
