#!/usr/bin/env python3

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run `puffer eval puffer_drive` on the latest model from each run folder."
    )
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders")
    parser.add_argument("--num_scenarios", type=int, default=20, help="Number of scenarios for eval")
    parser.add_argument(
        "--command-prefix",
        default="puffer eval_multi_scenarios_render puffer_drive --eval_simulation gigaflow --render 1 --render_obs 1 --num_carla_maps 2",
        help="Full base command to run before --load-model-path and --num_scenarios are appended",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution when one evaluation fails",
    )
    return parser.parse_args()


def latest_model_in_run(run_dir):
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        return None

    candidates = [path for path in models_dir.glob("*.pt") if path.is_file() and "trainer_state" not in path.name]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def list_run_dirs(runs_dir):
    return sorted([path for path in runs_dir.iterdir() if path.is_dir()])


def build_command(prefix, model_path, num_scenarios):
    return [*shlex.split(prefix), "--load-model-path", str(model_path), "--num_scenarios", str(num_scenarios)]


def move_videos_to_run(run_dir, workspace_dir):
    source = workspace_dir / "videos"
    if not source.is_dir():
        print("  -> no videos directory found to move")
        return

    target = run_dir / "videos"
    if target.exists():
        shutil.rmtree(target)

    shutil.move(str(source), str(target))
    print(f"  -> moved videos to {target}")


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)

    if not runs_dir.is_dir():
        raise SystemExit(f"Error: runs directory not found: {runs_dir}")

    run_dirs = list_run_dirs(runs_dir)
    if not run_dirs:
        raise SystemExit(f"Error: no run directories found in {runs_dir}")

    selected = [(run_dir, latest_model_in_run(run_dir)) for run_dir in run_dirs]

    selected = [(run_dir, model_path) for run_dir, model_path in selected if model_path is not None]

    if not selected:
        raise SystemExit(f"Error: no model files found under {runs_dir}/*/models")

    print(f"Found {len(selected)} runs with models in {runs_dir}\n")

    failures = []
    for run_dir, model_path in selected:
        cmd = build_command(args.command_prefix, model_path, args.num_scenarios)
        cmd_str = " ".join(shlex.quote(part) for part in cmd)
        print(f"[{run_dir.name}] {cmd_str}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=Path.cwd())
        if result.returncode != 0:
            failures.append((run_dir.name, model_path, result.returncode))
            print(f"  -> failed with exit code {result.returncode}\n")
            if args.stop_on_error:
                break
        else:
            move_videos_to_run(run_dir, Path.cwd())
            print("  -> done\n")

    if failures:
        print("Failures:")
        for run_name, model_path, code in failures:
            print(f"- {run_name}: {model_path} (exit={code})")
        raise SystemExit(1)

    print("All evaluations completed.")


if __name__ == "__main__":
    main()
