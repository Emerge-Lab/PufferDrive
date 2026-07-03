#!/usr/bin/env python3
"""Re-render failing episodes from an episode-metrics CSV by replaying their seeds.

Stage 2 of the failure-triage pipe:
  1. puffer eval puffer_drive --evaluator validation_gigaflow \\
         --load-model-path <ckpt> --render 0
     -> writes episode_metrics/<evaluator>_epoch{E}_step{S}.csv (one row per
        episode, with the episode's RNG seed in the Seed column)
  2. python scripts/eval/render_failure_seeds.py \\
         --csv <that csv> --load-model-path <ckpt>
     -> filters rows with collision/offroad/red-light infractions and renders
        each one by pinning env.starting_map to the row's map and forcing
        env.episode_seed to the row's Seed. One HTML/mp4 per failure under
        <run_dir>/gif_failures/, suffixed _epoch<csv_row>_step<seed>.

The render env keeps the evaluator's exact env config (agent counts, obs
layout, scenario length), so the inference batch shape matches the metric
pass and the replayed episode is bit-identical to the logged one. Only the
first env slot (the pinned map + forced seed) is rendered.
"""

import argparse
import os
import sys


FAILURE_COLUMNS = ["collision_rate", "offroad_rate", "red_light_violation_rate"]


def load_failures(csv_path, max_failures):
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in FAILURE_COLUMNS + ["Seed", "map_name", "scenario_id"] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV {csv_path} is missing required columns: {missing}")
    mask = (df[FAILURE_COLUMNS] > 0).any(axis=1)
    failures = df[mask]
    if max_failures is not None:
        failures = failures.head(max_failures)
    return failures


# Identity/positional columns whose replay-vs-original diff is meaningless:
# the replay always runs in env slot 0 as episode 0.
NON_COMPARED_COLUMNS = {"Seed", "env_slot", "episode_index"}


def write_replay_diff_csv(source_csv, output_dir, replay_diffs):
    """One row per replayed failure: <metric>_orig / _replay / _diff for every
    numeric metric shared by the source CSV row and the replay's episode
    summary. Lands next to the source CSV so runs can be compared in place."""
    if not replay_diffs:
        return None
    import pandas as pd

    records = []
    for csv_row, row, summary in replay_diffs:
        record = {"csv_row": int(csv_row), "scenario_id": row["scenario_id"], "Seed": int(row["Seed"])}
        for column in row.index:
            original = row[column]
            replayed = summary.get(column)
            if column in NON_COMPARED_COLUMNS or not isinstance(replayed, (int, float)):
                continue
            record[f"{column}_orig"] = float(original)
            record[f"{column}_replay"] = float(replayed)
            record[f"{column}_diff"] = float(replayed) - float(original)
        records.append(record)

    df = pd.DataFrame(records)
    diff_columns = [c for c in df.columns if c.endswith("_diff")]
    exact = int((df[diff_columns] == 0.0).all(axis=1).sum())
    print(f"\nReplay check: {exact}/{len(df)} episode(s) bit-exact vs the source CSV.")
    for _, record in df[~(df[diff_columns] == 0.0).all(axis=1)].iterrows():
        drifted = [c.removesuffix("_diff") for c in diff_columns if record[c] != 0.0]
        print(f"  row {int(record['csv_row'])} ({record['scenario_id']}) drifted on: {', '.join(drifted)}")

    out_path = os.path.join(
        output_dir, "episode_metrics", os.path.basename(source_csv).removesuffix(".csv") + "_replay_diff.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Per-episode metrics CSV with a Seed column")
    parser.add_argument("--load-model-path", required=True, help="Checkpoint the CSV was produced with")
    parser.add_argument("--evaluator", default="validation_gigaflow", help="[eval.<name>] section the CSV came from")
    parser.add_argument("--render-backend", default="obs_html", choices=["obs_html", "triage_html", "egl"])
    parser.add_argument("--max-failures", type=int, default=None, help="Render at most N failures")
    parser.add_argument(
        "--render-max-steps", type=int, default=None, help="Steps to render (default: env scenario_length)"
    )
    parser.add_argument(
        "--output-dir", default=None, help="render_results_dir (default: the CSV's run dir, i.e. its grandparent)"
    )
    cli = parser.parse_args()

    failures = load_failures(cli.csv, cli.max_failures)
    if failures.empty:
        print("No failed episodes found in the CSV. Nothing to render.")
        return
    print(f"Found {len(failures)} failed episode(s):")
    print(failures[["scenario_id", "Seed", "env_slot"] + FAILURE_COLUMNS].to_string())

    output_dir = cli.output_dir or os.path.dirname(os.path.dirname(os.path.abspath(cli.csv)))

    # load_config consumes sys.argv; hand it only pufferl-compatible flags.
    sys.argv = ["render_failure_seeds"]
    from pufferlib.ocean.benchmark.manager import EvalManager
    from pufferlib.pufferl import _merge_checkpoint_arch, load_config, load_env, load_policy

    env_name = "puffer_drive"
    args = load_config(env_name)
    args["load_model_path"] = cli.load_model_path
    _merge_checkpoint_arch(args, cli.load_model_path)
    args["render_results_dir"] = output_dir

    manager = EvalManager.from_config(args)
    target = next((e for e in manager.evaluators if e.name == cli.evaluator), None)
    if target is None:
        raise SystemExit(f"No [eval.{cli.evaluator}] section found. Known: {[e.name for e in manager.evaluators]}")

    target.config["render_backend"] = cli.render_backend
    target.render = True
    env_cfg = target.config.setdefault("env", {})
    eval_cfg = target.config.setdefault("eval", {})
    eval_cfg["render_num_scenarios"] = 1
    scenario_length = int(env_cfg.get("scenario_length") or args["env"]["scenario_length"])
    eval_cfg["render_max_steps"] = int(cli.render_max_steps or scenario_length)

    map_dir = env_cfg.get("map_dir") or args["env"]["map_dir"]
    map_basenames = sorted(f for f in os.listdir(map_dir) if f.endswith(".bin"))

    probe_args = manager._build_eval_args(target, env_name=env_name, global_step=None)
    probe_vec = load_env(env_name, probe_args)
    policy = load_policy(probe_args, probe_vec, env_name)
    probe_vec.close()

    rendered = []
    rendered_rows = []
    replay_diffs = []
    for csv_row, row in failures.iterrows():
        map_basename = os.path.basename(str(row["map_name"]))
        if map_basename not in map_basenames:
            print(f"Skipping row {csv_row}: map {map_basename} not found in {map_dir}")
            continue
        seed = int(row["Seed"])
        env_cfg["starting_map"] = map_basenames.index(map_basename)
        env_cfg["episode_seed"] = seed

        # epoch=csv row / global_step=seed land in the output filename suffix,
        # making each failure's file unique and traceable back to the CSV.
        render_args = manager._build_eval_args(target, env_name=env_name, global_step=seed, epoch=int(csv_row))
        render_args["render_out_dir"] = os.path.join(output_dir, "gif_failures")
        print(f"Rendering row {csv_row}: {row['scenario_id']} seed={seed} (map index {env_cfg['starting_map']})")
        paths = target._render_pass(None, policy, render_args)
        rendered.extend(paths)
        rendered_rows.extend((path, row) for path in paths)
        # The pinned map + forced seed always runs in env slot 0.
        replay_summary = getattr(target, "last_render_summaries", {}).get(0)
        if replay_summary is None:
            print(f"  (no replay summary for row {csv_row} — episode did not complete within render_max_steps)")
        else:
            replay_diffs.append((csv_row, row, replay_summary))

    diff_csv_path = write_replay_diff_csv(cli.csv, output_dir, replay_diffs)

    # Each _render_pass call rebuilds the gallery index with only its own
    # file's metrics; rebuild once at the end with every failure's CSV row
    # (the replay is seed-exact, so the CSV metrics are the render's metrics).
    if len(rendered_rows) > 1 and cli.render_backend != "egl":
        from pufferlib import viz
        from pufferlib.ocean.benchmark.evaluators.base import _GALLERY_METRIC_KEYS

        file_metrics = {
            os.path.basename(str(path)): {k: float(row[k]) for k in _GALLERY_METRIC_KEYS if k in row}
            for path, row in rendered_rows
        }
        viz.build_gallery_index(os.path.dirname(os.path.abspath(str(rendered[0]))), file_metrics=file_metrics)

    print(f"\nDone. {len(rendered)} file(s):")
    for path in rendered:
        print(f"  {path}")
    if diff_csv_path:
        print(f"Replay diff CSV: {diff_csv_path}")


if __name__ == "__main__":
    main()
