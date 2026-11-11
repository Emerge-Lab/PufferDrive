"""Tidy a nuPlan closed-loop run of the PufferDrive planner (the run_simulation.py `group` directory that
scripts/kesai/9_nuPlan.sh writes under <run>/eval) down to what is read after the run:

  overview.csv    CLS, the eight sub-metrics and the loss_* importance of each (CaRL's csv_main_callback)
  scenarios.csv   per-scenario scores, worst first (compare_nuplan_evals.py)
  obs_html/       the observation replay pages render_obs_html.py rendered before this script ran
  simulation_log/<token>.msgpack.xz  nuPlan's simulation logs of the kept scenarios
  details/
    aggregated_metrics.csv  the aggregator's full table: scenario type, comfort sub-metrics, per-type and final rows
    metrics/                nuPlan's per-metric parquets with the per-step series of every scenario
    runner_report.parquet, config.yaml, overrides.yaml
    videos/<token>.mp4      carl_visualization_callback's ground-truth view of the kept scenarios, when it ran

usage: python scripts/eval/tidy_nuplan_run.py <group_dir> [--keep infractions] [--max-score 0.9] [--strip-planner]

--keep selects the scenarios whose simulation log and video survive, with render_obs_html.py's rules: infractions
(an at-fault collision, drivable-area or driving-direction violation, or no progress), failures (score below
--max-score), all, none. Everything else nuPlan wrote is deleted: the other logs and videos, the aggregator parquet,
hydra.yaml, log.txt (hydra's job log, repeated in the SLURM log), the nuBoard descriptor, and every
obs_html/*.replay.zlib whose page exists (the page embeds it).
Videos are re-encoded with ffmpeg (imageio-ffmpeg's binary or the system's; h264, ~1.3 MB each) or, without one,
with OpenCV (mp4v, ~4.4 MB). --strip-planner rewrites the kept logs of runs recorded before the planner stopped
pickling its policy into them (28 MB -> 0.2 MB each; needs the nuPlan devkit env). A run that was never scored
keeps nuPlan's tree, as does a group holding both challenges; an already tidied run is left alone, an interrupted
tidy resumes.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nuplan_run_layout  # noqa: E402
from render_obs_html import FLAG_METRICS, GALLERY_INDEX, REPLAY_SUFFIX, page_token, scenario_rows, selected_tokens  # noqa: E402
from tidy_alpasim_run import remove_empty_dirs  # noqa: E402

KEEP_MODES = ("infractions", "failures", "all", "none")
SIMULATION_LOG_SUFFIX = nuplan_run_layout.SIMULATION_LOG_SUFFIX
RUN_FILES = ("runner_report.parquet",)
HYDRA_FILES = ("config.yaml", "overrides.yaml")
HYDRA_JOB_LOG = "log.txt"
VIDEO_CRF = 23  # h264 quality: an 11 MB MJPEG scenario video -> 1.3 MB with lines and labels still legible
MAX_VIDEO_FRAMES = 100000


def kept_tokens(rows, keep, max_score):
    if keep == "all":
        return set(rows)
    if keep == "none":
        return set()
    if keep == "failures":
        return selected_tokens(rows, max_score, [])
    return selected_tokens(rows, 0.0, [metric for _, metric in FLAG_METRICS])


def move(src, dst):
    """Rename src to dst; a src that an interrupted earlier tidy already moved is skipped."""
    if not src.exists():
        return
    if dst.exists():
        raise SystemExit(f"{dst} already exists, not overwriting it with {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def remove(path):
    """-> bytes freed by deleting the file, 0 when it does not exist."""
    if not path.is_file():
        return 0
    size = path.stat().st_size
    path.unlink()
    return size


def ffmpeg_executable():
    """-> imageio-ffmpeg's bundled ffmpeg or the system's, None when there is neither."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return shutil.which("ffmpeg")


def transcode_video(avi_path, mp4_path, ffmpeg):
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg:
        command = [ffmpeg, "-v", "error", "-y", "-i", str(avi_path), "-c:v", "libx264", "-preset", "fast"]
        command += ["-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p", str(mp4_path)]
        subprocess.run(command, check=True)
        return
    import cv2

    reader = cv2.VideoCapture(str(avi_path))
    size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.get(cv2.CAP_PROP_FPS), size)
    if not writer.isOpened():
        raise SystemExit(f"{mp4_path}: OpenCV could not open the video writer")
    frame_count = 0
    while frame_count < MAX_VIDEO_FRAMES:
        ok, frame = reader.read()
        if not ok:
            break
        writer.write(frame)
        frame_count += 1
    writer.release()
    reader.release()


def strip_planner_from_log(log_path, target):
    """Re-save a log recorded before the planner dropped its policy from its pickled state. -> bytes freed."""
    import nuplan_log_remap  # remaps the scenario paths of logs written on another machine

    log = nuplan_log_remap.load_log(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    log.file_path = target
    log.save_to_file()
    freed = log_path.stat().st_size - target.stat().st_size
    log_path.unlink()
    return freed


def tidy_summaries(sim_dir, group_dir, details_dir):
    """The aggregator table into details/, overview.csv and scenarios.csv to the top of the run. -> bytes freed."""
    aggregator_dir = sim_dir / "aggregator_metric"
    for csv_path in aggregator_dir.glob("*_weighted_average_metrics_*.csv"):
        move(csv_path, details_dir / nuplan_run_layout.AGGREGATED_METRICS_CSV)
    move(aggregator_dir / "overview.csv", group_dir / "overview.csv")
    move(aggregator_dir / "scenarios.csv", group_dir / "scenarios.csv")
    return sum(remove(path) for path in aggregator_dir.glob("*.parquet"))


def tidy_run_files(sim_dir, details_dir):
    """metrics/, the runner report and the resolved hydra config into details/. -> bytes freed."""
    move(sim_dir / "metrics", details_dir / "metrics")
    for name in RUN_FILES:
        move(sim_dir / name, details_dir / name)
    for name in HYDRA_FILES:
        move(sim_dir / "code" / "hydra" / name, details_dir / name)
    freed_bytes = remove(sim_dir / "code" / "hydra" / "hydra.yaml") + remove(sim_dir / HYDRA_JOB_LOG)
    return freed_bytes + sum(remove(path) for path in sim_dir.glob("*.nuboard"))


def tidy_simulation_logs(sim_dir, log_dir, kept, strip_planner):
    """-> (logs deleted, bytes freed, logs kept as <log_dir>/<token>.msgpack.xz)."""
    deleted_count = freed_bytes = kept_count = 0
    for log_path in sorted((sim_dir / nuplan_run_layout.SIMULATION_LOG_DIR).rglob(f"*{SIMULATION_LOG_SUFFIX}")):
        token = log_path.name[: -len(SIMULATION_LOG_SUFFIX)]
        target = log_dir / log_path.name
        if token not in kept:
            freed_bytes += remove(log_path)
            deleted_count += 1
        elif strip_planner:
            freed_bytes += strip_planner_from_log(log_path, target)
            kept_count += 1
        else:
            move(log_path, target)
            kept_count += 1
    return deleted_count, freed_bytes, kept_count


def tidy_videos(sim_dir, details_dir, kept, ffmpeg):
    """-> (videos deleted, bytes freed, videos kept as details/videos/<token>.mp4); nothing without the callback."""
    deleted_count = freed_bytes = kept_count = 0
    for avi_path in sorted((sim_dir / "visualization").glob("*.avi")):
        if avi_path.stem not in kept:
            freed_bytes += remove(avi_path)
            deleted_count += 1
            continue
        mp4_path = details_dir / "videos" / f"{avi_path.stem}.mp4"
        transcode_video(avi_path, mp4_path, ffmpeg)
        freed_bytes += remove(avi_path) - mp4_path.stat().st_size
        kept_count += 1
    return deleted_count, freed_bytes, kept_count


def prune_rendered_replays(obs_dir):
    """-> (replays deleted, bytes freed): obs_html/*.replay.zlib whose rendered page exists (the page embeds it)."""
    pages = {page_token(page) for page in obs_dir.glob("*.html") if page.name != GALLERY_INDEX}
    deleted_count = freed_bytes = 0
    for replay_path in sorted(obs_dir.glob(f"*{REPLAY_SUFFIX}")):
        if replay_path.name[: -len(REPLAY_SUFFIX)] not in pages:
            continue
        freed_bytes += remove(replay_path)
        deleted_count += 1
    return deleted_count, freed_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group_dir")
    ap.add_argument("--keep", choices=KEEP_MODES, default="infractions", help="scenarios whose log and video survive")
    ap.add_argument("--max-score", type=float, default=0.9, help="--keep failures: scenarios scoring below this")
    ap.add_argument("--strip-planner", action="store_true", help="re-save the kept logs without the pickled policy")
    args = ap.parse_args()
    group_dir = Path(args.group_dir).resolve()
    details_dir = group_dir / nuplan_run_layout.DETAILS_DIR
    log_dir = group_dir / nuplan_run_layout.SIMULATION_LOG_DIR
    simulation_root = group_dir / "simulation"
    if details_dir.is_dir() and not simulation_root.exists():
        print(f"[tidy_nuplan_run] already tidied: {group_dir}")
        return
    sim_dir = nuplan_run_layout.raw_simulation_dir(group_dir)
    if sim_dir is None:
        raise SystemExit(f"no simulation/<challenge>/<timestamp> under {group_dir}")
    try:
        nuplan_run_layout.aggregated_metrics_csv(group_dir)
    except FileNotFoundError:
        print(f"[tidy_nuplan_run] {group_dir} was never scored (no aggregator csv): nuPlan's tree kept as is")
        return
    kept = kept_tokens(scenario_rows(group_dir), args.keep, args.max_score)
    ffmpeg = ffmpeg_executable()
    freed_bytes = tidy_summaries(sim_dir, group_dir, details_dir)
    freed_bytes += tidy_run_files(sim_dir, details_dir)
    log_count, log_bytes, kept_logs = tidy_simulation_logs(sim_dir, log_dir, kept, args.strip_planner)
    video_count, video_bytes, kept_videos = tidy_videos(sim_dir, details_dir, kept, ffmpeg)
    replay_count, replay_bytes = prune_rendered_replays(group_dir / "obs_html")
    remove_empty_dirs(simulation_root)
    leftovers = sorted(str(path.relative_to(group_dir)) for path in simulation_root.rglob("*") if path.is_file())
    if leftovers:
        print(f"[tidy_nuplan_run] left in place, unknown: {leftovers}")
    freed_bytes += log_bytes + video_bytes + replay_bytes
    print(
        f"[tidy_nuplan_run] removed {log_count} logs, {video_count} videos and {replay_count} replays "
        f"({freed_bytes / 1e9:.2f} GB freed), kept {kept_logs} logs and {kept_videos} videos "
        f"({args.keep}, {'h264' if ffmpeg else 'mp4v'}) -> {group_dir}"
    )


if __name__ == "__main__":
    main()
