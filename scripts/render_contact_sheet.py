"""Render a ReGentS run's scenarios as a scannable contact sheet.

The replay gallery shows one scenario at a time behind a dropdown, so finding the
few worth watching means clicking through every entry. This draws all of them at
once: one thumbnail per scenario with the ego, the selected adversary's logged and
adversarial paths, and the surrounding traffic, over an outcome badge.

    python scripts/render_contact_sheet.py experiments/regents/regents_nuplan

Thumbnails come from the saved npz artifacts, so no Drive instance is built and no
map is loaded. Background traffic is drawn faintly, which traces the road layout
without needing the map geometry. Each card links to its interactive replay.
"""

import argparse
import base64
import html
import io
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ_DIR_NAME = "npz"
REPLAY_DIR_NAME = "rendered_replays"
SHEET_FILE_NAME = "contact_sheet.html"

# Drive pins the ego to stable agent row zero and the exporter preserves that order.
EGO_ROW = 0
THUMB = 200
THUMB_PAD = 10
# A scenario whose ego never exceeds this is parked: its "collision" is a parked-car hit.
STATIONARY_SPEED_MPS = 0.5
# Traffic is context, so thin it; the adversary and ego keep every point.
TRAFFIC_STRIDE = 3
MAX_TRAFFIC_PATHS = 90
# Context traffic is restricted to agents that actually travel. Pedestrians, cyclists and
# parked cars sit off the drivable surface by rights, and drawing them over the field
# makes the off-road area look like where the driving happens.
TRAFFIC_MIN_TRAVEL_METERS = 5.0
# The field is one greyscale image per thumbnail: white is drivable, black is off-road.
# The same image serves both themes; inverting it per theme would reverse that meaning.
FIELD_PIXELS = 200

STYLE = """
:root{color-scheme:light;
 --ground:#f5f7f8;--surface:#fff;--surface-2:#edf1f3;--line:#dce3e8;--line-soft:#e8edf1;
 --ink:#12161c;--ink-2:#57616d;--ink-3:#8a949f;--traffic:#b9c3cc;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
 --sans:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{
 --ground:#131619;--surface:#1a1e23;--surface-2:#21262c;--line:#2c333a;--line-soft:#242a30;
 --ink:#f1f4f6;--ink-2:#a8b3be;--ink-3:#79838f;--traffic:#414c57;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.5;
 padding:30px 22px 60px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1280px;margin:0 auto}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--ink-3);margin:0}
h1{font-size:29px;font-weight:600;letter-spacing:-.02em;margin:6px 0 0}
.note{font-size:13.5px;color:var(--ink-2);margin:10px 0 0;max-width:74ch}
.bar{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:22px 0 6px}
button{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
 color:var(--ink-2);background:var(--surface);border:1px solid var(--line);border-radius:20px;
 padding:6px 13px;cursor:pointer}
button:hover{color:var(--ink)}
button[aria-pressed=true]{background:var(--ink);color:var(--ground);border-color:var(--ink)}
button:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.count{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);margin-left:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(214px,1fr));gap:15px;margin-top:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:7px;overflow:hidden;
 text-decoration:none;color:inherit;display:flex;flex-direction:column}
.card:hover{border-color:var(--ink-3)}
.card:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.card svg{display:block;width:100%;height:auto;background:var(--surface-2)}
.field{opacity:.92}
.meta{padding:9px 11px 11px;display:flex;flex-direction:column;gap:6px}
.idline{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.idline .id{font-family:var(--mono);font-size:12.5px;font-weight:500}
.idline .t{font-family:var(--mono);font-size:10.5px;color:var(--ink-3)}
.badges{display:flex;gap:5px;flex-wrap:wrap}
.b{font-family:var(--mono);font-size:9.5px;letter-spacing:.05em;text-transform:uppercase;
 border:1px solid var(--line);border-radius:20px;padding:1px 8px;color:var(--ink-3);white-space:nowrap}
.b.hit{border-color:var(--s3);color:var(--s3)}
.b.warn{border-color:var(--s2);color:var(--s2)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;
 color:var(--ink-2);margin-top:14px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.k{width:20px;height:0;border-top-width:2.5px;border-top-style:solid;flex:none}
.empty{font-size:13px;color:var(--ink-3);padding:30px 0}
footer{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);line-height:1.7;margin-top:30px}
"""

SCRIPT = """
const CARDS = Array.from(document.querySelectorAll(".card"));
const COUNT = document.getElementById("count");
const FILTERS = Array.from(document.querySelectorAll("[data-filter]"));
function apply(name) {
  let shown = 0;
  for (const card of CARDS) {
    const ok = name === "all" || card.dataset[name] === "true";
    card.style.display = ok ? "" : "none";
    shown += ok ? 1 : 0;
  }
  COUNT.textContent = `${shown} of ${CARDS.length} scenarios`;
  for (const b of FILTERS) b.setAttribute("aria-pressed", String(b.dataset.filter === name));
  document.getElementById("empty").hidden = shown > 0;
}
for (const b of FILTERS) b.addEventListener("click", () => apply(b.dataset.filter));
apply("interesting");
"""


def _thumbnail_extent(adversarial, valid):
    """Square world window a thumbnail covers: centre in metres and half-width."""
    live = valid[:, : adversarial.shape[1]]
    points = adversarial[:, :, :2][live]
    if points.size == 0:
        return None
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
    half = 0.5 * span * THUMB / (THUMB - 2 * THUMB_PAD)
    return float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0), half


def _travelled_metres(states, live, row):
    """Net displacement of one agent over its valid steps."""
    track = states[row, :, :2][live[row]]
    if len(track) < 2:
        return 0.0
    return float(np.linalg.norm(track[-1] - track[0]))


def _thumbnail(logged, adversarial, valid, ego_row, adversary_row, collision_t, field_png=None):
    """Draw one scenario: the potential field, faint traffic, then ego and adversary."""
    live = valid[:, : adversarial.shape[1]]
    extent = _thumbnail_extent(adversarial, valid)
    if extent is None:
        return '<svg viewBox="0 0 200 200"></svg>'
    centre_x, centre_y, half = extent
    scale = (THUMB / 2) / half

    def project(xy):
        x = THUMB / 2 + (xy[..., 0] - centre_x) * scale
        y = THUMB / 2 - (xy[..., 1] - centre_y) * scale  # world y up, SVG y down
        return x, y

    def path(states, row, stride=1):
        mask = live[row][::stride]
        xs, ys = project(states[row, ::stride][:, :2])
        pts = [f"{x:.1f},{y:.1f}" for x, y, keep in zip(xs, ys, mask) if keep]
        return "M" + "L".join(pts) if len(pts) > 1 else ""

    parts = []
    if field_png is not None:
        parts.append(
            f'<image class="field" href="data:image/png;base64,{field_png}" x="0" y="0"'
            f' width="{THUMB}" height="{THUMB}" preserveAspectRatio="none"/>'
        )
    traffic = [
        row
        for row in range(adversarial.shape[0])
        if row not in (ego_row, adversary_row)
        and _travelled_metres(adversarial, live, row) >= TRAFFIC_MIN_TRAVEL_METERS
    ]
    for row in traffic[:MAX_TRAFFIC_PATHS]:
        d = path(adversarial, row, TRAFFIC_STRIDE)
        if d:
            parts.append(f'<path d="{d}" fill="none" stroke="var(--traffic)" stroke-width="1" opacity=".55"/>')
    if adversary_row >= 0:
        for states, colour, dash in (
            (logged, "var(--ink-3)", ' stroke-dasharray="3 3"'),
            (adversarial, "var(--s2)", ""),
        ):
            d = path(states, adversary_row)
            if d:
                parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2"{dash}/>')
    ego_x, ego_y = project(adversarial[ego_row, :, :2])
    ego_d = path(adversarial, ego_row)
    if ego_d:
        parts.append(f'<path d="{ego_d}" fill="none" stroke="var(--s1)" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{ego_x[0]:.1f}" cy="{ego_y[0]:.1f}" r="3.5" fill="var(--s1)"/>')
    if collision_t is not None and adversary_row >= 0 and collision_t < adversarial.shape[1]:
        cx, cy = project(adversarial[ego_row, collision_t, :2])
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="none" stroke="var(--s2)" stroke-width="1.6"/>')
    return f'<svg viewBox="0 0 {THUMB} {THUMB}" role="img" aria-hidden="true">{"".join(parts)}</svg>'


def _field_thumbnail(task):
    """Rebuild one scenario's out-of-bounds potential and crop it to the thumbnail window.

    Runs in a spawned worker: Drive and Torch are imported here, not at module load, so
    the sheet costs nothing when fields are not requested.
    """
    from PIL import Image
    import torch

    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.regents.adapter import export_drive_scenarios
    from pufferlib.ocean.regents.losses import ReGentSCostConfig, prepare_out_of_bounds_rasters

    torch.set_num_threads(1)
    artifact_path, resolution_meters = task
    with np.load(artifact_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json_utf8"].tobytes().decode())
        adversarial = archive["c_states"][0]
        valid = archive["c_state_valid"][0]
    scenario_idx = int(Path(artifact_path).name.split("_")[1].split(".")[0])
    extent = _thumbnail_extent(adversarial, valid)
    if extent is None:
        return scenario_idx, None
    centre_x, centre_y, half = extent

    environment = dict(metadata["source_configuration"]["env"])
    environment["num_maps"] = scenario_idx + 1
    seed = int(metadata["deterministic_seed"])
    drive = Drive(**environment, eval_map_indices=[scenario_idx], eval_scenario_seeds=[seed], seed=seed)
    try:
        drive.reset(seed=seed)
        scenario = export_drive_scenarios(drive, raster_resolution_meters=resolution_meters)
    finally:
        drive.close()
    raster = prepare_out_of_bounds_rasters(scenario.drivable_area_rasters, ReGentSCostConfig())[0]
    potential = raster.potential.numpy()
    transform = raster.transform

    # Top image row is maximum y; sampling clamps to the raster's out-of-bounds frame,
    # so a window reaching past the map reads as off-road rather than wrapping.
    x_meters = np.linspace(centre_x - half, centre_x + half, FIELD_PIXELS)
    y_meters = np.linspace(centre_y + half, centre_y - half, FIELD_PIXELS)
    columns = np.clip(
        np.round((x_meters - transform.origin_x_m) / transform.resolution_meters_per_pixel), 0, transform.width - 1
    ).astype(np.int64)
    rows = np.clip(
        np.round((y_meters - transform.origin_y_m) / transform.resolution_meters_per_pixel), 0, transform.height - 1
    ).astype(np.int64)
    window = np.clip(potential[np.ix_(rows, columns)], 0.0, 1.0)
    grey = ((1.0 - window) * 255.0).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(grey, "L").save(buffer, format="PNG", optimize=True)
    return scenario_idx, base64.b64encode(buffer.getvalue()).decode()


def _render_fields(artifacts, resolution_meters, num_workers):
    """Render every scenario's field thumbnail, in parallel when it is worth it."""
    tasks = [(str(path), resolution_meters) for path in artifacts]
    if num_workers <= 1:
        return dict(_field_thumbnail(task) for task in tasks)
    with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
        return dict(pool.imap_unordered(_field_thumbnail, tasks, chunksize=1))


def _artifact_index(path):
    return int(Path(path).name.split("_")[1].split(".")[0])


def _scenario_card(path, replay_dir, field_png=None):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json_utf8"].tobytes().decode())
        adversarial = archive["c_states"][0]
        valid = archive["c_state_valid"][0]
        logged = archive["original_states"][0]
        agent_id = archive["agent_id"][0]
    replay = metadata["c_replay"]
    optimization = metadata["optimization"]
    scenario_idx = int(path.name.split("_")[1].split(".")[0])

    adversary_id = int(optimization.get("selected_adversary_id", -1))
    matches = np.where(agent_id == adversary_id)[0]
    adversary_row = int(matches[0]) if adversary_id >= 0 and matches.size else -1
    horizon = adversarial.shape[1]
    ego_speed = float(np.abs(adversarial[EGO_ROW, :, 3][valid[EGO_ROW, :horizon]]).max(initial=0.0))
    parked = ego_speed < STATIONARY_SPEED_MPS
    success = bool(replay["success"])
    collision_t = replay.get("first_collision_timestep") if success else None

    badges = []
    if success:
        badges.append(('<span class="b hit">collision t=%s</span>' % collision_t))
    else:
        badges.append('<span class="b">%s</span>' % html.escape(str(replay.get("failure_reason") or "no collision")))
    if parked:
        badges.append('<span class="b warn">ego parked</span>')
    if replay.get("background_collision"):
        badges.append('<span class="b warn">bg collision</span>')
    if replay.get("offroad"):
        badges.append('<span class="b warn">offroad</span>')

    href = f"../{REPLAY_DIR_NAME}/scenario_{scenario_idx:05d}.adversarial.html"
    has_replay = (replay_dir / f"scenario_{scenario_idx:05d}.adversarial.html").exists()
    tag = "a" if has_replay else "div"
    link = f' href="{href}"' if has_replay else ""
    return {
        "index": scenario_idx,
        "success": success,
        "parked": parked,
        "interesting": success and not parked,
        "html": (
            f'<{tag} class="card"{link} data-success="{str(success).lower()}"'
            f' data-parked="{str(parked).lower()}"'
            f' data-interesting="{str(success and not parked).lower()}">'
            f"{_thumbnail(logged, adversarial, valid, EGO_ROW, adversary_row, collision_t, field_png)}"
            f'<div class="meta"><div class="idline"><span class="id">#{scenario_idx:05d}</span>'
            f'<span class="t">ego {ego_speed:.1f} m/s</span></div>'
            f'<div class="badges">{"".join(badges)}</div></div></{tag}>'
        ),
    }


def render_contact_sheet(run_dir, output_path=None, with_field=False, field_resolution=None, num_workers=None):
    """Write one scannable page of thumbnails for every artifact in a run.

    With ``with_field`` each thumbnail is backed by that scenario's own out-of-bounds
    potential, rebuilt at the resolution the run optimized against.
    """
    run_dir = Path(run_dir)
    npz_dir = run_dir / NPZ_DIR_NAME
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"No {NPZ_DIR_NAME}/ directory under {run_dir}")
    artifacts = sorted(npz_dir.glob("scenario_*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"No scenario artifacts in {npz_dir}")

    replay_dir = run_dir / REPLAY_DIR_NAME
    fields = {}
    if with_field:
        if field_resolution is None:
            with np.load(artifacts[0], allow_pickle=False) as archive:
                first_metadata = json.loads(archive["metadata_json_utf8"].tobytes().decode())
            field_resolution = float(first_metadata["source_configuration"]["raster_resolution_meters"])
        if num_workers is None:
            num_workers = min(os.cpu_count() or 1, len(artifacts))
        print(f"Rendering {len(artifacts)} potential fields at {field_resolution} m/px on {num_workers} workers...")
        fields = _render_fields(artifacts, field_resolution, num_workers)
    cards = [_scenario_card(path, replay_dir, fields.get(_artifact_index(path))) for path in artifacts]
    total = len(cards)
    hits = sum(card["success"] for card in cards)
    parked = sum(card["parked"] for card in cards)
    interesting = sum(card["interesting"] for card in cards)

    field_note = (
        f" Behind each path is that scenario's own Gaussian out-of-bounds potential at"
        f" {field_resolution} m per pixel: <b>white is drivable road, black is off-road</b>, in both"
        f" light and dark themes. It looks almost binary because the &sigma;&nbsp;=&nbsp;0.5&nbsp;m kernel"
        f" makes the graded band only about 1.5&nbsp;m wide &mdash; a pixel or two at this zoom &mdash;"
        f" and that band is the whole gradient the drivable-area term descends."
        if with_field
        else " Run with <code>--with-field</code> to draw each scenario's out-of-bounds potential behind it."
    )
    legend = "".join(
        f'<span><i class="k" style="border-top-color:{colour};border-top-style:{style}"></i>{label}</span>'
        for colour, label, style in (
            ("var(--s1)", "Ego", "solid"),
            ("var(--s2)", "Adversary, generated", "solid"),
            ("var(--ink-3)", "Adversary, logged", "dashed"),
            ("var(--traffic)", "Other traffic", "solid"),
        )
    )
    if with_field:
        legend += (
            '<span><i style="width:20px;height:11px;border-radius:2px;flex:none;'
            'background:linear-gradient(90deg,#fff,#000);border:1px solid var(--line)"></i>'
            "white drivable &rarr; black off-road</span>"
        )
    buttons = "".join(
        f'<button data-filter="{key}" aria-pressed="false">{label}</button>'
        for key, label in (
            ("interesting", f"Moving ego + collision ({interesting})"),
            ("success", f"Collision ({hits})"),
            ("parked", f"Parked ego ({parked})"),
            ("all", f"All ({total})"),
        )
    )
    body = f"""<div class="wrap">
  <p class="eyebrow">ReGentS &middot; {html.escape(run_dir.name)}</p>
  <h1>Scenario contact sheet</h1>
  <p class="note">Every generated scenario at a glance. Each thumbnail is the C replay seen from
    above, autoscaled to its own extent; the faint lines are the other traffic, which traces the road
    layout.{field_note} A ring marks where the adversary reaches the ego. Click a card for its
    interactive replay.</p>
  <div class="legend">{legend}</div>
  <div class="bar">{buttons}<span class="count" id="count"></span></div>
  <div class="grid">{"".join(card["html"] for card in cards)}</div>
  <p class="empty" id="empty" hidden>No scenarios match that filter.</p>
  <footer>{total} scenarios &middot; {hits} with an ego collision &middot; {parked} with a parked ego
    (peak speed below {STATIONARY_SPEED_MPS} m/s) &middot; {interesting} with both a moving ego and a collision<br>
    drawn from {NPZ_DIR_NAME}/scenario_*.npz &middot; no map geometry loaded</footer></div>"""
    page = (
        "<!doctype html><meta charset=utf-8><title>Scenario contact sheet</title>"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
        f"<style>{STYLE}</style>{body}<script>{SCRIPT}</script>"
    )
    output_path = Path(output_path) if output_path is not None else run_dir / REPLAY_DIR_NAME / SHEET_FILE_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Generation output directory containing npz/")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Where to write the sheet (default: <run_dir>/{REPLAY_DIR_NAME}/{SHEET_FILE_NAME})",
    )
    parser.add_argument(
        "--with-field",
        action="store_true",
        help="Draw each scenario's out-of-bounds potential behind its thumbnail",
    )
    parser.add_argument(
        "--field-resolution",
        type=float,
        default=None,
        help="Raster resolution in m/px (default: whatever the run optimized against)",
    )
    parser.add_argument("--workers", type=int, default=None, help="Field-rendering workers (default: all cores)")
    args = parser.parse_args()
    sheet = render_contact_sheet(args.run_dir, args.output, args.with_field, args.field_resolution, args.workers)
    print(f"Contact sheet: {sheet}")


if __name__ == "__main__":
    main()
