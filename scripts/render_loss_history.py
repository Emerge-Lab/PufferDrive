"""Render a ReGentS run's loss history CSVs as self-contained HTML.

Generation writes `losses/scenario_XXXXX.losses.csv` with one row per optimizer
iteration. This turns those into an offline report: one page per scenario plus an
index, written to `loss_report/` beside the replay gallery.

    python scripts/render_loss_history.py experiments/regents/regents_nuplan

Every page inlines its own data, so the output directory can be copied anywhere.
"""

import argparse
import csv
import html
import json
from pathlib import Path

from pufferlib.ocean.regents.losses import ReGentSCostConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
LOSS_DIR_NAME = "losses"
REPORT_DIR_NAME = "loss_report"
METRICS_FILE_NAME = "generation_metrics.csv"

# The three cost terms, in the order they are stacked in the report.
TERMS = (
    ("ego_collision_cost", "Ego collision", "ego_collision_weight", "var(--s1)"),
    ("background_collision_cost", "Background collision", "background_collision_weight", "var(--s2)"),
    ("drivable_area_cost", "Drivable area", "drivable_area_weight", "var(--s3)"),
)

PAGE_STYLE = """
:root{color-scheme:light;
 --ground:#f5f7f8;--surface:#fff;--surface-2:#edf1f3;--line:#dce3e8;--line-soft:#e8edf1;
 --ink:#12161c;--ink-2:#57616d;--ink-3:#8a949f;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
 --sans:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{
 --ground:#131619;--surface:#1a1e23;--surface-2:#21262c;--line:#2c333a;--line-soft:#242a30;
 --ink:#f1f4f6;--ink-2:#a8b3be;--ink-3:#79838f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.5;
 padding:30px 20px 52px;-webkit-font-smoothing:antialiased}
.wrap{max-width:960px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
a{color:var(--s1)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--ink-3);margin:0}
h1{font-size:28px;font-weight:600;letter-spacing:-.02em;margin:6px 0 0}
h2{font-size:15px;font-weight:600;margin:0}
.note{font-size:12.5px;color:var(--ink-2);margin:5px 0 0;max-width:72ch}
section{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:17px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(172px,1fr));gap:13px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px}
.tile .k{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--ink-3);display:flex;align-items:center;gap:6px}
.tile .v{font-family:var(--mono);font-size:25px;font-weight:500;margin-top:6px;line-height:1.1;
 font-variant-numeric:tabular-nums}
.tile .n{font-size:12px;color:var(--ink-2);margin-top:5px}
.sw{width:9px;height:9px;border-radius:2px;flex:none}
.multiples{display:grid;grid-template-columns:repeat(auto-fit,minmax(252px,1fr));gap:15px;margin-top:14px}
.panel{border:1px solid var(--line-soft);border-radius:5px;padding:11px 11px 4px}
.panel h3{font-size:12.5px;font-weight:600;margin:0;display:flex;align-items:center;gap:7px}
.panel .rng{font-family:var(--mono);font-size:11px;color:var(--ink-2);margin-top:4px;
 font-variant-numeric:tabular-nums}
svg{display:block;width:100%;height:auto;overflow:visible}
.gridline{stroke:var(--line-soft);stroke-width:1}
.axis{stroke:var(--line);stroke-width:1}
.tick{font-family:var(--mono);font-size:10px;fill:var(--ink-3);font-variant-numeric:tabular-nums}
.dlabel{font-family:var(--mono);font-size:11px;font-weight:500;fill:var(--ink);
 font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto;margin-top:12px;max-height:340px}
table{border-collapse:collapse;font-family:var(--mono);font-size:12px;width:100%;
 font-variant-numeric:tabular-nums;white-space:nowrap}
th,td{text-align:right;padding:5px 9px;border-bottom:1px solid var(--line-soft)}
th{color:var(--ink-3);font-weight:500;font-size:10px;letter-spacing:.08em;text-transform:uppercase;
 position:sticky;top:0;background:var(--surface)}
th:first-child,td:first-child{text-align:left}
.rows{display:flex;flex-direction:column;gap:0;border:1px solid var(--line);border-radius:6px;
 background:var(--surface);overflow:hidden}
.row{display:grid;grid-template-columns:112px 1fr 128px 96px 108px;gap:14px;align-items:center;
 padding:10px 14px;border-bottom:1px solid var(--line-soft);text-decoration:none;color:inherit}
.row:last-child{border-bottom:0}
.row:hover{background:var(--surface-2)}
.row .id{font-family:var(--mono);font-size:12.5px}
.row .num{font-family:var(--mono);font-size:12px;text-align:right;font-variant-numeric:tabular-nums;
 color:var(--ink-2)}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;
 border:1px solid var(--line);border-radius:20px;padding:2px 9px;justify-self:start;color:var(--ink-2)}
.pill.ok{border-color:var(--s3);color:var(--s3)}
footer{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);line-height:1.7}
"""

PAGE_SCRIPT = """
const D = __DATA__;
const fmt = (v, d = 1) => v.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
const el = (id) => document.getElementById(id);
// The table already carries every column; the plotted series are read back out of it
// rather than shipped a second time.
const AT = D.columnIndex;
D.rows = D.raw.map(row => {
  const parsed = {i: +row[AT.iteration], total: +row[AT.total_loss]};
  for (const term of D.terms) parsed[term.key] = +row[AT[term.key]];
  return parsed;
});
const last = D.rows[D.rows.length - 1], first = D.rows[0];
const IMAX = last.i || 1;

function axes(g, l, r, t, b, w, h, ymin, ymax, xmax, ticks) {
  const y = (v) => (h - b) - ((v - ymin) / (ymax - ymin || 1)) * (h - b - t);
  const x = (v) => l + (v / (xmax || 1)) * (w - r - l);
  for (let k = 0; k <= ticks; k++) {
    const v = ymin + (ymax - ymin) * k / ticks;
    g.push(`<line class="gridline" x1="${l}" x2="${w - r}" y1="${y(v)}" y2="${y(v)}"/>`);
    g.push(`<text class="tick" x="${l - 8}" y="${y(v) + 3.5}" text-anchor="end">${fmt(v, 0)}</text>`);
  }
  if (ymin < 0 && ymax > 0) g.push(`<line class="axis" x1="${l}" x2="${w - r}" y1="${y(0)}" y2="${y(0)}"/>`);
  return {x, y};
}

/* total loss */
{
  const W = 880, H = 250, L = 66, R = 74, T = 14, B = 32, g = [];
  const hi = Math.max(...D.rows.map(r => r.total)), lo = Math.min(0, ...D.rows.map(r => r.total));
  const s = axes(g, L, R, T, H, W, H, lo, hi, IMAX, 4);
  for (const i of [0, Math.round(IMAX / 2), IMAX])
    g.push(`<text class="tick" x="${s.x(i)}" y="${H - B + 17}" text-anchor="middle">${i}</text>`);
  g.push(`<text class="tick" x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">iteration</text>`);
  const pts = D.rows.map(r => `${s.x(r.i)},${s.y(r.total)}`).join(" ");
  g.push(`<polyline points="${pts}" fill="none" stroke="var(--s1)" stroke-width="2"
    stroke-linejoin="round" stroke-linecap="round"/>`);
  g.push(`<circle cx="${s.x(IMAX)}" cy="${s.y(last.total)}" r="4.5" fill="var(--surface)"
    stroke="var(--s1)" stroke-width="2"/>`);
  g.push(`<text class="dlabel" x="${s.x(IMAX) + 10}" y="${s.y(last.total) + 4}">${fmt(last.total)}</text>`);
  el("total").innerHTML = g.join("");
}

/* one panel per weighted term */
el("multiples").innerHTML = D.terms.map(term => {
  const W = 300, H = 150, L = 52, R = 44, T = 12, B = 24, g = [];
  const vs = D.rows.map(r => r[term.key] * term.weight);
  const hi = Math.max(...vs, 0), lo = Math.min(...vs, 0), pad = (hi - lo) * 0.12 || 1;
  const s = axes(g, L, R, T, H, W, H, lo - pad, hi + pad, IMAX, 2);
  g.push(`<polyline points="${D.rows.map((r, i) => `${s.x(r.i)},${s.y(vs[i])}`).join(" ")}"
    fill="none" stroke="${term.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`);
  g.push(`<circle cx="${s.x(IMAX)}" cy="${s.y(vs[vs.length - 1])}" r="4" fill="var(--surface)"
    stroke="${term.color}" stroke-width="2"/>`);
  g.push(`<text class="dlabel" x="${s.x(IMAX) + 8}" y="${s.y(vs[vs.length - 1]) + 4}">${fmt(vs[vs.length - 1], 0)}</text>`);
  return `<div class="panel"><h3><i class="sw" style="background:${term.color}"></i>${term.label}</h3>
    <div class="rng">weighted &times;${term.weight} &middot; ${fmt(vs[0], 0)} &rarr; ${fmt(vs[vs.length - 1], 0)}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img"
      aria-label="${term.label} weighted, ${fmt(vs[0], 0)} to ${fmt(vs[vs.length - 1], 0)}">${g.join("")}</svg></div>`;
}).join("");

/* tiles */
el("tiles").innerHTML = [
  ["Total loss", fmt(last.total), `from ${fmt(first.total)}`],
  ...D.terms.map(t => [t.label, fmt(last[t.key] * t.weight, 0),
    `from ${fmt(first[t.key] * t.weight, 0)} &middot; weight &times;${t.weight}`]),
].map(([k, v, n]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>
  <div class="n">${n}</div></div>`).join("");

/* table: every column the CSV carries */
el("table").innerHTML = `<thead><tr>${D.columns.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
  D.raw.map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("") + `</tbody>`;
"""

INDEX_SCRIPT = """
const R = __DATA__;
const fmt = (v, d = 1) => v.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
document.getElementById("rows").innerHTML = R.map(s => {
  const w = 150, h = 26, n = s.spark.length;
  const hi = Math.max(...s.spark), lo = Math.min(...s.spark);
  const pts = s.spark.map((v, i) =>
    `${(i / Math.max(1, n - 1)) * w},${h - ((v - lo) / (hi - lo || 1)) * h}`).join(" ");
  return `<a class="row" href="${s.href}">
    <span class="id">#${String(s.index).padStart(5, "0")}</span>
    <svg viewBox="0 0 ${w} ${h}" style="max-width:${w}px"><polyline points="${pts}" fill="none"
      stroke="var(--s1)" stroke-width="1.5" stroke-linejoin="round"/></svg>
    <span class="num">${fmt(s.first)} &rarr; ${fmt(s.last)}</span>
    <span class="num">${s.iterations} iter</span>
    <span class="pill ${s.success ? "ok" : ""}">${s.success ? "collision" : (s.reason || "no collision")}</span>
  </a>`;
}).join("");
"""


def _page(title, body, script, data):
    return (
        f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
        f"<style>{PAGE_STYLE}</style>{body}"
        f"<script>{script.replace('__DATA__', json.dumps(data))}</script>"
    )


def _read_history(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        raw = [row for row in reader]
    column_index = {name: idx for idx, name in enumerate(columns)}
    required = ("iteration", "total_loss", *(key for key, _, _, _ in TERMS))
    missing = [name for name in required if name not in column_index]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {', '.join(missing)}")
    rows = [
        {
            "i": int(row[column_index["iteration"]]),
            "total": float(row[column_index["total_loss"]]),
            **{key: float(row[column_index[key]]) for key, _, _, _ in TERMS},
        }
        for row in raw
    ]
    return columns, raw, rows


def _scenario_page(scenario_idx, columns, raw, weights, outcome):
    body = f"""<div class="wrap">
  <header><p class="eyebrow">ReGentS &middot; loss history</p>
  <h1>Scenario {scenario_idx:05d}</h1>
  <p class="note">{outcome}</p>
  <p class="note"><a href="index.html">&larr; all scenarios</a></p></header>
  <div class="tiles" id="tiles"></div>
  <section><h2>Total weighted loss</h2>
    <p class="note">Sum of the three terms after their weights, per optimizer iteration.</p>
    <svg id="total" viewBox="0 0 880 250" role="img" aria-label="Total weighted loss per iteration"></svg>
  </section>
  <section><h2>Each term on its own scale</h2>
    <p class="note">The terms differ by orders of magnitude; one shared axis would flatten them.</p>
    <div class="multiples" id="multiples"></div></section>
  <section><h2>Every recorded value</h2>
    <div class="tablewrap"><table id="table"></table></div></section>
  <footer>weights &middot; {" &nbsp; ".join(f"{label.lower()} {w}" for _, label, _, w in weights)}<br>
    source &middot; {LOSS_DIR_NAME}/scenario_{scenario_idx:05d}.losses.csv</footer></div>"""
    data = {
        "columns": columns,
        "columnIndex": {name: idx for idx, name in enumerate(columns)},
        "raw": raw,
        "terms": [
            {"key": key, "label": label, "weight": weight, "color": color} for key, label, color, weight in weights
        ],
    }
    return _page(f"Scenario {scenario_idx:05d} loss", body, PAGE_SCRIPT, data)


def _index_page(summaries, run_name):
    body = f"""<div class="wrap">
  <header><p class="eyebrow">ReGentS &middot; loss history</p>
  <h1>{html.escape(run_name)}</h1>
  <p class="note">{len(summaries)} scenarios. Each sparkline is that run's total weighted loss over
    its optimizer iterations; a run that found a collision stops early.</p></header>
  <div class="rows" id="rows"></div>
  <footer>generated from {LOSS_DIR_NAME}/*.losses.csv</footer></div>"""
    return _page(f"{run_name} loss history", body, INDEX_SCRIPT, summaries)


def render_loss_report(run_dir, output_dir=None):
    """Write one HTML page per loss-history CSV plus an index. Returns the index path."""
    run_dir = Path(run_dir)
    loss_dir = run_dir / LOSS_DIR_NAME
    if not loss_dir.is_dir():
        raise FileNotFoundError(f"No {LOSS_DIR_NAME}/ directory under {run_dir}")
    histories = sorted(loss_dir.glob("scenario_*.losses.csv"))
    if not histories:
        raise FileNotFoundError(f"No loss history CSVs in {loss_dir}")

    costs = ReGentSCostConfig()
    weights = [(key, label, color, getattr(costs, weight_field)) for key, label, weight_field, color in TERMS]

    outcomes = {}
    metrics_path = run_dir / METRICS_FILE_NAME
    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                outcomes[int(row["scenario_index"])] = row

    report_dir = Path(output_dir) if output_dir is not None else run_dir / REPORT_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in histories:
        scenario_idx = int(path.name.split("_")[1].split(".")[0])
        columns, raw, rows = _read_history(path)
        metric = outcomes.get(scenario_idx)
        if metric is None:
            outcome = f"{len(rows)} iterations recorded."
            success, reason = False, None
        else:
            success = metric["generation_success"] == "1"
            reason = metric["failure_reason"] or None
            outcome = f"Stopped after {len(rows) - 1} iterations &mdash; " + (
                f"ego collision at timestep {metric['collision_timestep']}, "
                f"adversary id {metric['selected_adversary_id']}."
                if success
                else f"no ego collision ({html.escape(reason or 'unknown')})."
            )
        page = _scenario_page(scenario_idx, columns, raw, weights, outcome)
        page_name = f"scenario_{scenario_idx:05d}.html"
        (report_dir / page_name).write_text(page, encoding="utf-8")
        summaries.append(
            {
                "index": scenario_idx,
                "href": page_name,
                "first": rows[0]["total"],
                "last": rows[-1]["total"],
                "iterations": len(rows) - 1,
                "success": success,
                "reason": reason,
                # A long history would bloat the index; the shape survives thinning.
                "spark": [row["total"] for row in rows[:: max(1, len(rows) // 120)]],
            }
        )

    index_path = report_dir / "index.html"
    index_path.write_text(_index_page(summaries, run_dir.name), encoding="utf-8")
    return index_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir", type=Path, help="Generation output directory containing losses/ (and generation_metrics.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Where to write the report (default: <run_dir>/{REPORT_DIR_NAME})",
    )
    args = parser.parse_args()
    index_path = render_loss_report(args.run_dir, args.output_dir)
    print(f"Loss report: {index_path}")


if __name__ == "__main__":
    main()
