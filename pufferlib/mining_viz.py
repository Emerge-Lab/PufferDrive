import json
import math
import os
import pickle
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pufferlib.ocean.drive.drive import Drive


def load_compact_replay(path):
    with open(path, "rb") as f:
        replay_bundle = pickle.loads(zlib.decompress(f.read()))

    schema_version = int(replay_bundle.get("schema_version", 0) or 0)
    if schema_version != 2:
        raise ValueError(f"Unsupported compact replay schema_version={schema_version}. Expected schema_version=2.")

    required_top_level = ("metadata", "agent_arrays", "traffic_arrays")
    missing_top_level = [key for key in required_top_level if key not in replay_bundle]
    if missing_top_level:
        raise ValueError(f"Compact replay is missing required fields: {', '.join(missing_top_level)}")

    return replay_bundle


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _resolve_map_path(map_path):
    path = Path(map_path)
    if path.exists():
        return path.resolve()

    repo_candidate = (_repo_root() / map_path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = (Path.cwd() / map_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError(f"Could not resolve map path: {map_path}")


def _normalize_scenario(scenario):
    if isinstance(scenario, list):
        return scenario[0] if scenario else {}
    return scenario or {}


def _ensure_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _materialize_agent_frames(replay_bundle):
    agent_arrays = replay_bundle["agent_arrays"]
    valid = agent_arrays["valid"]

    frames = []
    num_frames = int(valid.shape[0]) if hasattr(valid, "shape") and valid.ndim > 0 else 0
    for frame_idx in range(num_frames):
        frame = []
        frame_valid = valid[frame_idx]
        for slot_idx in np.flatnonzero(frame_valid):
            frame.append(
                {
                    "id": int(agent_arrays["id"][frame_idx, slot_idx]),
                    "type": int(agent_arrays["type"][frame_idx, slot_idx]),
                    "is_target": bool(agent_arrays["is_target"][frame_idx, slot_idx]),
                    "active": bool(agent_arrays["active"][frame_idx, slot_idx]),
                    "stopped": bool(agent_arrays["stopped"][frame_idx, slot_idx]),
                    "x": float(agent_arrays["x"][frame_idx, slot_idx]),
                    "y": float(agent_arrays["y"][frame_idx, slot_idx]),
                    "z": float(agent_arrays["z"][frame_idx, slot_idx]),
                    "heading": float(agent_arrays["heading"][frame_idx, slot_idx]),
                    "length": float(agent_arrays["length"][frame_idx, slot_idx]),
                    "width": float(agent_arrays["width"][frame_idx, slot_idx]),
                }
            )
        frames.append(frame)
    return frames


def _materialize_traffic_frames(replay_bundle):
    traffic_arrays = replay_bundle["traffic_arrays"]
    valid = traffic_arrays["valid"]

    frames = []
    num_frames = int(valid.shape[0]) if hasattr(valid, "shape") and valid.ndim > 0 else 0
    for frame_idx in range(num_frames):
        frame = []
        frame_valid = valid[frame_idx]
        for slot_idx in np.flatnonzero(frame_valid):
            frame.append(
                {
                    "type": int(traffic_arrays["type"][frame_idx, slot_idx]),
                    "state": int(traffic_arrays["state"][frame_idx, slot_idx]),
                    "stop_line": [float(coord) for coord in traffic_arrays["stop_line"][frame_idx, slot_idx].tolist()],
                }
            )
        frames.append(frame)
    return frames


def _materialize_replay_bundle(replay_bundle):
    materialized = dict(replay_bundle)
    materialized["agent_frames"] = _materialize_agent_frames(replay_bundle)
    materialized["traffic_frames"] = _materialize_traffic_frames(replay_bundle)
    return materialized


@lru_cache(maxsize=16)
def load_map_static(map_path):
    resolved_map_path = _resolve_map_path(map_path)
    env = Drive(
        map_dir=str(resolved_map_path.parent),
        maps=resolved_map_path.name,
        num_maps=1,
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        simulation_mode="gigaflow",
        scenario_length=1,
        resample_frequency=0,
        report_interval=10_000,
    )
    try:
        scenario = _normalize_scenario(env.get_state())
    finally:
        env.close()

    return {
        "map_name": scenario.get("map_name"),
        "map_corners": scenario.get("map_corners", []),
        "road_elements": scenario.get("road_elements", []),
    }


def _compute_bounds(map_static, replay_bundle):
    corners = map_static.get("map_corners") or []
    if len(corners) >= 4:
        min_x, min_y, max_x, max_y = [float(v) for v in corners[:4]]
        return [min_x, min_y, max_x, max_y]

    xs, ys = [], []
    for elem in map_static.get("road_elements", []):
        xs.extend(elem.get("x", []))
        ys.extend(elem.get("y", []))
    for frame in replay_bundle.get("agent_frames", []):
        for agent in frame:
            xs.append(agent["x"])
            ys.append(agent["y"])

    if not xs or not ys:
        return [-100.0, -100.0, 100.0, 100.0]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = max(10.0, 0.05 * max(max_x - min_x, max_y - min_y, 1.0))
    return [min_x - pad, min_y - pad, max_x + pad, max_y + pad]


def _build_render_payload(replay_bundle):
    materialized_bundle = _materialize_replay_bundle(replay_bundle)
    metadata = {key: _ensure_python_scalar(value) for key, value in materialized_bundle.get("metadata", {}).items()}
    map_static = load_map_static(metadata["map_path"])
    return {
        "metadata": metadata,
        "map": map_static,
        "bounds": _compute_bounds(map_static, materialized_bundle),
        "agent_frames": materialized_bundle.get("agent_frames", []),
        "traffic_frames": materialized_bundle.get("traffic_frames", []),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #eef1f5;
      --panel: rgba(255,255,255,0.92);
      --text: #1f2933;
      --muted: #5b6570;
      --border: rgba(31,41,51,0.12);
      --accent: #1261a0;
      --target: #d64545;
      --stopped: #f39c12;
      --active: #2a7fff;
      --inactive: #95a5a6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f7f9fc 0%, #e7edf4 100%);
    }
    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr) 300px;
      min-height: 100vh;
      gap: 18px;
      padding: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(28,38,51,0.08);
      backdrop-filter: blur(8px);
    }
    .sidebar { padding: 18px; }
    .title { margin: 0 0 6px; font-size: 22px; line-height: 1.15; }
    .subtitle { margin: 0 0 18px; color: var(--muted); font-size: 13px; }
    .meta {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-bottom: 18px;
    }
    .meta-item {
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(18,97,160,0.05);
      border: 1px solid rgba(18,97,160,0.08);
    }
    .meta-label { display: block; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .meta-value { display: block; margin-top: 4px; font-size: 16px; font-weight: 600; }
    .controls {
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }
    .nav {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    .nav a, .nav button {
      text-align: center;
      text-decoration: none;
    }
    .controls-row {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
    }
    button, select {
      border: 1px solid var(--border);
      background: white;
      color: var(--text);
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 14px;
      cursor: pointer;
    }
    input[type="range"] { width: 100%; }
    .legend {
      margin-top: 20px;
      display: grid;
      gap: 8px;
      font-size: 13px;
    }
    .legend-row { display: flex; align-items: center; gap: 8px; color: var(--muted); }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 4px;
      border: 1px solid rgba(0,0,0,0.08);
    }
    .viewer {
      position: relative;
      padding: 18px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 12px;
    }
    .episode-list {
      padding: 18px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 12px;
      min-height: 0;
    }
    .episode-list h3 {
      margin: 0;
      font-size: 16px;
    }
    .episode-scroll {
      overflow: auto;
      display: grid;
      gap: 8px;
      padding-right: 4px;
    }
    .episode-link {
      display: grid;
      gap: 3px;
      text-decoration: none;
      color: var(--text);
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.72);
    }
    .episode-link.active {
      border-color: rgba(18,97,160,0.35);
      background: rgba(18,97,160,0.09);
    }
    .episode-link small {
      color: var(--muted);
    }
    .badge {
      justify-self: start;
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 700;
      background: rgba(39,174,96,0.12);
      color: #226a47;
    }
    .badge.fail {
      background: rgba(214,69,69,0.12);
      color: #8f2d2d;
    }
    .viewer-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }
    .viewer-title {
      margin: 0;
      font-size: 18px;
    }
    .viewer-subtitle {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .viewer-tools {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    canvas {
      width: 100%;
      height: calc(100vh - 96px);
      min-height: 540px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #fbfcfe;
      display: block;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(214,69,69,0.1);
      color: #8f2d2d;
    }
    .pill.ok {
      background: rgba(39,174,96,0.12);
      color: #226a47;
    }
    @media (max-width: 1280px) {
      .layout {
        grid-template-columns: 320px minmax(0, 1fr);
      }
      .episode-list {
        grid-column: 1 / -1;
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel sidebar">
      <h1 class="title" id="title"></h1>
      <p class="subtitle" id="subtitle"></p>
      <div class="meta">
        <div class="meta-item"><span class="meta-label">Episode</span><span class="meta-value" id="meta-episode"></span></div>
        <div class="meta-item"><span class="meta-label">Map</span><span class="meta-value" id="meta-map"></span></div>
        <div class="meta-item"><span class="meta-label">Scenario</span><span class="meta-value" id="meta-scenario"></span></div>
        <div class="meta-item"><span class="meta-label">Episode Length</span><span class="meta-value" id="meta-length"></span></div>
      </div>
      <div class="controls">
        <div class="controls-row">
          <button id="play-toggle" type="button">Play</button>
          <input id="frame-slider" type="range" min="0" max="0" value="0">
          <span id="frame-label">0 / 0</span>
        </div>
        <div class="controls-row">
          <span>Speed</span>
          <input id="speed-slider" type="range" min="1" max="12" value="6">
          <span id="speed-label">1.0x</span>
        </div>
      </div>
      <div class="nav">
        <a id="back-link" href="index.html">Back To Table</a>
        <button id="focus-target" type="button">Focus Target</button>
        <a id="prev-link" href="#">Previous</a>
        <a id="next-link" href="#">Next</a>
      </div>
      <div class="legend">
        <div class="legend-row"><span class="swatch" style="background: var(--target)"></span>Target</div>
        <div class="legend-row"><span class="swatch" style="background: var(--active)"></span>Active adversary</div>
        <div class="legend-row"><span class="swatch" style="background: var(--inactive)"></span>Inactive / static</div>
        <div class="legend-row"><span class="swatch" style="background: var(--stopped)"></span>Stopped / crashed</div>
      </div>
    </aside>
    <main class="panel viewer">
      <div class="viewer-head">
        <div>
          <h2 class="viewer-title">Compact Replay</h2>
          <p class="viewer-subtitle">Lightweight mining viewer</p>
        </div>
        <div class="viewer-tools">
          <button id="reset-view" type="button">Reset View</button>
          <div id="status-pill" class="pill">Target failed</div>
        </div>
      </div>
      <canvas id="scene"></canvas>
    </main>
    <aside class="panel episode-list">
      <h3>Replay Episodes</h3>
      <div class="episode-scroll" id="episode-list"></div>
    </aside>
  </div>
  <script>
    const DATA = __DATA__;
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const slider = document.getElementById('frame-slider');
    const frameLabel = document.getElementById('frame-label');
    const playToggle = document.getElementById('play-toggle');
    const speedSlider = document.getElementById('speed-slider');
    const speedLabel = document.getElementById('speed-label');
    const statusPill = document.getElementById('status-pill');
    const backLink = document.getElementById('back-link');
    const prevLink = document.getElementById('prev-link');
    const nextLink = document.getElementById('next-link');
    const episodeList = document.getElementById('episode-list');
    const focusTargetButton = document.getElementById('focus-target');
    const resetViewButton = document.getElementById('reset-view');

    const metadata = DATA.metadata || {};
    const navigation = DATA.navigation || {};
    const frames = DATA.agent_frames || [];
    const trafficFrames = DATA.traffic_frames || [];
    const roadElements = (DATA.map && DATA.map.road_elements) || [];
    const bounds = DATA.bounds || [-100, -100, 100, 100];

    let frameIndex = 0;
    let playing = false;
    let lastTimestamp = 0;
    let speed = 1.0;
    let camera = null;
    let dragState = null;
    let hitAgents = [];

    function createDefaultCamera() {
      const minX = bounds[0], minY = bounds[1], maxX = bounds[2], maxY = bounds[3];
      return {
        x: (minX + maxX) / 2,
        y: (minY + maxY) / 2,
        zoom: 1.0,
      };
    }

    function setMeta() {
      document.getElementById('title').innerText = metadata.map_name || 'Replay';
      document.getElementById('subtitle').innerText = `dynamics=${metadata.dynamics_model || 'unknown'} | target=${metadata.target_type || 'unknown'}`;
      document.getElementById('meta-episode').innerText = metadata.episode_id ?? 'N/A';
      document.getElementById('meta-map').innerText = metadata.map_name || 'N/A';
      document.getElementById('meta-scenario').innerText = metadata.scenario_id || 'N/A';
      document.getElementById('meta-length').innerText = metadata.episode_length ?? frames.length;
      const failed = Number(metadata.did_target_fail || 0) > 0;
      statusPill.className = failed ? 'pill' : 'pill ok';
      statusPill.innerText = failed ? 'Target failed' : 'Target survived';
      backLink.href = navigation.index_html || 'index.html';
      prevLink.href = navigation.prev_html || '#';
      nextLink.href = navigation.next_html || '#';
      prevLink.style.pointerEvents = navigation.prev_html ? 'auto' : 'none';
      nextLink.style.pointerEvents = navigation.next_html ? 'auto' : 'none';
      prevLink.style.opacity = navigation.prev_html ? '1' : '0.4';
      nextLink.style.opacity = navigation.next_html ? '1' : '0.4';
      const items = navigation.episodes || [];
      episodeList.innerHTML = items.map(item => {
        const active = item.episode_id === metadata.episode_id ? 'episode-link active' : 'episode-link';
        const badge = item.did_target_fail ? '<span class="badge fail">fail</span>' : '<span class="badge">ok</span>';
        return `<a class="${active}" href="${item.href}">${badge}<strong>Episode ${item.episode_id}</strong><small>${item.map_name || ''} | ${item.scenario_id || ''}</small></a>`;
      }).join('');
    }

    function resizeCanvas() {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function worldToCanvas(x, y) {
      const minX = bounds[0], minY = bounds[1], maxX = bounds[2], maxY = bounds[3];
      const pad = 22;
      const usableW = canvas.clientWidth - pad * 2;
      const usableH = canvas.clientHeight - pad * 2;
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const baseScale = Math.min(usableW / spanX, usableH / spanY);
      const scale = baseScale * (camera ? camera.zoom : 1.0);
      const centerX = canvas.clientWidth / 2;
      const centerY = canvas.clientHeight / 2;
      return {
        x: centerX + (x - (camera ? camera.x : (minX + maxX) / 2)) * scale,
        y: centerY - (y - (camera ? camera.y : (minY + maxY) / 2)) * scale,
        scale,
      };
    }

    function canvasToWorld(x, y) {
      const minX = bounds[0], minY = bounds[1], maxX = bounds[2], maxY = bounds[3];
      const pad = 22;
      const usableW = canvas.clientWidth - pad * 2;
      const usableH = canvas.clientHeight - pad * 2;
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const baseScale = Math.min(usableW / spanX, usableH / spanY);
      const scale = baseScale * (camera ? camera.zoom : 1.0);
      const centerX = canvas.clientWidth / 2;
      const centerY = canvas.clientHeight / 2;
      return {
        x: (x - centerX) / scale + camera.x,
        y: (centerY - y) / scale + camera.y,
      };
    }

    function findTarget(frame) {
      return (frame || []).find(agent => agent.is_target);
    }

    function focusOnAgent(agent, zoom = 2.5) {
      if (!agent) return;
      camera.x = agent.x;
      camera.y = agent.y;
      camera.zoom = zoom;
      draw();
    }

    function drawRoads() {
      for (const elem of roadElements) {
        const xs = elem.x || [];
        const ys = elem.y || [];
        if (xs.length < 2 || ys.length < 2) continue;
        const type = Number(elem.type || 0);
        let style = null;
        if (type >= 1 && type <= 3) style = { color: '#d9dde3', width: 1.0, alpha: 0.9 };
        else if (type >= 11 && type <= 18) style = { color: '#8f98a3', width: 0.7, alpha: 0.6 };
        else if (type >= 21 && type <= 23) style = { color: '#2f3640', width: 0.8, alpha: 0.8 };
        if (!style) continue;
        ctx.beginPath();
        for (let i = 0; i < xs.length; i++) {
          const p = worldToCanvas(xs[i], ys[i]);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = style.color;
        ctx.globalAlpha = style.alpha;
        ctx.lineWidth = style.width;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    }

    function drawTraffic(frame) {
      for (const control of frame || []) {
        const sl = control.stop_line || [];
        if (sl.length < 6) continue;
        const p1 = worldToCanvas(sl[0], sl[1]);
        const p2 = worldToCanvas(sl[3], sl[4]);
        let color = '#888';
        if (Number(control.type) === 1) {
          const state = Number(control.state || 0);
          if (state === 1) color = '#d64545';
          else if (state === 2) color = '#f1c40f';
          else if (state === 3) color = '#27ae60';
        } else if (Number(control.type) === 2) {
          color = '#d64545';
        } else if (Number(control.type) === 3) {
          color = '#d4ac0d';
        }
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.9;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    }

    function drawAgent(agent) {
      const center = worldToCanvas(agent.x, agent.y);
      const scale = center.scale;
      const length = Math.max(agent.length * scale, 6);
      const width = Math.max(agent.width * scale, 4);
      const heading = Number(agent.heading || 0);
      let fill = '#2a7fff';
      if (agent.is_target) fill = '#d64545';
      else if (!agent.active) fill = '#95a5a6';
      if (agent.stopped) fill = '#f39c12';

      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.rotate(-heading);
      ctx.fillStyle = fill;
      ctx.strokeStyle = 'rgba(20,20,20,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(-length / 2, -width / 2, length, width);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(length / 2, 0);
      ctx.lineTo(length / 2 - Math.max(width * 0.9, 6), width * 0.32);
      ctx.lineTo(length / 2 - Math.max(width * 0.9, 6), -width * 0.32);
      ctx.closePath();
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.fill();
      ctx.restore();

      hitAgents.push({
        agent,
        x: center.x,
        y: center.y,
        radius: Math.max(length, width) * 0.7,
      });
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      hitAgents = [];
      drawRoads();
      drawTraffic(trafficFrames[frameIndex] || []);
      const frame = frames[frameIndex] || [];
      for (const agent of frame) drawAgent(agent);
      frameLabel.innerText = `${frameIndex + 1} / ${Math.max(frames.length, 1)}`;
      slider.value = frameIndex;
    }

    function tick(ts) {
      if (!playing) return;
      if (!lastTimestamp) lastTimestamp = ts;
      const frameDuration = 1000 / (10 * speed);
      if (ts - lastTimestamp >= frameDuration) {
        frameIndex = (frameIndex + 1) % Math.max(frames.length, 1);
        lastTimestamp = ts;
        draw();
      }
      requestAnimationFrame(tick);
    }

    slider.max = Math.max(frames.length - 1, 0);
    slider.addEventListener('input', (e) => {
      frameIndex = Number(e.target.value || 0);
      draw();
    });
    playToggle.addEventListener('click', () => {
      playing = !playing;
      playToggle.innerText = playing ? 'Pause' : 'Play';
      lastTimestamp = 0;
      if (playing) requestAnimationFrame(tick);
    });
    speedSlider.addEventListener('input', (e) => {
      speed = Number(e.target.value || 6) / 6;
      speedLabel.innerText = `${speed.toFixed(1)}x`;
    });
    focusTargetButton.addEventListener('click', () => focusOnAgent(findTarget(frames[frameIndex])));
    resetViewButton.addEventListener('click', () => {
      camera = createDefaultCamera();
      draw();
    });
    canvas.addEventListener('mousedown', (e) => {
      dragState = { x: e.offsetX, y: e.offsetY, cameraX: camera.x, cameraY: camera.y };
    });
    canvas.addEventListener('mousemove', (e) => {
      if (!dragState) return;
      const before = canvasToWorld(dragState.x, dragState.y);
      const now = canvasToWorld(e.offsetX, e.offsetY);
      camera.x = dragState.cameraX + (before.x - now.x);
      camera.y = dragState.cameraY + (before.y - now.y);
      draw();
    });
    window.addEventListener('mouseup', () => { dragState = null; });
    canvas.addEventListener('mouseleave', () => { dragState = null; });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const pointerBefore = canvasToWorld(e.offsetX, e.offsetY);
      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.9;
      camera.zoom = Math.min(20, Math.max(0.4, camera.zoom * zoomFactor));
      const pointerAfter = canvasToWorld(e.offsetX, e.offsetY);
      camera.x += pointerBefore.x - pointerAfter.x;
      camera.y += pointerBefore.y - pointerAfter.y;
      draw();
    }, { passive: false });
    canvas.addEventListener('click', (e) => {
      const x = e.offsetX;
      const y = e.offsetY;
      for (const hit of hitAgents) {
        const dx = x - hit.x;
        const dy = y - hit.y;
        if (dx * dx + dy * dy <= hit.radius * hit.radius) {
          focusOnAgent(hit.agent, hit.agent.is_target ? 3.0 : 2.2);
          break;
        }
      }
    });

    setMeta();
    camera = createDefaultCamera();
    speedLabel.innerText = `${speed.toFixed(1)}x`;
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""


def render_compact_replay_html(replay_path, output_path, render_context=None):
    replay_bundle = load_compact_replay(replay_path)
    payload = _build_render_payload(replay_bundle)
    payload["metadata"]["episode_id"] = payload["metadata"].get("episode_id", Path(output_path).stem)
    payload["navigation"] = render_context or {}
    title = f"{payload['metadata'].get('map_name', 'Replay')} | {payload['metadata'].get('scenario_id', 'episode')}"
    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def _safe_value(value):
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def generate_failure_index(episodes_df, render_lookup, output_path):
    rows = []
    preferred_columns = [
        "episode_id",
        "map_name",
        "scenario_id",
        "did_target_fail",
        "did_target_collide",
        "did_target_offroad",
        "target_episode_return",
        "episode_return_adversarial",
        "target_episode_length",
        "active_agent_count",
        "has_replay",
    ]
    existing_columns = [col for col in preferred_columns if col in episodes_df.columns]
    for row in episodes_df.to_dict(orient="records"):
        replay_html = render_lookup.get(row.get("episode_id"))
        out = {key: _safe_value(row.get(key)) for key in existing_columns}
        out["rendered_html"] = replay_html
        rows.append(out)

    rows.sort(key=lambda item: (-(item.get("did_target_fail") or 0), item.get("target_episode_return") or 0))
    title = Path(output_path).parent.name
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Failure Index</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f7fb; color: #1f2933; }}
    .wrap {{ padding: 20px; max-width: 1600px; margin: 0 auto; }}
    .head {{ margin-bottom: 18px; }}
    .head h1 {{ margin: 0 0 6px; font-size: 28px; }}
    .head p {{ margin: 0; color: #5b6570; }}
    .controls {{ margin: 14px 0 18px; display: flex; gap: 12px; align-items: center; }}
    input {{ border: 1px solid rgba(31,41,51,0.16); border-radius: 10px; padding: 8px 12px; min-width: 280px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 28px rgba(28,38,51,0.08); }}
    thead {{ background: #eaf0f6; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid rgba(31,41,51,0.08); text-align: left; font-size: 14px; }}
    th {{ cursor: pointer; user-select: none; }}
    tbody tr:hover {{ background: #f8fbff; }}
    a {{ color: #1261a0; text-decoration: none; font-weight: 600; }}
    .muted {{ color: #7b8794; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <h1>Failure Index</h1>
      <p>{title}</p>
    </div>
    <div class="controls">
      <input id="search" type="search" placeholder="Filter rows">
      <span class="muted" id="count"></span>
    </div>
    <table id="failure-table">
      <thead><tr><th data-key='rendered_html'>render</th>{"".join(f"<th data-key='{col}'>{col}</th>" for col in existing_columns)}</tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <script>
    const ROWS = {json.dumps(rows, separators=(",", ":"))};
    const COLS = {json.dumps(existing_columns)};
    const tbody = document.querySelector('#failure-table tbody');
    const count = document.getElementById('count');
    const search = document.getElementById('search');
    let sortKey = 'did_target_fail';
    let sortDir = -1;

    function renderTable() {{
      const term = (search.value || '').toLowerCase();
      const filtered = ROWS.filter(row => JSON.stringify(row).toLowerCase().includes(term));
      filtered.sort((a, b) => {{
        const av = a[sortKey];
        const bv = b[sortKey];
        if (av === bv) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        return av > bv ? sortDir : -sortDir;
      }});
      tbody.innerHTML = filtered.map(row => {{
        const cells = COLS.map(col => `<td>${{row[col] == null ? '' : row[col]}}</td>`).join('');
        const link = row.rendered_html ? `<a href="${{row.rendered_html}}">open</a>` : '<span class="muted">n/a</span>';
        return `<tr><td>${{link}}</td>${{cells}}</tr>`;
      }}).join('');
      count.innerText = `${{filtered.length}} rows`;
    }}

    document.querySelectorAll('th[data-key]').forEach(th => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.key;
        if (sortKey === key) sortDir *= -1;
        else {{
          sortKey = key;
          sortDir = key === 'rendered_html' ? 1 : -1;
        }}
        renderTable();
      }});
    }});
    search.addEventListener('input', renderTable);
    renderTable();
  </script>
</body>
</html>"""
    with open(output_path, "w") as f:
        f.write(html)
    return output_path
