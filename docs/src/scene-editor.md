# Interactive scenario editor

A browser-based playground for inspecting and editing Waymo Open Motion Dataset (WOMD) scenes. The tool runs fully client-side at <https://womd-editor.vercel.app/> and works directly with the JSON format produced by Waymo/ScenarioMax exports and PufferDrive conversions.

## Video walkthrough

<div class="video-embed">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/kzJptblJ4Kw?si=1lVRHmM1HjwCkgP5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Quick start
- Open <https://womd-editor.vercel.app/> in a modern Chromium/Firefox browser.
- Click **Import JSON…** in the left sidebar and drop one or more scenario files (Waymo/ScenarioMax JSON or editor exports).
- The app stores everything in-memory only; nothing is uploaded to a server.

## What you can do
- **Inspect**: Top-down canvas with zoom/pan/rotate, agent labels, and a playback timeline with variable speed.
- **Edit trajectories**: Select an agent and tweak paths via drag handles, draw a polyline with the Line tool, freehand record a path, or drive the agent with keyboard controls (WASD/arrow keys, Space to brake, Enter to save, Esc to cancel).
- **Edit roads**: Switch to Road mode to draw or refine lane/edge/crosswalk geometry, recolor vertices by elevation, and view the lane connectivity overlay when ROAD_LANE/ROAD_LINE data exists.
- **Configure metadata**: Rename the scenario, toggle label mode (ID vs. array index), mark agents as experts, and choose which agents belong to `tracks_to_predict`.
- **Export**: Preview changes versus the import baseline, then download either Waymo-style JSON or a compact `.bin` suitable for PufferDrive’s loader.

## Editing workflow
1. **Load a scene**: Import one or multiple JSONs; each appears as a row in the Scenarios list with a quick delete button.
2. **Playback**: Use the timeline to scrub frames or Space/Arrow keys to play/pause/step. Agent labels and trajectory visibility can be toggled in the editor panel.
3. **Trajectory tools** (Trajectory mode):
   - **Adjust Path**: Drag existing vertices/handles on the canvas.
   - **Line Tool**: Click to lay out a polyline, set per-segment duration (seconds), then **Apply Path** to rebuild timestamps/velocity.
   - **Record Path**: Freehand capture a path with the pointer; playback resets to frame 0.
   - **Drive Agent**: Enter a lightweight driving loop; W/A/S/D or arrow keys steer, Space brakes, Enter saves, Esc cancels. Tunable speed/accel/steer sliders live under “Drive Tune.”
4. **Road tools** (Road mode):
   - **Edit Geometry**: Select segments/vertices to move, insert, split, or delete (Shift/Ctrl-click to insert on-canvas; Alt/Cmd-click to delete).
   - **Draw Road**: Click to add vertices; Enter finishes, Esc cancels. Set the default Z used for new vertices in the right-hand panel.
   - **Type & overlays**: Tag segments as ROAD_LANE / ROAD_EDGE / ROAD_LINE / CROSSWALK / OTHER. Enable **Color by Z** to visualize elevation and **Lane Graph** to see lane entry/exit nodes plus downstream arrows.
5. **Export & diff**: Hit **Export** to open a preview modal that summarizes changes (metadata, agents, roads, tracks_to_predict, bounds, frames). Download JSON for round-tripping or `.bin` for simulator ingestion.

## Using exports with PufferDrive
- JSON exports retain the Waymo layout (`objects`, `roads`, `tracks_to_predict`, `tl_states`, `metadata`) and can be converted or re-imported.
- `.bin` exports match the compact format read by `pufferlib/ocean/drive/drive.py`; drop them into `resources/drive/binaries` (e.g., `map_000.bin`) to test inside the simulator.
- The editor auto-fills missing headings/speeds and clamps degenerate lanes to keep bounds reasonable; always spot-check via the Export preview before committing.

## Notes
- The app is currently work-in-progress; there is no persistent storage or backend sync.
- Large scenes may render slowly on low-power GPUs—hide trajectories or road overlays to keep the canvas responsive.
- Source lives in the `WOMD-Editor/web` directory of this repo if you want to run it locally with `npm install && npm run dev`.
