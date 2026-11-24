## Rules and guidelines for the datasets

PufferDrive can be initialized from logged datasets. Waymo Open Motion (WOMD) and I24-3D both use the JSON layout in `utils/common.py`. A converter writes the logs. It does not infer actions. Appendix A.4 of arXiv 2606.19370 does that on the logs in the simulator.

### This repository

`load_map_binary` in `pufferlib/ocean/drive/drive.h` reads the record written by `save_map_binary`. That record is 91 states. Object types are vehicle, pedestrian, and cyclist. Type 4 is a road lane, so a JSON partner labeled `truck` is stored as a vehicle. Logs longer than 91 states, including a full I24 scene, are truncated when the binary is written. JSON on disk is not retagged.

WOMD JSON already in this checkout is `data/processed/training` (50k) and `data/training` (1k). The trainer's map binaries are `resources/drive/binaries/training`. New binaries go under `data/<name>/binaries/` and are not a replacement for that training set.

I24 raw scenes are `data/i24/raw/I24-3D/data/spl_obj/scene{1,2,3}_splobj.json` and `data/i24/raw/I24-3D/data/hg/scene{n}_hg.json`. `python utils/i24.py` converts them. That conversion is not run as part of checking out the raw files.

Flat CARLA towns are `data/carla/json/`. A town is flat when its road geometry z span is at most 0.1 m. That keeps Town01, Town02, Town06, and Town10HD, and leaves out towns with elevation or overpasses (Town03, Town04, Town05, Town07). `python utils/carla.py` writes `data/carla/binaries/map_%03d.bin`.

### Layout

Each dataset is `data/<name>/`. `config` `data_dir` points at that directory.

- `json/*.json` — one scenario per file. Sorted names are the map index.
- `binaries/map_%03d.bin` — map `i` is the `i`-th JSON.
- `binaries/sdc.bin` — int count, then one int32 self-driving-car index per map.
- `assets/` — inspection PNG and GIF from `utils/sanity.py`.
- `raw/` — the download, when we keep it. The trainer does not read this directory.

### Logs

One step is 0.1 s. Both endpoints count. A log is 2 through 901 states (`DRIVE_LOG_MAX`). Waymo segments stay 91 states (9.0 s). An I24 scenario keeps its own length, capped at the longest scene clock, 90 s.

Every object in one binary shares that length. Object 0 is the self-driving car, a vehicle or a truck, and it is valid on every state. Its goal is the last valid logged position. `mark_as_expert` is 0, so the loader keeps the surrounding traffic. Distances in the logs are meters.

A file is either a log (objects, then roads) or a road map (roads only). A JSON with no objects is a road map. The record layout is `utils/common.py`.

### Categories

Object types are `vehicle`, `pedestrian`, `cyclist`, and `truck` (1, 2, 3, and 4), matching `ocean/drive/constants.h`. Roads start at 5, so a type below that is a body. A pedestrian or cyclist stays that class. Any other vehicle-like body whose length is at least 7.5 m is a truck, including a tractor-trailer. Waymo and I24 use this same cut. Object 0 is a vehicle or a truck.

Road `type` is one of `lane`, `road_line`, `road_edge`, `stop_sign`, `crosswalk`, `speed_bump`, `driveway`. `map_element_id` is the Waymo roadgraph id. The binary stores the collapsed drive class: lanes 5, lines 6, edges 7, then stop sign 8, crosswalk 9, speed bump 10, and driveway 11. Keep freeway lanes and median edges as those source ids. Do not rewrite them to surface street or boundary.

### I24-3D

`python utils/i24.py` reads `data/spl_obj/scene{1,2,3}_splobj.json` and `data/hg/scene{n}_hg.json`.

Processing assumptions:

- Positions and sizes in the splines are feet. Logs are meters.
- Time is 10 Hz with both endpoints. A track keeps its own length, capped at 90 s.
- Scene 1 is also cut on the file clock. The origin is the earliest timestamp in `scene1_splobj.json`. A sample later than 90 s after that origin is dropped before resampling, and a track with fewer than two samples left is removed. Scenes 2 and 3 keep their full clocks.
- Body styles `sedan`, `midsize`, `pickup`, `van`, `semi`, `truck`, and `cs` are vehicle-like. The exported length then selects `vehicle` or `truck` with the shared 7.5 m cut. `cs` is the one car-sized track in scene 3. Any other label is an error.
- Exported length, width, and height are 0.90 of the annotated size, after the small-size defaults (4.5 m, 1.8 m, 1.5 m).
- Heading is a 1.0 s centered regression on position. Below 0.5 m/s the last reliable heading is held. That regression leaves the samples unchanged. Position repair below rewrites samples first, and velocity is the centered difference of the repaired samples. `--no-stabilize-headings` uses the per-frame velocity heading instead.
- One scenario is one focal track plus nearby traffic, at most 128 objects. The default focal track is the longest clean one. Object 0 is that vehicle or truck and is valid on every state. `mark_as_expert` is 0.
- Track deletion is only the cleanup below. Quality flags (`speed_above_60_mps`, `more_than_200m_outside_calibrated_x_range`, `outside_calibrated_y_range`, `shorter_than_9_seconds`) are descriptive. `--exclude-major-flags` can skip a focal track. It does not delete neighbors.

#### Cleanup

When a rule changes, write the new rule here with its threshold and what it rewrites or deletes, then regenerate the logs. These logs have no true-positive collisions, so an overlap or a sudden stop-and-reverse is a tracker failure.

Rules run in this order. Position repair runs on every track at 10 Hz, before velocity, heading, and the choice of focal car. A track that repair cannot fix is removed there, so it is not chosen as the self-driving car and does not take a neighbor slot. Wrong-way tracks and tracks that do not move are removed next, before the nearest-128 cut. Deep overlaps are judged on the tracks that remain. Those pairs are marked first and deleted together. There is no second pass, and the nearest-128 cut is not filled again after that deletion.

1. Position repair. Two passes. Each pass puts single bad samples on a chord, then interpolates a short burst that returns to the surrounding motion.
   - Single sample. For an interior sample, measure the distance to the midpoint of its original neighbors. At least 0.25 m: replace the sample with that midpoint. At least 1.25 m: copy the previous sample onto it, then replace it with the midpoint, so the next step is not a zero-length catch-up. A last sample whose step differs from the previous step by at least 1.25 m has no following neighbor: copy the previous sample and drop the duplicate.
   - Burst. A step is marked when it differs by at least 0.25 m from the median of the other steps in a ±25-frame (5 s) window, and that window has at least 5 other steps. Marked runs with at most 5 steps between them are one burst. A burst longer than 30 steps (3 s) is left alone. An interior burst is replaced by the straight line from the sample before it to the sample two steps after it, and only when that far sample is within 4 m of where the burst's median step would have landed. The 4 m gate leaves a real 2 s stop in place (the car is about 6 m behind the moving median) and still repairs a there-and-back of a few meters. A burst that starts at the first sample is dropped through the end of the burst. A burst that reaches the last sample is cut off at its start.
   - Reversal, after the two passes. The freeway direction is the sign of the median Δx. Below 0.05 m the track has no direction; if both directions travel more than 10 m the track is removed. A consecutive run with Δx against that sign is opposing when the step is past 0.05 m. Between 0.5 m and 10 m, if the original direction resumes, the run is drawn as a straight line from the last forward sample to the first resumed sample. Steps after the run that are still at least 0.25 m off the median, up to 30 steps, are included so a snap-back is part of the same line. A reversal above 10 m that returns to within 4 m of the median continuation is drawn the same way. A reversal above 10 m at the start or the end is cut off, and the track is removed when fewer than 2 s (21 states) remain. A reversal above 10 m in the middle, where both directions travel more than 10 m and the path does not return, removes the track. A true U-turn is not replaced with a new forward path.
   - Anything still at least 2 m off its local median step is too strange. A prefix or suffix like that is cut. An interior one removes the track.
   - Dropped prefix samples advance `start_tick`. Velocity and heading are recomputed after this.

2. Wrong-way track. The median is the midpoint of the largest gap in the calibration y lines. A track belongs to the lower carriageway when the mean of its valid y in the focal window is below that line, and to the upper carriageway otherwise. Each side has one legal direction: the majority sign of net x travel (last valid x minus first valid x) among tracks on that side whose absolute net travel is at least 10 m. A tie leaves the lower side eastbound (+x) and the upper side westbound (−x). Opposing distance is the sum of |Δx| on steps that move against the legal sign. A gap of invalid states breaks the sum. A track whose opposing distance is greater than 10 m is deleted. The focal track is never deleted by this rule. Its opposing distance is stored on the scenario.

3. Stationary track. Travel is the diagonal of the axis-aligned box around the track's valid positions in the focal window. A diagonal of 1 m or less deletes the track. In the raw scenes the only such track is scene 3 id 2 (a semi whose samples stay inside 0.93 m); the next-smallest track travels 6.05 m. The automatic focal car is chosen from tracks above this threshold. An explicitly requested focal car is kept and recorded.

4. Deep overlap. Boxes are the exported length and width (already scaled by 0.90) at the logged heading. Both cars must be valid at the timestep. The timestep counts when the intersection area of the two rectangles is more than half the smaller box area, `length × width`. A pair qualifies when that count is greater than 5. The smaller car is the one with the smaller area. Equal areas treat the higher source id as the smaller car. Every smaller car in a qualifying pair is deleted, including a car that would no longer overlap once another car was gone. Object 0 is never deleted. If it is the smaller member, both tracks stay and the pair is recorded.

`metadata.track_cleanup` on each scenario lists the removed ids, the legal direction, the focal opposing distance, any stationary focal car that was kept, and any deep overlap that was kept because the focal car was the smaller one. `metadata.position_repair` lists the thresholds, the smoothed and copied frame counts, the dropped prefix and suffix frames, the reversal counts, and the ids removed as unfixable. The heading block stays `positions_modified: false` because the heading regression itself does not move samples.
- Lanes, broken white lines, and edges come from the 12-foot calibration grid. Eastbound runs toward +x. The lower carriageway is eastbound. The grid is shifted so the focal car starts at the origin, extended along the roadway, and sampled every 5 m. The simulator stores one observation per segment midpoint, so an unsampled chord leaves the view empty.
- No vehicle in the source data is treated as overlapping a real road edge. The calibration edge is the outer lane line, so each edge, including the median, is moved 1.5 ft away from the lanes.

### WOMD

`python utils/womd.py` turns JSON scenarios into simulator binaries. Roads are thinned in the binary only. The source JSON is not rewritten, and a long vehicle stays `vehicle` in the binary. The Hugging Face download documented upstream is `daphne-cornelisse/pufferdrive_womd_train_1000`. This checkout already has WOMD JSON under `data/processed/training`.

### Inspection

After processing a dataset, run `python utils/sanity.py --data-dir data/<name>`. It is the shared matplotlib and seaborn viewer for log replay. It picks two vehicles at random and writes under `data/<name>/assets`:

- a PNG contact sheet of the first 5 seconds, one frame per second, a cropped bird's-eye view beside that agent's observation. The observation plots the road points from the vector, and the ego box, speed arrow, goal, and corner text are the ego features in Appendix A of arXiv 2502.14706 plus this simulator's accel, yaw rate, and target speed
- a GIF of the same two views at 10 Hz for 10 seconds, or for the whole log when the log is shorter. A Waymo segment is 9.0 s.
