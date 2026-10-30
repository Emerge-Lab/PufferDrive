# Wheel demo — handoff

State of the steering-wheel demo as of **2026-10-30**, written so that a fresh session on another
machine can continue without the conversation history.

Read [`wheel_demo_plan.md`](wheel_demo_plan.md) for the design and the milestone definitions, and
[`wheel_demo_setup.md`](wheel_demo_setup.md) to get a machine running.

Branch: `aditya/wheel_demo`, off `3.0`. Last commit at time of writing: `9607061a` (M0–M6).
M7 and the render work below are committed after it.

---

## 1. Where the work stands

| Milestone | State |
| --- | --- |
| M0 Hardware probe and scaffold | done |
| M1 Input layer | done |
| M2 `CONTROLLER_HUMAN`, direct steering | done |
| M3 Powertrain: clutch, gears, two modes | done |
| M4 Real-time loop and sub-stepping | done |
| M5 Renderer and cameras | done |
| M6 Force feedback | done — transport proven, **feel never tuned by a human** |
| M7 Reward tuning and levels | implemented and measured, but **no automated tests** — see §3a |
| M8 Calibration, polish, determinism | **not started** |
| M9 Steering assist | **not started** |

### Expected test results

```
make -C tests/drive           ->  All test groups passed, Failed: 0   (234 PASS + test_error's 11)
make -C tests/drive render    ->  15 PASS
python -m pytest -q tests/unit_tests
                              ->  510 passed, 21 skipped, 1 failed
```

That one failure is **pre-existing and unrelated**:
`test_action_selection.py::test_drive_mean_action_is_exact_in_physical_units` fails on an
`ImportError` because `fused_slot_encoder=true` requires triton, which has no macOS wheel. On Linux
with triton installed it may well pass — if it does, that is fine, not a regression.

### The policy ships with the repo

`weights/k_scaled_0050_1000/` holds `final_model.pt` and the `config.yaml` the demo env is built
from, following the convention already used by `weights/k_scaled_0036_1000/` and friends.
`run_demo.py` defaults to it, so the demo runs on a fresh clone with no flags.

### Policy integrity

A golden jerk-dynamics trajectory was captured before any refactor and has stayed **bit-identical**
through every change since, including the `integrate_bicycle_arc` extraction, sub-stepping, and the
reward-bounds work. If a change ever moves it, that change is wrong. The human-control path never
touches jerk dynamics (see plan D3).

---

## 2. What was done most recently

### M6 fixes (after the independent test agent's first pass)

Four defects, all now fixed and pinned by tests:

1. **The collision pulse never fired.** `ForceFeedbackLoop._run` never passed `impact_age_s`, which
   is the only trigger. It now edge-detects `feel.collision`, records the impact time and feeds the
   age until `IMPACT_WINDOW_SECONDS`.
2. **The offroad rumble had no data source.** `car_feel` hardcoded `offroad=False`, and
   `vec_get_human_state` never exported the metric. The binding now exports
   `metrics_array[OFFROAD_IDX]` as `"offroad"` and `car_feel` reads it. Verified live: drive at full
   lock and full throttle and the flag goes to 1.0 within a few steps.
3. **`send()` awaited a reply at 200 Hz** — one dropped reply stalled the wheel for a whole timeout.
   Per-frame downloads now use `want_reply=False`; `start()` still waits.
4. **No "started" guard** on `G920ForceFeedback.send` — raised `TypeError` from inside `bytes()`.

### M7 — reward tuning levels

`c_set_reward_coef_bounds` / `c_resample_reward_coefs` in C, validated against the policy's trained
window; three presets in `pufferlib/wheel_demo/levels.py` expressed as *fractions* of that window so
a level can never ask for a coefficient the policy has not seen. Measured effect: conditioning
shifts **+0.70 / +0.01 / −0.70** and nearest-gap **9.7 / 9.1 / 8.3 m** across calm/normal/aggressive.
`reward_bounds_overridden` keeps the training sampler untouched by default.

### Renderer overhaul (2026-10-30, in response to driving it)

The first real drive showed the views were poor. Changed:

- **Chase camera was a drone** — 25 m back, 15 m up. Now 9 m / 3.6 m, target 14 m ahead. Your car is
  the subject of the shot instead of a speck behind other traffic.
- **Driver camera sat at the car's centre.** Now in the driver's seat: 0.30 m forward, 0.38 m left
  (left-hand drive), eye 1.26 m, aimed lower.
- **The HUD was a 104 px opaque slab** covering the near road and, in chase view, the car itself. Now
  76 px and lighter, with the key help moved to the top of the screen. In driver view the dashboard
  replaces the panel entirely.
- **Cockpit**: a dash with a raised cowl behind the wheel; wheel 138→112 px and dropped so it sits on
  the dash rather than floating over the road.
- **World palette was the single biggest problem** — road 154,161,171 on ground 220,223,214 is
  near-identical brightness. Now dark asphalt (72,74,78) on grass (108,132,88) with light kerbs.
- Ego marker (the cyan cone) now only in the top-down view.
- Cull 220 → 420 m. Measured: 220 vs 700 m is nearly identical in practice, so this is a middle
  value and costs nothing (26.1 ms compute against a 300 ms budget).
- `drive_render.screenshot(path)` added — raylib writes the frame from inside the process, which
  needs no screen-recording permission. Note raylib's `TakeScreenshot` **strips the directory** and
  writes the basename into the working directory.

Reference frames: `docs/wheel_demo_shots/`.

### Brake response

The G920 brake is a progressive rubber stop whose raw floor is 53, not 0, so a linear map put all the
bite at the very end. `WheelCalibration.brake_response_gamma` (default **0.6**) now bends it: 25%
travel gives 42% braking (was 25%), 50% gives 66%. **Only a foot can say whether this is now right.**

---

## 3a. M7 has no test coverage

This is a gap against the project's own working agreement, and it should be closed before M7 is
called done. Nothing under `tests/` references `levels`, `set_reward_coef_bounds`,
`resample_reward_coefs` or the reward-bounds C entry points — verified by grep, not by assumption.

What M7 currently rests on is **measurement, not tests**: applying the three presets shifts mean
conditioning by **+0.70 / +0.01 / −0.70** and nearest-gap by **9.7 / 9.1 / 8.3 m**, and the policy
golden stayed identical. That is evidence the feature works; it is not a regression guard.

Two test files are missing and should be written by an independent agent:

- `tests/unit_tests/test_wheel_levels.py` — that a level shifts the window rather than pinning a
  value; that fractions outside `0 <= low <= high <= 1` raise; that `COEF_SPEED` uses its special
  `[1/conditioning_speed_scale, scale]` limits; that the human car's coefficients are never touched;
  that cycling levels is reversible.
- `tests/drive/test_drive_reward_tuning.c` — that `c_set_reward_coef_bounds` rejects bounds outside
  the policy's trained window, that `c_resample_reward_coefs` returns the number of cars changed and
  skips the human, and that `reward_bounds_overridden` leaves the training sampler untouched by
  default.

## 3. Open findings, not yet fixed

From the independent test agent's second pass on M6. Numbered as it reported them.

| # | Severity | Finding |
| --- | --- | --- |
| F7 | MEDIUM | **The collision jolt can land up to ~300 ms late.** `compute_metrics` runs only in `c_step`, never in `c_substep`, so `collision` is constant across all 18 substeps. The FFB thread polls at 200 Hz but the signal it edge-detects is quantised to the 0.3 s policy grid. The fix is correct; the latency is in the data source. Making the jolt feel like the crash you just watched means running the human agent's collision check inside `c_substep`. |
| F8 | MEDIUM | **Hit something while off the road and you get no jolt at all.** `compute_metrics` sets `OFFROAD_IDX` and early-returns *before* the collision check, so the two flags are mutually exclusive. `car_feel` reads them as independent. Pinned by a test so it cannot change silently, but the force model is unaware of the coupling. |
| F9 | LOW-MED | **The damping term aliases.** The sim moves the steering at 60 Hz; the FFB thread polls `car_feel` at 200 Hz. Two polls in three see no change and report rate 0; the third divides the whole delta by a 5 ms interval and reports ~3.3× the true rate. So damping is a train of spikes at a 1/3 duty cycle, not a smooth force. Cheapest honest fix: low-pass the rate, or take it from the wheel input rather than the sim angle. Two tests characterise this and are labelled as pinning current behaviour, not endorsing it. |
| F10 | LOW | **Every jolt saturates.** `IMPACT_FORCE == FORCE_LIMIT == 1.0` and the jolt always shares sign with the centring and aligning terms, so a cornering impact always clips and for ~0.2 s the wheel carries no steering information. |
| F11 | LOW | **Dead defensive branch.** `if impact_age_s is not None and impact_age_s >= 0.0` — the loop derives the age from a monotonic clock and cannot produce a negative. Assert or delete; deliberately not pinned so it is free to remove. |
| F12 | LOW | **Unsynchronised read.** The FFB thread calls `env.get_human_state()` while the main thread is inside `env.substep()` mutating the same `Agent`. Benign in practice for aligned floats on aarch64, but it is a data race. `car_feel`'s `_last_steering_angle_rad` is safe only because `run_demo.py` is its sole caller — an invariant nobody had written down until now. |

### Flagged to the user, deliberately untouched

- **`drive_test_env_config` fixture gap.** It leaves `max_speed_mps = 0` and
  `conditioning_accel_scale = 0`, producing a physically impossible env handed to tests about real
  dynamics. This cost three separate agents about an hour each. Not fixed because it is shared test
  infrastructure well outside this feature's blast radius.
- **`reset_accel_on_stop` default mismatch** between the YAML and `drive.py:84`.

---

## 4. Bugs worth remembering (they will recur)

- **The sim is positive-LEFT.** Positive curvature, `accel_lat` and `steering_angle` all mean a left
  turn (`constants.h`: "lane_curvature positive = left turn"). Every driver-facing surface must
  negate. This was inverted in **three** places — the sim, the window HUD and the terminal HUD — and
  every existing test was blind to it because they all asserted on `steering_angle` itself. Tests now
  assert in driver terms. The G920's own force sign is also inverted relative to intuition:
  `+0x2800` turns the wheel **left**.
- **`Py_BuildValue` format/argument mismatches segfault silently.** `render_poll_keys` declared seven
  `s:O` slots and supplied six pairs, reading a garbage pointer off the stack. It survived every test
  because `poll_keys` returns early when no window is open and **no test ever opens a window**. If
  you add a key, count both sides.
- **The C render tests redirect every raylib call to a capture stub.** Add a new raylib draw call to
  `render.h` and you must add a matching stub in `tests/drive/test_drive_render.c`, or the test
  reaches real raylib headless and segfaults.
- **`PyErr_Format` has no `%f`.** Use `snprintf` + `PyErr_SetString`.
- **`DrawPlane` in a Z-up world draws a vertical wall.** The ground is a flat `DrawCube` slab.
- A `git checkout` in this repo can hang for 10+ minutes on the `cgc` post-checkout hook. Use
  `git -c core.hooksPath=/dev/null ...`.

---

## 5. Known gap: scenery

The maps contain **road networks only**. Every town JSON holds exactly `lane`, `road_line` and
`road_edge` elements — 386 in Town01, 11,565 in Town11. There are no buildings, trees or meshes
anywhere in the repo; the only `.glb` assets are 6 cars plus a pedestrian and a cyclist.

"Roads don't load as I move around" has a specific cause and it is **not** culling: only
`LANE_FREEWAY` and `LANE_SURFACE_STREET` get a drawn surface, so junction aprons, slip roads of other
types and parking areas render as bare ground even though the sim has geometry there.

Rendering through CARLA's own Unreal renderer is not viable on macOS (no Apple Silicon build) and is
a large change anywhere. The realistic options, undecided:

1. Export CARLA environment-object bounding boxes once (`world.get_environment_objects()`), ship a
   small `.bin` per town, draw buildings as blocks and trees as cones. Needs CARLA access once.
2. Generate scenery procedurally from the map's negative space. No CARLA, no assets, deterministic
   from the seed; looks like *a* city, not like CARLA's.
3. Draw the non-drivable road elements too — cheapest fix for the specific "road vanishes" symptom.

---

## 6. What still needs a human

Unchanged from plan §3a, and none of it is data that can be transferred:

1. **Force-feedback feel.** `ALIGN_TORQUE_GAIN`, `CENTRING_GAIN`, `DAMPING_GAIN`,
   `DEFAULT_FORCE_GAIN`. Currently plausible guesses, output at 55%. Nobody has ever felt them.
2. **Whether the grip-loss cue reads** — aligning torque peaks at the limit then falls away.
3. **Whether driving it is any good.**
4. **Whether the new brake curve is right.**
5. **Which town to ship.**
6. On macOS, Input Monitoring must be granted per machine; on Linux, a udev rule (see setup §7.3).

---

## 7. Working agreement in force

From the original brief, and it has caught real defects every time:

> Tests are written by an **independent agent**, from the contract and the code rather than from my
> claims, and **nothing counts as done until they pass**. Every suite up to and including the current
> milestone runs, not just the new tests.

The agents mutation-test their own assertions — deliberately break the source, confirm the test
fails, restore. The M6 round wrote 131 tests and killed 25/25 mutants; the follow-up added 27 more
and killed 26/26. That discipline is why the dead collision pulse and the unreachable rumble were
found rather than shipped.
