# Wheel demo — handoff

State of the steering-wheel demo as of **2026-10-30 (evening, Linux)**, written so that a fresh session
on another machine can continue without the conversation history.

Read [`wheel_demo_plan.md`](wheel_demo_plan.md) for the design and the milestone definitions, and
[`wheel_demo_setup.md`](wheel_demo_setup.md) to get a machine running.

Branch: `aditya/wheel_demo`, off `3.0`.

---

## 1. Where the work stands

| Milestone | State |
| --- | --- |
| M0 Hardware probe and scaffold | done |
| M1 Input layer | done; Linux backend added (hidraw) |
| M2 `CONTROLLER_HUMAN`, direct steering | done |
| M3 Powertrain: clutch, gears, two modes | done; automatic gained a D/R selector |
| M4 Real-time loop and sub-stepping | done, **except input recording** (see §3) |
| M5 Renderer and cameras | **replaced by CARLA rendering** (§2) |
| M6 Force feedback | done; Linux transport added, rebalanced after the first hands-on feel |
| M7 Reward tuning and levels | done, now tested (41 tests, 34/34 mutants) |
| M8 Calibration, polish, determinism | **not started** — run after M9, by the user's ordering |
| M9 Steering assist | implemented: DISPLAY, BLEND, NUDGE, FULL; feel not yet judged |

### Expected test results

```
make -C tests/drive           ->  All test groups passed in every binary, 0 warnings
python -m pytest -q tests/unit_tests   (in .venv312 with SDL_VIDEODRIVER=dummy, and in a 3.13 venv)
                              ->  all pass; CARLA-live tests skip unless WHEEL_DEMO_CARLA_TESTS=1
```

The old triton failure (`test_action_selection.py`) does not occur on Linux: triton ships with the
CUDA torch wheel.

### Policy integrity

The jerk branch of `move_dynamics` was extracted into a pure `integrate_jerk` so the steering assist
can reuse the exact chain. Verified **bit-identical** to the pre-refactor code: the same trajectory
program compiled against `HEAD` and the working tree gives identical state hashes over 300 steps x 32
agents for continuous, discrete, phantom-braking, eval-deadband and 18-substep runs, at `-O0` and
`-O2 -ffp-contract=off`; a 0.0001 rad change to the steering limit flips every hash. Hash explicit
float fields, never raw `Agent` bytes — the struct has padding and the bytes differ run to run.

---

## 2. What was done on 2026-10-30 (Linux)

### Port

- **HID:** pip `hidapi`'s `import hid` is libusb on Linux (usage page 0, and opening it detaches the
  kernel driver); `g920.hid_module()` uses `hidraw` there. All three G920 collections share one
  `/dev/hidraw*`.
- **Force feedback:** `EvdevForceFeedback` drives the kernel's `hid-logitech-hidpp` through evdev
  `FF_CONSTANT` (no root, no conflict with the driver). Sign verified by hand.
- **Permissions:** the udev rule must be numbered below 73 (`70-logitech-g920.rules`).
- **Buttons:** all 14 wheel controls mapped on Linux (`wheel_buttons.json`); the probe gained
  `--only` to re-capture just the controls that failed.

### CARLA rendering (replaces the raylib renderer)

`pufferlib/wheel_demo/carla_scene.py`, `carla_window.py`, `hud.py`. The sim stays the only source of
truth: every sim car is a physics-less CARLA actor posed from the sim each frame, CARLA's traffic
lights are frozen and mirror the sim's (new `get_traffic_light_states`), and nothing writes sim state.

- Alignment through `carla_bridge.calibrate_town_offset`: lane residual 0.00-0.004 m.
- **Ground-following, render only:** height and pitch come from CARLA's OpenDRIVE surface (client-side
  `get_waypoint`, with the sim's z choosing between bridge and road below), eased over 0.12 s. The
  sim's z jumps up to 113 cm in one frame on Town05; the render's largest frame-to-frame change is
  2.3 cm and road contact is median 0.0 cm (ray-cast against the road mesh).
- **Bug worth remembering:** CARLA reports waypoint pitch in [0, 360). Easing between 1 and 359 degrees
  flips cars over a crest; it is wrapped.
- Views: chase / driver / top-down (60 m span, closer than before), each keeping its tuned rig. Driver
  view has a rear-view and two side mirrors cut from one wide rear camera (each extra camera costs a
  full scene render), and a steering wheel drawn at the real wheel angle.
- HUD: speed, gear, throttle and brake bars (no RPM, no angles), goal arrow + distance, speed-limit
  sign with an over-limit warning on the sim's own test, minimap, traffic level, force-feedback gain,
  assist readout, wheel-button help.
- Night: weather + street and building lights, headlights (high beam on your car), brake and
  reversing lights; camera exposure clamped to EV 8.5-11 at night.
- Frame rate on an RTX 4080 with the policy on the GPU: chase and top-down 60 fps, driver view ~50 fps;
  the sim always runs in real time and drops frames instead.
- Ego car is the Dodge Charger 2020 (the Lincoln's glass roof reads as roofless from the chase cam).

### Driving changes from the first hands-on drive

- Throttle `throttle_response_gamma` 1.6 (half pedal = 33% torque); brake `brake_response_gamma` 0.4
  (a quarter press = 56% braking).
- Force model: saturating centring (full at rest, 38% of device output at the default gain, easing to
  30% of itself by 10 m/s, and collapsing with the grip like the aligning torque); steady forces
  soft-limited to 0.75 so rumble and the impact jolt keep headroom; default gain 0.7. The overspeed
  vibration was built and then **removed at the user's request**; the HUD warning stays.
- Automatic gearbox: left paddle at a standstill selects R, right paddle selects D.
- Wheel buttons drive the demo (table in setup §8).

### M7 fixes (independent agent findings)

"Normal" now restores the exact training distribution (the override flag cleared on the trained
window); the setter rejects windows for coefficients the sampler never reads and refuses when reward
randomisation is off (so `apply_level` is atomic); the binding rejects non-contiguous arrays.

### M9 steering assist

The reference is the policy's own action for the human car pushed through `integrate_jerk` from the
car's real state, once per policy step; it exists only while kept fresh, so re-enabling an assist never
acts on a stale target. Modes (Menu button): OFF; DISPLAY (HUD only); BLEND steers the car's road
wheels to `(1 - w) * yours + w * policy`, `w` easing in and out over 0.5 s (your physical wheel does not
move); NUDGE adds a gentle force toward the policy's angle; FULL re-centres the spring on the policy's
angle and adds a strong servo pull, so hands-off the wheel turns itself and steers the car along the
policy's line (the driver overrides by holding it). Steering only: the pedals are not motorised.

### Launcher and towns

- Desktop launcher (`scripts/wheel_demo/launcher.py`, installed by `install_launcher.py`): every
  setting with defaults pre-selected, starts or restarts CARLA as needed, then opens the demo.
- All 8 towns render, from a private CARLA copy with AdditionalMaps imported. Town07 had no offset in
  `carla_bridge.TOWN_OFFSETS` (co-sim never ran on it); derived by the same grid search as the others,
  lane residual 0.005 m. Alignment on all 8: 0.000-0.011 m.
- Goals are drawn in the 3D view: each CARLA image carries the camera pose it was rendered from, so
  the goal is projected into exactly the frame on screen.
- Driver view draws a steering wheel on the driver's side at the angle of the driver's hands.

---

## 3. Open items

| # | What | Notes |
| --- | --- | --- |
| O1 | **M8 not started** | pause/reset do not exist; final steering-calibration check (FFB can turn the wheel itself, so no hands needed) |
| O2 | **M4 input recording missing** | the plan says every WheelState plus the seed is written to a trace; `run_demo.py` does not. M8's "replay a real captured session" needs it |
| O4 | Old raylib renderer | `pufferlib/ocean/drive/render/` and its tests are unused by the demo; delete or keep is the user's call |
| O5 | Skidding | the human car understeers past the grip limit but never oversteers or slides (kinematic bicycle). A dynamic tyre model was offered; undecided |
| F7 | Collision jolt up to ~300 ms late | collision is only computed in `c_step`, not `c_substep` |
| F8 | No jolt when hitting something while off road | offroad and collision flags are mutually exclusive in `compute_metrics` |
| F9 | Damping spikes | steering rate sampled at 200 Hz from a 60 Hz sim |
| F11 | Dead `impact_age_s >= 0.0` branch | unchanged |
| F12 | Unsynchronised FFB read of the human agent | unchanged |

F10 (every jolt saturates) is resolved: the jolt now opposes the steady force, which has headroom.

### Flagged, deliberately untouched

- The assist readout's "you" and "policy" are rounded separately from the bracketed delta, so the
  three numbers can disagree by 1 degree.
- `probe_g920.py --only`: a re-capture landing on a bit another control already holds refuses to
  write (fail-fast) and the session's captures are lost; re-run the whole probe in that case.
- `launcher.py` repeats the 0.3 s policy step to turn minutes into steps; it would drift if the
  checkpoint's `dt` changed.
- Wheel buttons are only polled with `--render` (the headless loop is for tests).

- `drive_test_env_config` leaves `max_speed_mps` and `conditioning_accel_scale` at 0.
- `reset_accel_on_stop` default differs between the YAML and `drive.py`.
- `vec_set_traffic_light_states` does not check the array dtype.
- `WheelCalibration.to_json` writes the response gammas, so re-saving a calibration pins the current
  defaults into the file.

---

## 4. Bugs worth remembering (they will recur)

- **The sim is positive-LEFT.** Positive curvature, `accel_lat` and `steering_angle` all mean a left
  turn. Every driver-facing surface negates; CARLA's own steer and yaw are positive-RIGHT. Assert in
  driver terms. The G920's force sign is inverted too: `DEVICE_FORCE_SIGN = -1` on both transports.
- **`Py_BuildValue` format/argument mismatches segfault silently.** Count both sides; a throwaway
  parser that pairs specifiers with arguments is cheap insurance.
- **`PyErr_Format` has no `%f`.** Use `snprintf` + `PyErr_SetString`.
- **CARLA renders on Vulkan device 0** — on a CPU with integrated graphics that is the iGPU.
- **Never rebuild the extension while a demo is running**; the build overwrites the loaded `.so`.
- **Only one client may drive the CARLA world.** Tests and scripts must stay off it while someone drives.
- A `git checkout` in this repo can hang for 10+ minutes on the `cgc` post-checkout hook. Use
  `git -c core.hooksPath=/dev/null ...`.

---

## 5. What still needs a human

1. **Force-feedback feel** at the new defaults: centring strength, grip-loss lightening, rumble, jolt.
2. **Pedal curves**: whether the tamer throttle and the earlier brake are right.
3. **Whether the traffic levels are distinguishable** from the driver's seat.
4. **Assist**: blend versus nudge (plan M9: decide by trying both).
5. **Which town to ship.**

---

## 6. Working agreement in force

> Tests are written by an **independent agent**, from the contract and the code rather than from my
> claims, and **nothing counts as done until they pass**. Every suite up to and including the current
> milestone runs, not just the new tests.

The agents mutation-test their own assertions. This session's rounds found, among others: "normal"
never restoring the training distribution, a setter silently accepting windows nothing reads, the new
centring spring masking every other force cue, a libusb test call that could detach the kernel driver,
and a float32/float64 disagreement at the speed-limit threshold.
