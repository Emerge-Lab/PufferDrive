# Wheel Demo — Implementation Plan

Branch `aditya/wheel_demo`, sourced off `3.0` @ `2c1ca3e4`.

**Goal.** Drive one car on any of the eight CARLA towns with a real steering wheel and
pedals, while every other car is driven by a frozen PufferDrive policy under jerk dynamics. Egocentric
rendering with agent / chase / driver cameras. Live reward-conditioning tuning so nearby
traffic changes character mid-drive ("game levels"). Force feedback you can feel.

> **This document is the design and the milestone definitions — it is not a status report.**
> For where the work actually stands (milestones complete, open findings, known gaps), see
> [`wheel_demo_handoff.md`](wheel_demo_handoff.md). To get a new machine running, including the
> Linux porting notes, see [`wheel_demo_setup.md`](wheel_demo_setup.md).

---

## 0. What the code actually looks like today

Findings from reading `3.0`. These drive every design decision below; where a decision is
forced by one of them, it is cited.

**F1 — Jerk dynamics steer indirectly.** `move_dynamics` (`drive.h:5294`) under
`DYNAMICS_MODEL_JERK` takes lateral *jerk* as the action, integrates it to lateral
acceleration, converts to curvature `a_lat / v²`, then to a steering angle
`atan(curvature · wheelbase)`, then applies a **±0.6 rad/s rate limit** and a **±0.55 rad
position limit**. There is no way to command a steering angle through the action space.

**F2 — The policy trains at `dt = 0.3 s`** (`pufferlib/config/puffer_drive.yaml:54`, and
`weights/k_scaled_0036_1000/config.yaml:21`). 3.33 Hz. Env action type is `continuous`,
policy head is `discrete` with an `action_table` mapping classes to `[-1, 1]` pairs
(`ocean/torch.py:507`).

**F3 — Per-agent reward conditioning is already a live input to the policy.**
`Agent.reward_coefs[NUM_REWARD_COEFS]` (18 coefficients, `constants.h` §9) are written
straight into the observation by `write_reward_target_obs` (`drive.h:4819`), normalised
against `REWARD_BOUNDS`. Overwrite the array and the policy sees it on the next step. They
are currently only assigned once, at spawn, by `generate_reward_coefs` (`drive.h:2586`,
called from `spawn_agent` at `drive.h:3267`).

**F4 — There is a per-agent controller enum already.** `CONTROLLER_STATIC / POLICY /
REPLAY / IDM` (`constants.h` §2), dispatched by `if/else` in `c_step` (`drive.h:5753`).
Adding a controller is a known-shaped change.

**F5 — Native rendering was deleted on 3.0.** Commit `9f717623` (#593) removed
`render.h` (1800 lines), `visualize.c`, `egl_headless.h`, the car `.glb` models,
`drivenet.h` and `puffernet.h`. raylib is no longer referenced by `setup.py`, and
`extern/raylib-5.5_macos` is untracked local-only. Today's visualisation is offline: frames
are captured into numpy arrays via `get_obs_html_frame` (`drive.py:1177`) and rendered into
an HTML viewer by `pufferlib/viz.py`.

**F6 — The chase/driver cameras are on `aditya/trajectory_ablation`, not on 3.0.** That
branch is 41 commits ahead of the 3.0 merge base and adds `replay_format.py`,
`replay_video.py` (a numpy+PIL software 3D rasteriser with an `AgentViewCamera`) and a
Canvas2D agent-view in `viz.py`. The camera presets live in
`REPLAY_VIEW_STYLE["cameras"]`: `chase` (back 25 m, eye 15 m, ahead 40 m, fov 45°,
draws ego) and `driver` (back 0 m, eye 1.4 m, ahead 30 m, fov 60°, hides ego).

**F7 — Keyboard human control existed on 2.0.** `drive.c` read `IsKeyDown(KEY_UP)` etc.
and wrote discrete jerk indices into `actions[env.human_agent_idx]`, with SPACE to switch
the controlled agent. `human_agent_idx` survived into pre-#593 3.0 but only as a camera
focus. Useful precedent for the agent-switch UX; not reusable as a control path (F1).

**F8 — Non-active agents are still seen and still collide.** Both `collision_check`
(`drive.h:2129`) and `write_partner_obs` (`drive.h:4875`) iterate `env->num_agents`, which
spans active *and* static agents. Partners will observe the human car regardless of which
controller it carries.

**F9 — Partners observe 9 (+2) features per neighbour**, written at `drive.h:4972`:
relative x/y, dz, length, width, relative heading cos/sin, signed speed, seconds stopped,
and relative velocity x/y. **Steering angle and acceleration are *not* observed.** The
in-distribution constraint on the human car is therefore about pose, heading rate, size and
speed — not about how we generate its acceleration.

**F10 — `c_set_agent_states` already accepts a steering angle** (`drive.h:4106`) and does
the right bookkeeping: `copy_pose_to_prev`, `refresh_lane_association`, `accel_lat` derived
from steering. It is a teleport setter built for co-sim.

**F11 — The co-sim stack is the right architectural precedent.**
`ocean/cosim/carla/shadow_ego.py` runs "external simulator drives one ego, PufferDrive
policy does the thinking", and `ocean/cosim/arch.py` builds the env from *the checkpoint's
own `config.yaml`* plus `CLEAN_EVAL_OVERRIDES`, so the obs layout can never drift from the
frozen policy. We reuse that loader wholesale.

**F12 — Test harness is in place and fast.** `tests/drive/` is a plain C harness
(`EXPECT_NEAR`, `EXPECT_EQ_INT`) with a `Drive` fixture already pointed at
`carla/opendrive__Town01.bin`. `make test-c`, `make test-unit`, `make test` at the root.
CI runs unit + eval + C-drive + notebooks on PRs into `3.0`.

**F13 — The wheel is a Logitech G920 Driving Force Racing Wheel for Xbox One**, VID
`0x046D` / PID `0xC262`, found on this machine via IOKit. 900° lock-to-lock, three pedals
(throttle / brake / clutch), F1-style paddle shifters. No H-pattern shifter attached.

---

## 1. Decisions taken

| # | Decision | Why |
|---|---|---|
| D1 | **New `CONTROLLER_HUMAN`** in C, not Python-side teleporting via `c_set_agent_states` | Vehicle dynamics are sim logic and belong in C (CLAUDE.md: no Python in hot paths, determinism is a guardrail). F10's setter teleports, which zeroes the swept-collision displacement. |
| D2 | **The human car does not use jerk dynamics at all, on either axis.** Steering angle comes straight from the wheel; longitudinal acceleration comes straight from the powertrain. Neither goes through a jerk integrator | F1. A lateral-jerk servo can never be perfectly calibrated — it always lags. You drive the car the way a human drives a car. The jerk model stays where it belongs: on the policy-driven traffic. |
| D3 | **No artificial steering rate limit on the human path.** The physical wheel is the rate limiter | F1's ±0.6 rad/s is ~10× slower than a human's hands. Keeping it would make the wheel feel disconnected, which is the one thing this demo cannot afford. |
| D4 | **Sub-step every agent** at ~60 Hz; the policy still decides at 3.33 Hz and its action is held across the sub-steps | Revised after measurement (was human-only). Sub-stepping repeats dynamics and collision but NOT observation generation, which is the expensive part: the loop costs 6.3 ms of policy + 0.2 ms of env against a 300 ms budget, so ~150 cars at 18 sub-steps is a few ms. Fidelity is not a trade either — holding a policy action constant across sub-steps gives *exactly* the same velocity as one big step (trapezoidal integration of a linearly ramping acceleration is exact) and a strictly more accurate pose. `SUBSTEP_MODE_HUMAN_ONLY` stays as a fallback. |
| D5 | **No render-side extrapolation** under `SUBSTEP_MODE_ALL`: every car's pose is genuinely fresh at 60 Hz | Superseded by D4. Dead reckoning is only needed under `SUBSTEP_MODE_HUMAN_ONLY`, where partners would otherwise move at 3.33 Hz; it is kept with the fallback mode and must never feed back into sim state. Note neither mode changes how fast partners *react* — that is the policy's 3.33 Hz decision rate. |
| D6 | **Policy inference stays in PyTorch, in Python; Python hosts the loop** | The C inference path (`drivenet.h`/`puffernet.h`) was deleted in #593 and predates the fused slot encoder. One forward pass per 300 ms for ~150 agents is nothing. |
| D7 | **Env built from the checkpoint's `config.yaml`**, reusing `cosim/arch.py` | F11. A hand-written env config that disagrees with the checkpoint produces a policy reading garbage, silently. |
| D8 | **Native raylib renderer in C**, restored from `9f717623^:render.h`, as a demo-only build target kept out of the training extension | User decision. Lowest latency, and 1800 lines of working PufferDrive drawing code already exist in history. |
| D9 | **Levels shift reward-coef *sampling bounds*, not fixed values** | Pinning every partner to one coef vector makes all traffic behave identically. Shifting the sampling window keeps driver-personality diversity while moving the mean. |
| D10 | **Input layer behind one interface with swappable backends** (macOS IOKit, Linux evdev/SDL2, and a synthetic/replay source) | Mac dev + Linux/Windows demo. The synthetic source is also what makes the whole loop testable headless in CI. |
| D11 | **CARLA renders the demo; supersedes D8** (2026-10-30) | User decision after driving it: real towns with buildings and trees give speed perception the road-only raylib view could not. The sim stays the only source of truth; CARLA actors are physics-less and posed from it each frame, CARLA's lights mirror the sim's, and heights come from CARLA's road surface for the render only. Needs Python <= 3.12 (CARLA client wheels). |

---

## 2. Architecture

```
                    ┌──────────────── Python host (60 Hz fixed-timestep loop) ────┐
   G920 ──HID──▶ input thread (~250 Hz)                                           │
                    │  axes, paddles, buttons ──▶ lock-free snapshot              │
                    │                                   │                         │
                    │                       binding.set_human_input(...)          │
                    │                                   ▼                         │
                    │                       binding.substep_human(env)  × N       │
                    │                                   │                         │
                    │              every N substeps:  obs ─▶ torch policy ─▶ acts │
                    │                                 binding.vec_step(env)       │
                    │                                   ▼                         │
                    │                       binding.render(env, camera_mode)      │
                    │                                   │                         │
   G920 ◀──HID──  FFB thread (~200 Hz) ◀── shared state snapshot                  │
                    └─────────────────────────────────────────────────────────────┘
```

New / changed files:

```
pufferlib/ocean/drive/
  constants.h             CONTROLLER_HUMAN, gearbox + steering + FFB constants
  datatypes.h             HumanControl state on Agent (or a parallel struct)
  drive.h                 move_human, c_substep_human, c_set_human_input,
                          c_set_reward_coef_bounds, c_select_human_agent
  binding.c               the new setters/getters
  drive.py                thin Python wrappers for them
  wheel_demo/
    render.h              restored from 9f717623^, + 3 cameras + HUD
    demo_main.c           (only if we later want a C-hosted variant)
    Makefile              demo-only target; raylib NOT added to setup.py

pufferlib/wheel_demo/
  __init__.py
  loop.py                 the fixed-timestep host loop
  policy_host.py          checkpoint → env kwargs (via cosim/arch) + torch policy
  input/
    base.py               WheelInput interface + WheelState dataclass
    g920_hid.py           raw-HID backend (macOS IOKit / hidapi)
    sdl_backend.py        SDL2/evdev backend (Linux, Windows)
    synthetic.py          scripted + recorded-trace sources for tests
  ffb.py                  pure state → force-command function + the writer thread
  levels.py               the 3 presets as reward-coef bound sets
  hud.py                  HUD field assembly

tests/drive/              test_drive_human_control.c, test_drive_human_powertrain.c,
                          test_drive_human_substep.c, test_drive_reward_tuning.c
tests/unit_tests/         test_wheel_input_mapping.py, test_wheel_gearbox.py,
                          test_wheel_ffb.py, test_wheel_levels.py,
                          test_wheel_cameras.py, test_wheel_loop_headless.py
```

---

## 3. Milestones

Each milestone states what it delivers and what proves it. **Nothing is done until its
tests pass.** Tests are written by an independent agent (§5).

### M0 — Hardware probe and scaffold

Dump the G920's HID report descriptor and a raw report stream; map every axis and button
(wheel, throttle, brake, clutch, both paddles, face buttons). Record a few seconds of real
reports to `tests/fixtures/g920_reports/` — these become the fixtures that let every input
test run with no hardware attached. Confirm whether the 900° range command and a constant-
force command are accepted on each target platform.

*Proves it:* a probe script prints a live, labelled axis/button table; the recorded trace
replays through the (not yet written) parser in a later milestone.

### M1 — Input layer

`WheelState` = `{steer_norm ∈ [-1,1], throttle, brake, clutch ∈ [0,1], paddle_up_edge,
paddle_down_edge, buttons, timestamp}`. `WheelInput` interface with `start/stop/latest()`.
Backends: `g920_hid` (hidapi/IOKit), `sdl_backend` (Linux/Windows), `synthetic` (scripted
and trace-replay). Reader runs on its own thread at ~250 Hz into a lock-free snapshot —
the loop never blocks on HID.

Deadzone, saturation and per-pedal calibration (min/max capture) persisted to a JSON
calibration file.

*Proves it:* `test_wheel_input_mapping.py` — recorded reports from M0 decode to expected
normalised values; axis endpoints map to exactly ±1 and 0; a stuck/absent device raises
rather than silently returning zeros (fail-fast).

### M2 — `CONTROLLER_HUMAN` and direct steering

- `CONTROLLER_HUMAN` added to `constants.h` §2. `c_step`'s move loop dispatches to `move_human`
  (M4's sub-stepping moves that out of the step loop), while metrics, rewards, observations and
  goals all include it — so the HUD gets infractions and the policy's would-be action for free (§6).
- `move_human(Drive *env, int agent_idx)`: **no jerk integrator on either axis** (D2). It
  takes the road-wheel angle the wheel is physically at, and the longitudinal acceleration
  the powertrain is physically making (M3), and integrates the car forward.
  The only thing shared with the jerk path is the **kinematic bicycle arc integrator**
  itself (`drive.h:5490-5520`) — the geometry of how a car with a given steering angle and
  speed moves through the world, which is not jerk and is true of every car on the map.
  Extracted into a small shared helper so both callers use one copy.
  `accel_lat`, `jerk_long` and `jerk_lat` are still *recorded* on the agent afterwards,
  because the comfort metric and the HUD read them — but nothing reads them back as input.
- Steering calibration: `HUMAN_STEERING_RATIO` maps wheel angle to road-wheel angle.
  Wheel range 900° (±450°); road-wheel limit `HUMAN_STEERING_LIMIT_RAD = 0.55` (matching
  the jerk path's position clamp) gives a ratio of **14.3:1**, inside the realistic
  14–18:1 band for a passenger car. No rate limit (D3).
- `c_select_human_agent(env, agent_idx)`: hands the previous human car back to
  `CONTROLLER_POLICY`, takes the new one, resets powertrain and wheel state. **Default is
  agent 0 (`EGO_IDX`), but any active agent on the map can be taken over** — not just the
  next one in a cycle. Three ways in: a wheel/keyboard button cycles, a direct index can be
  given, and the BEV camera lets you pick the car under the cursor. Switching mid-drive
  re-centres the wheel and zeroes the FFB before handing control over, so the wheel does not
  fight you at the moment of transfer.
- `c_set_human_input(env, steer_rad, throttle, brake, clutch, gear_request)`.
- CARLA-only guard: the demo refuses to start on a non-CARLA map, with an explicit error
  naming the map (CLAUDE.md fail-fast; map files are an untrusted boundary).

*Proves it:* `test_drive_human_control.c` — zero wheel is exactly zero road-wheel; ±full
lock is exactly ±0.55 rad; the mapping is monotone and symmetric; beyond-lock input clamps;
a human agent is absent from the move loop but present in `active_agent_indices`, metrics
and partner observations; selecting a new human agent restores the old one to
`CONTROLLER_POLICY`; a non-CARLA map aborts env creation.

### M3 — Powertrain: clutch, gears, two modes

New sim subsystem, one function mutating one thing (CLAUDE.md). All ratios, torque-curve
breakpoints and thresholds are named constants in `constants.h`; no raw numbers in logic.

- Engine: torque curve `T(rpm) · throttle`, idle control, rev limiter, stall below
  `ENGINE_STALL_RPM` with the clutch engaged.
- Transmission: 6 forward + reverse, final drive. `GEARBOX_MODE_MANUAL` /
  `GEARBOX_MODE_AUTOMATIC` as mode constants, selected by `if`/`switch` (no function
  pointers).
- **Manual** (F13: paddles, not H-pattern): paddle up/down requests a gear; the request
  only succeeds with the clutch pedal depressed past `CLUTCH_ENGAGE_FRACTION`, otherwise
  the shift is refused and torque is interrupted. Releasing the clutch in too high a gear
  at too low an RPM stalls the engine. *(H-pattern support can be added later if the
  Driving Force Shifter is bought — it reports as buttons on the same HID report, so it is
  an input-layer change only.)*
- **Automatic**: upshift/downshift schedule on RPM and throttle, with hysteresis so it does
  not hunt. Clutch pedal ignored.
- Longitudinal force: traction − brake − aero drag − rolling resistance, over mass, to
  `a_long`.
- **In-distribution clamp** (F9): `a_long` is clamped to `ACCEL_LONG_LIMIT` and speed to
  the same `base_max_speed_mps · C_vel` band every other car obeys. Partners do not observe
  our acceleration, but they do observe our pose and signed speed, and a car doing 60 m/s
  is an object the policy has never seen. Named, configurable, on by default.

**Also fixed in M3 (found while running M4's loop):** the human car's speed cap currently comes
from `reward_coefs[REWARD_COEF_SPEED]`, which `reward_randomization` draws at random per episode —
so your top speed changes every time you reset (measured 22.3 m/s on one run). Your car's
capability must not be randomly assigned. `c_select_human_agent` will pin the human agent's
conditioning coefficients to a fixed reference-driver profile, with the speed cap a config knob.
This also matters for M9: the assist reference runs the policy on the human car's own observation,
and those coefficients are part of that observation.

*Proves it:* `test_drive_human_powertrain.c` — full throttle in gear 1 gives the expected
acceleration within tolerance; clutch fully depressed transmits zero torque; a paddle
upshift without clutch in manual mode leaves the gear unchanged; releasing the clutch below
stall RPM stalls; the rev limiter caps RPM; automatic mode reaches the expected gear at a
given speed and does not oscillate across the shift point; reverse produces negative
signed speed bounded by `MAX_BACKWARD_SPEED`; `a_long` never leaves `ACCEL_LONG_LIMIT`.

### M4 — Real-time loop and sub-stepping

- `c_substep(env)` advances every agent by `substep_dt = dt / SUBSTEPS_PER_POLICY_STEP`, running
  collision and offroad checks **every sub-step**. Policy agents re-integrate their held action;
  the human car re-reads the wheel. Under `SUBSTEP_MODE_HUMAN_ONLY` only the human advances.
  At 20 m/s a 300 ms step moves 6 m — more than a car length — so without per-sub-step
  checking the human car tunnels through traffic. At 60 Hz it moves 0.33 m.
- Collisions found mid-window are latched and applied to **both** cars at the policy
  boundary, so the partner's metrics and infraction behaviour stay correct even though the
  partner itself only moved once.
- `SUBSTEP_MODE_ALL` is the default; `SUBSTEP_MODE_HUMAN_ONLY` is the fallback (D4).
- Fixed-timestep accumulator in `loop.py`: `HUMAN_SUBSTEPS_PER_STEP` is a constant, never
  derived from wall-clock, so a given input trace replays identically (CLAUDE.md
  determinism). If the machine falls behind, render frames are dropped — the sub-step dt
  never changes.
- Per-step RNG (phantom braking, pose noise, partner blindness) continues to fire **once
  per policy step**, not per sub-step, so the robustness-feature statistics the policy
  trained against are unchanged.
- Policy host: checkpoint → `cosim/arch.shadow_env_kwargs` → `Drive` env; `torch.load` →
  `Drive` policy → `.eval()`; discrete logits → `discrete_actions_to_continuous` → the
  env's continuous action buffer. The human agent's row is computed and **kept** (M9 needs
  it) but not applied.
- Input recording: every `WheelState` plus the seed is written to a trace file, so any run
  can be replayed exactly for debugging and for regression tests.

*Proves it:* `test_drive_human_substep.c` — under constant input, N sub-steps and one full
step agree on velocity to float tolerance and on pose within a stated bound *(the jerk
integration is exactly equal under sub-stepping for velocity: trapezoidal integration of a
linear acceleration ramp is exact, and the arc integration is strictly more accurate at
smaller dt)*; a human car driven at 20 m/s into a stationary partner registers a collision
rather than passing through; a mid-window collision appears in both agents' metrics after
the policy boundary; the same seed plus the same input trace produces a bit-identical pose
hash over 1000 steps. `test_wheel_loop_headless.py` runs the whole loop against the
synthetic input source with no hardware and no window.

### M5 — Renderer and cameras

- Restore `render.h` from `9f717623^` into `wheel_demo/`, as a demo-only target. raylib is
  vendored for that target only and is **not** re-added to `setup.py` — training builds
  stay render-free, which is what #593 was for.
- Three cameras as mode constants: `CAMERA_AGENT` (the existing orthographic BEV at
  vision-range zoom), `CAMERA_CHASE`, `CAMERA_DRIVER`. Chase and driver reuse the exact
  parameters already tuned on `aditya/trajectory_ablation`
  (`REPLAY_VIEW_STYLE["cameras"]`, F6) so the live view matches the offline renders:
  chase = back 25 m / eye 15 m / ahead 40 m / fov 45° / ego drawn;
  driver = back 0 m / eye 1.4 m / ahead 30 m / fov 60° / ego hidden.
- **The rendered front wheels are drawn at the agent's actual `steering_angle`.** This is
  explicitly part of "the turn in the renders should be in line with the actual turn" and
  is asserted by a test, not eyeballed.
- Partner dead reckoning for render only (D5), behind a clearly named function that takes
  a const agent and returns a pose, so it cannot write sim state.
- HUD: speed, RPM, gear, clutch state, selected camera, selected agent, current level,
  live infraction flags (collision / offroad / red light / stop sign, straight out of
  `metrics_array`), and the assist readout placeholder (M9).

*Proves it:* `test_wheel_cameras.py` — camera eye/target/up for a known pose match the
closed-form expectation for all three modes; the driver camera sits inside the ego box and
excludes it from the draw list; the drawn front-wheel angle equals `steering_angle` for a
swept range of inputs; dead reckoning over a full window stays within a stated bound of the
true next pose and leaves `Agent` untouched (asserted by comparing a struct hash).

### M6 — Force feedback

Pure function `compute_ffb(sim_state, wheel_state) → force_command`, so it is testable with
no hardware; a thin writer thread pushes it at ~200 Hz on whichever backend is live.

Effects, in order of importance to feel:
1. **Self-aligning torque** — the main one. Grows with steering angle and with v², saturates
   at the front-tyre grip limit. This is the weight that builds through a corner.
2. **Grip loss** — the saturating term going soft is what tells your hands the front has
   let go. Falls out of (1) if the saturation is modelled rather than clipped.
3. **Centring spring** at low speed, **damping** throughout.
4. **Impacts** — a constant-force pulse on collision; a rumble while offroad or over a
   kerb.

**Both platforms get haptics.** Two backends behind one `compute_ffb` function.

*Linux / Windows* — the kernel's `hid-logitech-hidpp` driver (or DirectInput on Windows)
exposes the G920 as a standard force-feedback device, reachable through evdev or SDL2.
Constant force, spring, damper and periodic effects come for free. Low risk.

*macOS* — there is no driver, so we speak to the wheel directly over raw HID via `hidapi`
(which sits on IOHIDManager). The important detail: **the G920 is not an lg4ff-class wheel.**
Unlike the G29/G27, it is handled in Linux by `hid-logitech-hidpp.c`, and its force feedback
is **HID++ 2.0 feature `0x8123` (G920 Force Feedback)**, not the classic 7-byte Logitech
command set. So the macOS backend must:
  1. open the device with hidapi and resolve feature `0x8123` to a runtime feature index via
     the HID++ `IRoot.GetFeature` call on feature `0x0000`;
  2. send effect download/play/stop commands to that feature index, using HID++ short
     (report `0x10`) and long (report `0x11`) reports;
  3. set the rotation range to 900° so the calibration in M2 holds.

Every command is **transcribed from the kernel source** (`hidpp_ff_init` and friends in
`drivers/hid/hid-logitech-hidpp.c`), not written from memory, and each one is validated
against the real wheel in the M0 probe before any other milestone depends on it. macOS also
requires granting **Input Monitoring** to the terminal or app in System Settings → Privacy
& Security.

The M0 gate is deliberately early: we learn in hours, not weeks, whether a constant force
command makes the wheel push back. If HID++ on macOS turns out to be blocked by something
outside our control, the loss is contained — axes, pedals, rendering and the whole sim work
on macOS regardless, and the Linux demo path is unaffected.

*Proves it:* `test_wheel_ffb.py` — zero speed and zero steering gives zero force; force
opposes the steering direction (sign test across the range); magnitude is monotone in
steering angle below saturation and flat above it; magnitude grows with speed; a collision
event produces a pulse that decays; output is always inside the device's force range.

### M7 — Reward tuning and levels

- `c_set_reward_coef_bounds(env, min_vals, max_vals)` + resample: rewrites every
  policy-controlled agent's `reward_coefs` from the new bounds, mid-episode. F3 means the
  policy sees them on the very next step — no reset, no reload.
- **Hard constraint:** every value must stay inside `REWARD_BOUNDS` (or `REWARD_BOUNDS_LOG`
  when `reward_log_sampling` is set). Outside that window `write_reward_target_obs` clamps
  the normalised value to ±1 and the conditioning signal saturates — the policy goes out of
  distribution and the behaviour change you asked for silently stops happening. The setter
  rejects out-of-range bounds rather than clamping (fail-fast).
- Three presets in `levels.py`, each a set of bounds over the aggression-bearing
  coefficients: `COLLISION` (low = reckless), `SPEED`/C_vel (high = fast), `OVERSPEED`
  (low = speeders), `COMFORT` (low = jerky), `LANE_CENTER` / `LANE_ALIGN` (low = sloppy),
  `STOP_LINE` (low = runs lights), `ACC` / `THROTTLE` (high = punchy).
- Bound to keys/wheel buttons so levels can be switched **while driving**, which is the
  point — you should be able to feel the traffic turn nasty without stopping.

*Proves it:* `test_drive_reward_tuning.c` — bounds outside `REWARD_BOUNDS` are rejected;
after a live set, the conditioning slice of a partner's observation changes and the new
values round-trip through the normalisation; coefficients of the human car are not
disturbed; no allocation happens in the setter. `test_wheel_levels.py` — all three presets
lie inside `REWARD_BOUNDS`; applying a preset to a real env and stepping produces
measurably different partner behaviour (e.g. mean speed, mean time-to-collision) from the
calm preset, same seed.

### M8 — Calibration, polish, determinism

End-to-end steering calibration check against the physical wheel: command a known wheel
angle, read back the road-wheel angle, the rendered wheel graphic and the yaw rate
achieved, and assert all three agree. Agent-switch UX. Pause/reset. Trace record/replay
proven on a real captured session.

---

### M9 — Steering assist (last milestone)

Built last, but built. Everything before it is designed so this drops in without rework.

**Yes — the reference comes from the policy, which means it comes through jerk.** The human
car does not use jerk (D2), but the *policy* only speaks jerk, so the assist reference is
produced by running the policy's jerk action through the standard jerk chain and reading
off what it would have meant physically.

Concretely, every policy step:

1. The frozen policy already runs on the human agent's own observation — the human car is
   an active agent, so its action row is computed anyway. M4 keeps that row instead of
   discarding it. **No extra network, no extra inference, no extra cost.**
2. That action is a `(j_long, j_lat)` jerk pair. Push it through exactly the chain
   `move_dynamics` uses: `j_long → a_long`, `j_lat → a_lat → curvature = a_lat / v² →
   steering = atan(curvature · wheelbase)`, including the rate and position clamps.
3. Out comes **the road-wheel angle and longitudinal acceleration the policy would have
   used from exactly where you are sitting** — in the same units as your own wheel and
   pedals, so the two are directly comparable.

The chain is stateful: it integrates from the car's *current* `accel_long`, `accel_lat` and
`steering_angle`. This works without any shadow state because M2 already records those
three fields on the human agent from the human model (the comfort metric and HUD read
them). So the jerk chain runs straight on top of the car's real physical state, and the
reference answers the right question: *from this exact pose, speed and acceleration, one
step from now, what would the policy do?*

Three modes, as a mode constant:

- **ON** — the correction is applied. Either blended into the commanded steering
  (`(1-k)·human + k·policy`) or, better for feel, applied as an FFB torque nudging the
  wheel toward the policy's angle, so the car still does what your hands do but your hands
  are guided. Decide by trying both.
- **DISPLAY ONLY** — nothing is applied; the HUD shows the **difference** between your
  steering angle and the policy's, and between your acceleration and the policy's, so you
  can adapt yourself. This is the interesting one for a study.
- **OFF** — reference not computed or not shown.

*Proves it:* `test_drive_steering_assist.c` / `test_wheel_assist.py` — with assist OFF the
commanded steering is bit-identical to the human input; with assist ON at `k = 0` it is
also bit-identical (no silent drift), and at `k = 1` it equals the policy reference; the
displayed delta equals `policy_reference − human_command` for a swept range; mode switching
mid-drive never produces a discontinuity larger than a stated bound.

---

## 3a. What needs a human

Everything else can be built and verified unattended. These cannot:

1. **Force-feedback FEEL (M6).** The transport is proven: the wheel moves on command, the sign is
   correct, and a cornering car pushes the wheel back towards centre. What a machine cannot judge is
   whether it feels like a car. Four numbers need hands on the wheel:
   `ALIGN_TORQUE_GAIN` (how heavy it gets), `CENTRING_GAIN` (self-centring strength at low speed),
   `DAMPING_GAIN` (how dead or lively it feels), `DEFAULT_FORCE_GAIN` (overall strength). I have set
   them to plausible values and the demo opens at 55% of the device's full output.
2. **Whether the grip-loss cue reads.** The aligning torque peaks at the limit and then falls away.
   Whether that drop is legible through your hands, or needs to be sharper, is a judgement only a
   driver can make.
3. **Whether driving it is any good.** The point of the exercise.
4. **The brake pedal's usable range.** Calibration bottomed at raw 53 under a hard press but only 93
   in normal driving. Whether full braking is comfortably reachable, or the floor should be raised,
   needs a foot.
5. **Which CARLA town to ship.** Town01/Town10HD are flat; Town03/04/05 have real elevation.
6. **A raylib build for the Linux demo box.** Only the macOS build is vendored.

## 4. Working agreement

Implementation runs in phases, one milestone at a time:

1. I implement the milestone.
2. I tell you **exactly what changed and why**, in plain terms.
3. An independent agent writes the tests for it (§5) and I run **every suite up to and
   including that milestone** — not just the new tests. `make test` must be green.
4. I report the actual result, including anything that failed or that I skipped.
5. **You check it.** Nothing moves to the next milestone until you have.

No milestone is reported as done on the strength of "it compiles" or "it looks right".

---

## 5. Testing

Per the brief: **tests are written by an independent agent, and nothing counts as done
until they pass.** Working arrangement:

1. For each milestone I write the implementation and state the behavioural contract.
2. A separate agent writes tests **from that contract and the code**, not from my claims
   about what the code does, and runs them.
3. Failures come back to me as findings. The milestone stays open until `make test` is
   green.

Suites: `make test-c` (C sim), `make test-unit` (Python), `make test` (everything).
CI already runs all of these on PRs into `3.0`.

**No test may require the wheel to be attached.** Hardware access is confined to the input
and FFB backends; everything above them is tested through the synthetic source and the M0
recorded traces. That is a design requirement on M1 and M6, not an afterthought.

---

## 6. Risks

| Risk | Impact | Handling |
|---|---|---|
| **macOS force feedback** (F13, D10) | The "feel it" requirement fails on the dev machine | Implemented properly via HID++ feature `0x8123` (M6), transcribed from kernel source. Gated in M0 so we find out immediately. If it is blocked, axes/pedals/rendering/sim still work on macOS and the Linux path is untouched. |
| **Sub-stepping all agents costs more than measured** (D4) | Frames drop once ~150 cars are on screen | The 46x headroom measured at 48-64 cars gives a lot of room, but it is measured, not extrapolated: re-measure at the real agent count, and fall back to `SUBSTEP_MODE_HUMAN_ONLY` plus dead reckoning if it bites. |
| **Human car out of distribution** (F9) | Partners behave strangely around you, and it looks like a policy bug | Clamp `a_long` and speed to the bands every other car obeys (M3). Keep the bicycle integration shared with the jerk path (M2). |
| **Reward coefficients outside `REWARD_BOUNDS`** (F3) | Levels appear to do nothing; conditioning silently saturates | The setter rejects out-of-range bounds; a test pins every preset inside the window. |
| **Restoring raylib drags into the training build** (F5) | Undoes #593 | Demo-only Makefile target; `setup.py` untouched; CI unchanged. |
| **Checkpoint config drift** (F11) | Policy reads a mis-shaped observation and drives badly, with no error | Build the env from the checkpoint's own `config.yaml` via `cosim/arch.py` (D7); assert the observation size matches what the loaded weights expect. |

---

## 7. Resolved and open

**Resolved.**

- **Checkpoint:** now shipped in-repo at `weights/k_scaled_0050_1000/` — `final_model.pt` + `config.yaml`,
  exactly the layout `cosim/arch.checkpoint_config_path` expects. Trained at `dt = 0.3`,
  `dynamics_model: jerk`, `action_type: continuous`, `reward_conditioning: true`,
  `reward_randomization: true`, `base_max_speed_mps: 20`, `conditioning_speed_scale: 2.0`
  (so C_vel spans 10–40 m/s), `simulation_mode: gigaflow`, `control_mode: control_vehicles`,
  `max_agents_per_env: 150`, `num_goals: 3`, 20 partner / 70 lane / 50 boundary obs slots.
  All of this is adopted automatically; none of it is hand-copied.
- **Maps — all eight verified by loading them, not by inspecting bytes.** A probe built
  against the real `drive.h` loader (`tests/drive/include/drive_fixture.h` + the
  checkpoint's env config) loaded each town, spawned 150 agents, and ran 30 steps at
  `dt = 0.3` under jerk dynamics:

  | town | roads | drivable | edges | zones | lights | stops | lane graph | spawned | obs | extent |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Town01 | 269 | 164 | 59 | 19 | 36 | 0 | 164 | 150 | finite | 403 m |
  | Town02 | 188 | 124 | 26 | 12 | 24 | 0 | 124 | 150 | finite | 206 m |
  | Town03 | 1093 | 443 | 423 | 26 | 38 | 15 | 443 | 150 | finite | 549 m |
  | Town04 | 1580 | 694 | 530 | 36 | 43 | 9 | 694 | 150 | finite | 940 m |
  | Town05 | 1347 | 580 | 442 | 27 | 54 | 17 | 580 | 150 | finite | 496 m |
  | Town06 | 1164 | 467 | 429 | 27 | 19 | 0 | 467 | 150 | finite | 1049 m |
  | Town07 | 630 | 339 | 188 | 28 | 17 | 19 | 339 | 150 | finite | 377 m |
  | Town10HD | 385 | 168 | 138 | 7 | 15 | 3 | 168 | 150 | finite | 233 m |

  All eight: speed zones present, traffic lights present, lane graph built, 150 agents
  spawn, observations and rewards finite after 30 steps. **Town01, Town02 and Town06 carry
  no stop signs** — that is true of those CARLA towns (01/02 are light-controlled grids,
  06 is highway), not a defect; the stop-sign reward and metric simply never fire there.
  Town07 and Town10HD also carry yield signs, which stay unobserved because
  `yield_signs_enabled` is false. The older `carla_data/map_*.bin` set lacks `SPDZONE1` and
  `TLPHASE1` entirely and must not be used.

- **Checkpoint/env layout — verified against the weights.** Every encoder input width in
  `final_model.pt` matches what 3.0's constants produce under this config, exactly:

  | encoder | checkpoint expects | 3.0 produces |
  |---|---|---|
  | ego | 11 | `EGO_FEATURES` = 11 |
  | lane | 10 | `LANE_FEATURES` 9 + speed-limit 1 |
  | boundary | 6 | `BOUNDARY_FEATURES` = 6 |
  | partner | 11 | `PARTNER_FEATURES` 9 + rel-velocity 2 |
  | traffic control | 14 | (7 − 2) + 4 types + 5 states |
  | context | 27 | 18 reward coefs + 3 goals × 3 |
  | backbone in | 1152 | 128+256+256+256+128+128 |
  | actor head out | 12 | 4 long × 3 lat jerk actions |

  Nothing drifted between this checkpoint and 3.0's observation layout.

- **`fused_slot_encoder` must be turned off for the demo.** The checkpoint config sets
  `policy.fused_slot_encoder: true`, and `DriveBackbone.__init__` raises
  `ImportError("fused_slot_encoder=true requires triton")` when triton is missing — which
  it is here, and triton ships no macOS wheels. It is a *kernel* choice (same bf16 math,
  fewer activation bytes), not a weight-shape change, so disabling it is numerically
  equivalent and the weights load unchanged. M4 overrides it to `false`.
- **Renderer scope:** lift `aditya/trajectory_ablation`'s camera *parameters* only (F6),
  not its offline rasteriser.

- **M0 hardware probe — already run, on real hardware.** `hidapi 0.15.0` installed.
  The G920 exposes three HID interfaces: `usage_page 0x0001 usage 0x0004` (Generic Desktop
  joystick: axes + buttons) and two Logitech vendor interfaces on `usage_page 0xff43`
  (usages `0x0602`, `0x0604`) — the HID++ channel the `0x8123` force-feedback feature lives
  behind.

  Captured 6572 reports in 34.6 s (~190 Hz), 10 bytes each:

  | byte | meaning | observed |
  |---|---|---|
  | 0 | report id | `0x01` |
  | 1 | hat (low nibble) + buttons | only `0x0f`/`0x1f` seen — **one button bit** |
  | 2–3 | (buttons?) | constant `0x00` — nothing seen |
  | 4–5 | **steering, 16-bit LE** | 1281..65535, centre 32755 — near-full range |
  | 6 | **throttle** | 0..255, inverted (255 = released) |
  | 7 | **brake** | 202..255 only — shallow, see below |
  | 8 | **clutch** | 0..255, inverted |
  | 9 | constant | `0x05` |

  Pedal identity was established by *temporal correlation*, not guesswork: pressed in the
  instructed order throttle → brake → clutch, the three bytes first left 255 at frames 5082,
  5206 and 5231 respectively.

- **macOS concurrent HID access — solved.** hidapi opens devices in exclusive mode on
  macOS by default, so holding the joystick interface made both vendor interfaces fail to
  open (and vice versa) — which would have made reading axes and writing FFB mutually
  exclusive. The Python `hidapi` package does not expose the toggle, but the symbol
  `hid_darwin_set_open_exclusive` **is** exported from its bundled library and is reachable
  via `ctypes.CDLL(hid.__file__)`. With it set to 0, all three interfaces open and stay
  open simultaneously. Verified. No Homebrew dependency.

- **macOS Input Monitoring — needed, and here is the proof.** Opening the wheel works from
  a foreground process but **failed from a detached background process** (`OSError: open
  failed`), because macOS attributes TCC permission to the responsible process. The demo
  must therefore run with an explicit Input Monitoring grant rather than relying on
  inherited attribution.

**Open — needs your call.**

- **Which map set.** `/Users/aditya/carla_maps` holds the same eight towns, but it is an
  **older build**: every file lacks the `SPDZONE1` and `TLPHASE1` sections, and loading
  them through the real loader shows the cost against the repo's tracked `carla/` set:

  | town | speed zones | stop signs | road edges |
  |---|---|---|---|
  | | `carla_maps` → tracked | `carla_maps` → tracked | `carla_maps` → tracked |
  | Town01 | 0 → **19** | 0 → 0 | 60 → **59** |
  | Town03 | 0 → **26** | 0 → **15** | 127 → **423** |
  | Town04 | 0 → **36** | 0 → **9** | 221 → **530** |
  | Town05 | 0 → **27** | 5 → **17** | 155 → **442** |
  | Town07 | 0 → **28** | 19 → 19 | 69 → **188** |

  Three consequences for a checkpoint trained on `..._zones_stop_...` maps: the per-lane
  speed-limit observation collapses to the constant `UNKNOWN_LANE_SPEED_LIMIT_MPS` (15 m/s)
  everywhere instead of varying; `speed_limit_random_prob` has nothing to act on; and with
  roughly a third of the road edges, the offroad boundary and the boundary observation are
  far sparser than the policy expects. **Recommendation: use the tracked
  `pufferlib/resources/drive/binaries/carla/` set.** Both load and run fine — this is a
  fidelity call, not a blocker.



- **Training map set.** The checkpoint's `config.yaml` records
  `map_dir: .../binaries/carla_128_affine_zones_stop_ndshoulders_v19`, `num_maps: 128`.
  That is a registered, public dataset: `data_utils/datasets.yaml` lists
  `carla_128_affine_zones_stop` at `s3://pufferdrive-bins/carla/0.3.2/128_affine_zones_stop/`,
  ~106 MB, 128 bins, described as *"CARLA Towns 01-07/10HD as PufferDrive bins with 16
  affine augmentations each (flips, shears, scales), per-lane speed zones, stop signs and
  the 0.1 polyline tolerance (123Drive mapforge at v0.3.2-19)"*.

  So the training set is **the same eight towns we already have, times 16 affine
  augmentations each** — not a different world. Driving the eight un-augmented towns is
  well inside the training distribution; the augmentations exist to stop the policy
  memorising map geometry, and the demo wants the real towns anyway.

  Two small deltas: the registered name lacks the checkpoint's `_ndshoulders` suffix (a
  later variant, probably non-drivable shoulders) and its `_v19` tag. Fetch is one command,
  no credentials (the entry is `public: true`, so `fetch_data.py` passes
  `--no-sign-request`):

      python data_utils/fetch_data.py carla_128_affine_zones_stop

  **Decision needed:** ship on the eight tracked towns (simplest, and what a demo wants —
  recognisable maps, no extra download), or pull the 128-bin set for closer parity. My
  recommendation is the eight towns. Note `aws` CLI is not installed on this machine, so
  fetching needs that first.

**Decided by me — say if you disagree.**

- **Repo-level issues found while building, NOT fixed (adjacent code, your call):**
  (a) `drive_test_env_config` leaves `max_speed_mps` and `conditioning_accel_scale` at 0, so with
  `reward_randomization = 0` every agent gets a speed cap of 0 and an acceleration clamp of
  `clip(a, -5, 0)` — a physically impossible env handed to tests about real dynamics. Two existing
  tests already patch around it by hand. (b) `pufferlib/config/puffer_drive.yaml` sets
  `reset_accel_on_stop: true` while `drive.py:84` defaults the Python constructor to `False`, so a
  `Drive(...)` built in Python gets different physics from one built from the YAML. The demo is
  immune to (b) either way.
- **The human car ignores infraction behaviour.** The checkpoint sets
  `collision_behavior: stop`, which freezes a car on contact. That is right for traffic and
  wrong for a game — you would hit a kerb once and be stuck. The human car gets its own
  setting, defaulting to `ignore`, with every infraction still shown live on the HUD.
  Partners keep `stop`, so a car you hit does stop, which is what you would expect.
- **No early episode termination.** `termination_mode` with
  `inactive_agent_threshold: 0.4` resets the scenario once enough cars are stopped. In a
  demo that means the world can vanish while you are driving. Disabled; the episode runs
  the full `scenario_length` (2500 steps = 12.5 min at dt 0.3) and reset is manual.
