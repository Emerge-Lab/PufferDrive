# Wheel demo — setting up on a new machine (Linux)

Everything needed to go from a bare Linux box to the steering-wheel demo running. Written for
Ubuntu 22.04/24.04; adjust package names for other distributions.

The companion documents are [`wheel_demo_plan.md`](wheel_demo_plan.md) (why the thing is built the
way it is) and [`wheel_demo_handoff.md`](wheel_demo_handoff.md) (what state the work is in).

---

## 0. What travels with the repo, and what does not

**In git, nothing to do:**

- All 8 CARLA towns (`pufferlib/resources/drive/binaries/carla/opendrive__Town*.bin`).
- The recorded G920 traces used as test fixtures (`tests/fixtures/g920/`). These cost real time at
  the wheel — they never need recording again.
- The wheel calibration (`wheel_calibration.json`, `wheel_buttons.json` at the repo root). These
  were untracked until now, which would have silently fallen back to defaults with an unreachable
  brake floor. See §6.
- Car models (`pufferlib/resources/drive/*.glb`).
- **The trained policy**, at `weights/k_scaled_0050_1000/` — the frozen checkpoint that drives all
  the traffic, plus the `config.yaml` the env is built from. `run_demo.py` defaults to it, so the
  demo runs with no flags at all. Detail in §4a.

**Not in git, you must bring or recreate:**

- `.claude/settings.local.json` if you want the same permission rules (it is gitignored).
- The vendored raylib is **macOS only** (`extern/raylib-5.5_macos`). Linux builds its own (§3).

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y build-essential git cmake pkg-config \
                    libhidapi-dev libhidapi-hidraw0 \
                    libgl1-mesa-dev libx11-dev xorg-dev libasound2-dev \
                    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

- `libhidapi-*` — the wheel is read over hidraw through the `hidapi` Python package.
- The X11/GL/ALSA set are raylib's build dependencies.

## 2. Python environment

The project uses **Python 3.13.7** managed by [uv](https://docs.astral.sh/uv/). Matching the minor
version matters: the renderer is compiled against the interpreter's ABI (`EXT_SUFFIX`).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # if uv is not installed
cd /path/to/PufferDrive
uv venv --python 3.13.7 .venv
source .venv/bin/activate
```

Then the dependencies. **`setup.py` does not declare everything the demo needs** — `hidapi` and
`pytest` were installed ad hoc on the original machine and are not in `install_requires`:

```bash
pip install -e .              # numpy, gymnasium==0.29.1, pyyaml, torch, hydra-core, omegaconf, ...
pip install hidapi pytest     # NOT declared in setup.py — required for the wheel and the tests
```

`hidapi` provides the module imported as `hid`. Installed versions on the reference machine:
`hidapi 0.15.0`, `pytest 9.2.0`, `torch 2.8.0`, `numpy 1.26.4`, `gymnasium 0.29.1`.

> `torch` on the old machine is a CPU build. If the Linux box has a GPU and you want CUDA, install
> the matching wheel from the PyTorch index. The demo runs the policy on CPU by default
> (`--device cpu`) and that is fast enough: policy inference measures ~23 ms against a 300 ms budget.

Note `pufferlib` is **not** pip-installed as a package — extensions are built in place and imports
resolve from the working directory. **Always run from the repo root.**

## 3. raylib 5.5

The render Makefile's Linux branch links `-lraylib` and includes `/usr/local/include`. Distribution
packages are often too old (Ubuntu 24.04 ships raylib 4.5), and `render.h` is written against 5.5,
so build from source — it installs to `/usr/local` which is exactly where the Makefile looks:

```bash
git clone --depth 1 --branch 5.5 https://github.com/raysan5/raylib.git /tmp/raylib
cmake -S /tmp/raylib -B /tmp/raylib/build \
      -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/raylib/build -j"$(nproc)"
sudo cmake --install /tmp/raylib/build
sudo ldconfig
```

## 4. Build the two extensions

Two separate shared objects, deliberately: the training extension never links raylib (see #593).

```bash
cd /path/to/PufferDrive
source .venv/bin/activate

python setup.py build_ext --inplace --force     # the sim: pufferlib/ocean/drive/binding*.so
make -C pufferlib/ocean/drive/render            # the renderer: drive_render*.so
```

Both must finish with **zero warnings from our own code** (raylib's own build may warn).

## 4a. The policy checkpoint

Every car except yours is driven by a frozen PufferDrive policy. Nothing about the demo env is
hand-configured: the observation layout, `dt`, goal count and conditioning ranges are all adopted
from the checkpoint's own `config.yaml`. This is deliberate — an env whose layout disagrees with the
weights does not crash, it feeds the policy nonsense and the traffic merely drives badly, which is
the worst kind of bug to chase.

### It ships with the repo — nothing to copy

```
weights/k_scaled_0050_1000/
├── final_model.pt     29 MB   the frozen policy
└── config.yaml         7 KB   the env is built from this
```

`run_demo.py` defaults `--checkpoint` to that path, so a fresh clone runs with no flags. This follows
the convention already in the repo — `weights/k_scaled_0036_1000/`, `weights/mimolette/` and
`weights/06_02_checkpoint/` are all committed the same way.

`trainer_state.pt` (92 MB) is deliberately **not** shipped: it is optimiser state for resuming
training and the demo never loads it. Copy it from the original run directory only if you intend to
continue training from this checkpoint.

### Where `config.yaml` has to sit

`arch.checkpoint_config_path()` looks in exactly two places and raises if neither exists:

1. next to the checkpoint — `<run_dir>/config.yaml` beside `final_model.pt`, which is the layout here;
2. one level above — for the `<run_dir>/models/model_*.pt` layout.

```
FileNotFoundError: no config.yaml next to or one level above <path>
```

If you point `--checkpoint` at a bare `.pt` with no config beside it, that is the error you get. It
is deliberate: building a shadow env from `Drive` defaults would be silently wrong.

The config must have an `env` section, or loading aborts with
`"... has no 'env' section; it is not a PufferDrive run config"`. It also needs `policy` and
`policy_name` (`Drive` for this checkpoint).

### What gets adopted, and what the demo overrides

From the checkpoint's `env` block, every key `Drive.__init__` accepts is adopted as-is. For the
reference checkpoint that includes:

```
dynamics_model: jerk        action_type: continuous     dt: 0.3
num_goals: 3                reward_randomization: true  conditioning_speed_scale: 2.0
```

On top, the demo forces a set of structural overrides (`DEMO_ENV_OVERRIDES` in
`pufferlib/wheel_demo/policy_host.py`): termination and goal-termination off so the world never
resets out from under you, co-sim eval semantics off, pose noise zeroed because visible jitter reads
as a rendering bug, `vehicle_size_profile: 1` because training spawns 0.8–7 m boxes and a demo wants
cars, and reward randomisation on so each car has its own personality for the level presets to shift.

One policy override matters for portability: `fused_slot_encoder` is forced **False**
(`DEMO_POLICY_OVERRIDES`), because triton ships no macOS wheels. It is a kernel choice with identical
bf16 math and the same weight shapes, so the same checkpoint loads and nothing about the output
changes. On Linux triton is available, but the override is unconditional — leave it alone unless you
have a reason, and if you change it, re-check the policy golden.

### The layout check

`FrozenTraffic._assert_layout_matches` compares the first encoder layer of each branch against what
the env emits, and raises rather than letting a mismatch through:

```
checkpoint and env observation layouts disagree -- partner_features: checkpoint expects N, env emits M
```

For this checkpoint and the demo env the widths are:

| branch | env features |
| --- | --- |
| ego | 11 |
| lane | 10 |
| boundary | 6 |
| partner | 11 |

If you see that error after a `drive.h` change, the observation layout moved and the checkpoint is
stale — do not "fix" it by editing the assertion.

> **The one failure mode that is not caught:** a `config.yaml` from a *different* run, sitting next
> to these weights. The layout check only covers the four encoder widths, so a mismatched config can
> load cleanly and simply drive badly. Copy the config that shipped with the checkpoint.

### Other constraints

- At least **2 agents** (`--agents`, default 64): you plus traffic. Fewer raises.
- Runs on CPU by default (`--device cpu`) and that is fast enough — policy inference measures ~23 ms
  against a 300 ms budget at 64 agents.
- The map is passed separately (`--town`); it is not part of the checkpoint.

## 5. Verify without touching the wheel

In order — each step rules out a layer:

```bash
# 1. C simulation tests
make -C tests/drive                 # expect "All test groups passed", Failed: 0

# 2. C renderer maths (headless; raylib calls are redirected to capture stubs)
make -C tests/drive render          # expect 15 PASS

# 3. Python suite
python -m pytest -q tests/unit_tests
#    expect ~509 passed, 22 skipped, and ONE pre-existing failure:
#    test_action_selection.py::test_drive_mean_action_is_exact_in_physical_units
#    (fused_slot_encoder needs triton; unrelated to the wheel demo)

# 4. The demo itself, no hardware, scripted input, with a window
python scripts/wheel_demo/run_demo.py \
  --town Town01 --input synthetic --steps 40 --render --camera chase
```

Step 4 opens a window, drives a scripted lap and exits 0. If it segfaults immediately with no
output, the renderer is the problem, not the sim.

## 6. Wheel calibration

The calibration is 13 numbers and is **committed to the repo**, so if you are using the same
physical G920 there is nothing to do — `run_demo.py` picks up `wheel_calibration.json` from the
working directory by default.

```json
{"steering_raw_min": 0, "steering_raw_center": 32767, "steering_raw_max": 65535,
 "steering_deadzone_norm": 0.0, "pedal_deadzone_norm": 0.02,
 "throttle_raw_rest": 255, "throttle_raw_floor": 0,
 "brake_raw_rest":    255, "brake_raw_floor":    53,
 "clutch_raw_rest":   255, "clutch_raw_floor":   0}
```

`wheel_buttons.json` is `{"shift_up_bit": 4, "shift_down_bit": 5}` — right paddle bit 4, left bit 5.

Two things worth knowing about these numbers:

- **The brake bottoms at 53, not 0.** It is a progressive rubber stop, so a hard press only covers
  79% of the travel the other pedals do. This is why the brake felt weak, and why
  `brake_response_gamma` (default 0.6) now bends the curve so light presses bite harder.
- Steering and the other two pedals cover their full electrical range, so a lock-to-lock sweep was
  genuinely captured.

**For a different physical wheel**, recalibrate:

```bash
python scripts/wheel_demo/run_demo.py --calibrate
```

It walks you through a steering sweep and each pedal, then validates the result and prints warnings
for the mistakes that have actually happened before (rest captured mid-press, a sweep that never
reached the stops, a brake far shorter than the other pedals).

---

## 7. Linux porting — what is genuinely untested

The demo was built and verified on macOS (Apple Silicon). The code has Linux branches, but the
following were **never exercised on Linux**. In rough order of risk:

### 7.1 HID interface selection — the most likely thing to break

`find_g920_interfaces()` in `pufferlib/wheel_demo/input/g920.py` picks interfaces by
`usage_page`/`usage`:

```python
if entry["usage_page"] == 0x0001 and entry["usage"] == 0x0004:   # axes
elif entry["usage_page"] == 0xFF43:                               # Logitech vendor (force feedback)
```

hidapi fills these reliably on macOS and Windows. On Linux's hidraw backend they are parsed out of
the report descriptor and **may come back as 0**, in which case `axes` stays `None` and the demo
aborts with "no G920 axis interface found".

Diagnose first — do not guess:

```bash
python -c "
import hid
for e in hid.enumerate(0x046D, 0xC262):
    print(f\"path={e['path']} iface={e['interface_number']} \"
          f\"usage_page=0x{e['usage_page']:04x} usage=0x{e['usage']:04x}\")
"
```

If `usage_page` is `0x0000` for every entry, select by `interface_number` instead. Keep the macOS
path intact and branch on `sys.platform`; do not replace it.

### 7.2 Kernel driver ownership

The G920 is claimed by the in-tree `hid-logitech-hidpp` driver, which implements force feedback
itself through evdev. Our force feedback talks HID++ feature `0x8123` directly over hidraw. The two
may conflict — the kernel driver can swallow the vendor reports, or reset effects underneath us.

If force feedback is silent on Linux while the axes read fine, that is the first suspect. Options,
in increasing order of intrusiveness: check whether writes to the vendor interface return an error;
try the evdev force-feedback API instead (`FF_CONSTANT` via `/dev/input/eventN`, which is the
idiomatic Linux route and what the `ForceFeedbackOutput` interface was designed to allow as a second
backend); or unbind the kernel driver.

**There is no Linux force-feedback backend written.** `ffb.py` has exactly two real outputs:
`G920ForceFeedback` (HID++ over hidapi) and `RecordingForceOutput` (used by tests). The docstring
naming "evdev/SDL (Linux)" describes the intended shape of the interface, not something implemented.
Budget for writing one.

### 7.3 Permissions

macOS needs Input Monitoring granted to the foreground terminal. Linux needs a udev rule instead,
or you will be running as root:

```bash
sudo tee /etc/udev/rules.d/99-logitech-g920.rules >/dev/null <<'EOF'
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c262", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="input", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c262", MODE="0660", TAG+="uaccess"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Replug the wheel afterwards. `TAG+="uaccess"` grants the locally logged-in user access, which is
preferable to a group.

The macOS-only exclusivity workaround (`set_darwin_non_exclusive`, which lets the axis and vendor
interfaces be open at once) already no-ops off Darwin — `g920.py:103` guards on
`sys.platform != "darwin"`. Linux hidraw does not open exclusively, so nothing is needed there.

### 7.4 Wheel mode

The G920 has an Xbox mode and a PC mode. The report layout this code parses (byte 0 report id `0x01`,
bytes 4–5 steering u16 LE, bytes 6/7/8 throttle/brake/clutch, all pedals inverted) was verified on
hardware in PC mode. If the bytes look like nonsense, check the mode switch before changing the
parser.

### 7.5 Rendering

The Makefile's Linux branch is written but never compiled. It links
`-lraylib -lGL -lm -lpthread -ldl -lrt -lX11`. On Wayland you may need to force X11
(`SDL_VIDEODRIVER=x11`, or run under XWayland) — raylib's default Linux backend is GLFW/X11.

---

## 8. Running it

```bash
source .venv/bin/activate
python scripts/wheel_demo/run_demo.py --town ask --render --ffb --camera chase --steps 4000
```

`--town ask` lists all 8 towns and prompts. `--steps 4000` is about 20 minutes at dt=0.3.

| flag | meaning |
| --- | --- |
| `--checkpoint PATH` | a different policy; defaults to `weights/k_scaled_0050_1000/final_model.pt` |
| `--input synthetic` | scripted input, no hardware — how CI exercises the loop |
| `--calibrate` | run calibration before driving |
| `--camera chase\|driver\|agent` | starting view |
| `--gearbox manual\|automatic` | manual uses the clutch and paddles |
| `--grip realistic\|kinematic` | realistic enforces a tyre-grip limit |
| `--level calm\|normal\|aggressive` | how the surrounding traffic behaves |
| `--ffb`, `--ffb-gain 0..1` | force feedback and its overall strength (default 0.55, **untuned**) |
| `--night` | night lighting with headlights and brake lights |

In the window: `1` chase, `2` driver, `3` top-down, `N` night, `L` cycle traffic level,
`TAB` take over the next car, `R` restart a stalled engine.
