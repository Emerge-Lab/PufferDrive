# Wheel demo — setting up on a new machine (Linux)

Everything needed to go from a bare Linux box to the steering-wheel demo running. Written for
Ubuntu 22.04/24.04; adjust package names for other distributions. Verified end to end on Ubuntu
24.04 with an RTX 4080 on 2026-10-30, without sudo for the working user.

The companion documents are [`wheel_demo_plan.md`](wheel_demo_plan.md) (why the thing is built the
way it is) and [`wheel_demo_handoff.md`](wheel_demo_handoff.md) (what state the work is in).

---

## 0. What travels with the repo, and what does not

**In git, nothing to do:**

- All 8 CARLA towns as sim maps (`pufferlib/resources/drive/binaries/carla/opendrive__Town*.bin`).
- The recorded G920 traces used as test fixtures (`tests/fixtures/g920/`).
- The wheel calibration and the full button map (`wheel_calibration.json`, `wheel_buttons.json` at
  the repo root). The button map was captured on Linux and covers both paddles, A/B/X/Y, View, Menu,
  LSB, RSB and the D-pad. See §6.
- **The trained policy**, at `weights/k_scaled_0050_1000/`. `run_demo.py` defaults to it. Detail in §4a.

**Not in git, you must bring:**

- **A CARLA 0.9.16 server** — the demo renders through CARLA (§3). The base package has Towns 01-05
  and 10HD; Town06 and Town07 come from the separate AdditionalMaps package.
- `.claude/settings.local.json` if you want the same permission rules (it is gitignored).

## 1. System packages

Nothing beyond a C toolchain is required: the pip `hidapi` wheel bundles its own HID library and the
renderer is CARLA. Root is only needed for the udev rule in §7.3.

```bash
sudo apt install -y build-essential git
```

## 2. Python environment

**The demo runs on Python 3.12.** CARLA's Python client ships wheels for 3.10-3.12 only, and the sim,
the policy and the CARLA client share one process. Python 3.13 still works for everything except
rendering (the test suite passes on both).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # if uv is not installed
cd /path/to/PufferDrive
uv venv --python 3.12 .venv312
source .venv312/bin/activate
uv pip install "torch==2.8.0" "numpy==1.26.4" "setuptools<81" wheel "pyarrow<22"
uv pip install --no-build-isolation -e .
uv pip install hidapi pytest "carla==0.9.16" pygame    # NOT declared in setup.py
```

- `pyarrow<22`: unpinned, pip pulls pyarrow 26, which requires numpy 2; numpy is pinned to 1.26.4.
- On a machine with a CUDA toolkit matching torch's (12.8 for torch 2.8.0), `-e .` also builds the
  CUDA extension. The demo runs the traffic policy on the GPU by default when one is present
  (`--device`), which removes a once-per-policy-step render hitch.
- `run_demo.py` puts the repo root on `sys.path` itself (it needs the top-level `data_utils`), so it
  runs from any working directory.

## 3. The CARLA server

The demo renders through a CARLA 0.9.16 server. Only the server binary is needed; nothing is built.

```bash
export CARLA_ROOT=~/carla-0.9.16          # the directory containing CarlaUE4.sh
scripts/wheel_demo/start_carla.sh         # keep it running in its own terminal
```

- **Use your own copy of CARLA.** A shared install you cannot write to cannot take AdditionalMaps.
  Copying it (19 GB) is enough: `cp -a /path/to/shared/carla ~/carla-0.9.16`.
- **Town06 and Town07** need AdditionalMaps (14.8 GB):
  ```bash
  curl -fL -o ~/carla-0.9.16/Import/AdditionalMaps_0.9.16.tar.gz \
    https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.16.tar.gz
  cd ~/carla-0.9.16 && ./ImportAssets.sh && rm Import/AdditionalMaps_0.9.16.tar.gz
  ```
- **On a CPU with integrated graphics, CARLA silently renders on the iGPU** (Vulkan device 0): 132 ms
  per frame instead of 11 ms. `start_carla.sh` points `VK_ICD_FILENAMES` at the NVIDIA driver to
  prevent this. Check with `nvidia-smi`: `CarlaUE4-Linux-Shipping` must be listed.
- Only one client can drive the CARLA world. Do not run tests or scripts that connect to CARLA while
  someone is driving.

## 4. Build the sim extension

```bash
python setup.py build_ext --inplace --force     # pufferlib/ocean/drive/binding*.so
```

It must finish with **zero warnings**. The extension is built per interpreter (`EXT_SUFFIX`), so a
3.12 and a 3.13 venv in the same checkout each need their own build. Never rebuild while a demo is
running: the build overwrites the `.so` the running process has loaded.

The old raylib renderer (`pufferlib/ocean/drive/render/`, `make -C pufferlib/ocean/drive/render`) is
no longer used by the demo.

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
- Runs on the GPU when one is present (`--device cuda`, ~3 ms per policy step) and on CPU otherwise
  (~12-23 ms); both fit the 300 ms policy step. On CPU the forward pass causes a visible render hitch
  once per policy step.
- The map is passed separately (`--town`); it is not part of the checkpoint.

## 5. Verify without touching the wheel

In order — each step rules out a layer:

```bash
# 1. C simulation tests
make -C tests/drive                 # expect "All test groups passed" from every binary, 0 warnings

# 2. Python suite (both venvs, if you keep both)
python -m pytest -q tests/unit_tests

# 3. The demo itself: no hardware, scripted input, rendered through CARLA (server running, §3)
python scripts/wheel_demo/run_demo.py --town Town03 --input synthetic --steps 40 --render --camera chase
```

Step 3 opens a window, drives a scripted lap in real time and exits 0. Tests that need a live CARLA
server are skipped unless `WHEEL_DEMO_CARLA_TESTS=1`.

## 6. Wheel calibration and buttons

The calibration (`wheel_calibration.json`) and the button map (`wheel_buttons.json`) are committed.
For the same physical G920 there is nothing to do.

`wheel_buttons.json` maps every control the demo uses (bits within the report's 20-bit button mask,
hat values for the D-pad): right paddle 4, left paddle 5, A 0, B 1, X 2, Y 3, Menu 6, View 7, RSB 8,
LSB 9; D-pad up/right/down/left = hat 0/2/4/6.

To re-map, or to re-capture only the controls that failed (the probe refuses a control unless exactly
one bit changes):

```bash
python scripts/wheel_demo/probe_g920.py buttons --out wheel_buttons.json
python scripts/wheel_demo/probe_g920.py buttons --out wheel_buttons.json --only a_bit,shift_down_bit
```

The G920 only sends a report when something changes: press Enter first, then press and hold. A
button already held when recording starts is never seen.

For a different physical wheel, recalibrate with `run_demo.py --calibrate`. The brake bottoms at raw
53 on the reference unit, not 0; `brake_response_gamma` (0.4) makes light presses bite and
`throttle_response_gamma` (1.6) tames part throttle. Both are `WheelCalibration` defaults.

---

## 7. Linux — how each porting risk was resolved

### 7.1 HID backend

On Linux the pip `hidapi` wheel's `import hid` is the **libusb** backend: it reports `usage_page 0`,
gives USB bus paths, and opening a device detaches the kernel driver. `import hidraw` from the same
wheel is the hidraw backend and reports the descriptor's usages correctly (all three collections —
joystick `0x0001/0x0004` and vendor `0xff43/0x0602`, `0xff43/0x0604` — share one node,
`/dev/hidraw*`). `g920.hid_module()` picks `hidraw` on Linux and `hid` elsewhere; every opener uses it.
On Linux the axis report arrives at up to ~600 Hz, only when something changes.

### 7.2 Force feedback

Force feedback goes through the kernel's own `hid-logitech-hidpp` driver: `EvdevForceFeedback` uploads
one `FF_CONSTANT` effect to the wheel's `/dev/input/eventN` and rewrites its level. It needs no root
(the joystick node is `uaccess` by default) and does not fight the kernel driver. It refuses to start
unless the kernel reports a 900-degree range. Sign verified by hand. `G920ForceFeedback` (raw HID++)
remains the macOS path; `platform_force_output()` chooses.

### 7.3 Permissions

Wheel input reads `/dev/hidraw*`, which is root-only by default. Install a udev rule **numbered below
73** — systemd applies `uaccess` in `73-seat-late.rules`, so a `99-` rule silently grants nothing:

```bash
sudo tee /etc/udev/rules.d/70-logitech-g920.rules >/dev/null <<'RULES'
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c262", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="input", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c262", MODE="0660", TAG+="uaccess"
RULES
sudo udevadm control --reload-rules && sudo udevadm trigger
```

`getfacl /dev/hidrawN` must then show `user:<you>:rw-`. Without sudo, ask an admin to run it from an
interactive `su - <admin>` shell (`su -c 'sudo ...'` cannot prompt for the password).

### 7.4 Wheel mode

Verified on Linux in PC mode: the report layout is byte-identical to macOS (report id `0x01`, steering
u16 LE in bytes 4-5, throttle/brake/clutch in 6/7/8, all inverted).

### 7.5 Rendering

CARLA, see §3. At 1280x720 on an RTX 4080: chase and top-down hold 60 fps, the driver view (with
mirrors, one extra camera) about 50 fps; the sim always runs in real time and drops frames rather than
slowing down.

---

## 8. Running it

**From the desktop:** install the launcher once, then double-click "PufferDrive Wheel Demo" (also in
the app menu). On GNOME, right-click the desktop icon once and choose "Allow Launching".

```bash
.venv312/bin/python scripts/wheel_demo/install_launcher.py
```

The launcher shows every setting below with the defaults pre-selected (Town03), starts the CARLA
server when it is not running (or restarts it from `CARLA_ROOT`, default `~/carla-0.9.16`, when the
running server lacks the chosen town), opens the demo, and comes back when it exits. Logs go to
`~/.cache/pufferdrive_wheel_demo/`.

**From a terminal:**

```bash
source .venv312/bin/activate
python scripts/wheel_demo/run_demo.py --town ask --render --ffb --camera chase --steps 4000
```

| flag | meaning |
| --- | --- |
| `--checkpoint PATH` | a different policy; defaults to `weights/k_scaled_0050_1000/final_model.pt` |
| `--input synthetic` | scripted input, no hardware |
| `--camera chase\|driver\|agent` | starting view |
| `--gearbox automatic\|manual` | manual uses the clutch and paddles |
| `--grip realistic\|kinematic` | realistic understeers past the tyre limit |
| `--level calm\|normal\|aggressive` | how the surrounding traffic behaves |
| `--assist off\|display\|blend\|nudge\|full`, `--assist-blend` | steering assist (M9) |
| `--ffb`, `--ffb-gain 0..1` | force feedback and its strength (default 0.7) |
| `--night` | night, with street lights and headlights |
| `--carla-host`, `--carla-port` | where the CARLA server is |

On the wheel (the keyboard keys `1 2 3 N L M TAB R Esc` still work too):

| control | does |
| --- | --- |
| A | cycle camera: chase, driver, top-down |
| B | night / day |
| X | traffic level: calm, normal, aggressive |
| Y | take over the next car |
| View | restart a stalled engine |
| Menu | steering assist: off, display, blend, nudge, full |
| D-pad up / down | force feedback stronger / weaker |
| Left paddle, stopped | reverse (automatic) / downshift (manual) |
| Right paddle, stopped | drive (automatic) / upshift (manual) |
