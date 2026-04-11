# Trajviz: Vulkan Offline Renderer

`trajviz` is a Vulkan-backed offline renderer that turns saved Drive
trajectories into MP4 videos at high throughput. It runs **headlessly**
on a single GPU (no X server required) and supports **batched
multi-episode rendering** so you can amortize the per-frame overhead
across many episodes in one pass.

It is independent of the existing raylib visualizer (`scripts/build_ocean.sh
visualize`) — they share no code and can coexist. Trajviz is the path you
want when you need to render many checkpoint videos quickly, on a cluster
node, or from a Python script.

Source: `pufferlib/ocean/drive/trajviz/`.

## When to use trajviz

| You want to… | Use |
|---|---|
| Replay one scenario interactively, debug a policy live | `visualize` (raylib) |
| Render N saved-checkpoint videos in one Python call | **trajviz** |
| Render hundreds of trajectories on a headless cluster node | **trajviz** |
| Drive your render from a notebook / training-loop callback | **trajviz** |

Trajviz outputs the same two views the live raylib visualizer does:

- **Top-down** (`RenderView.FULL_SIM_STATE`): orthographic full-map view
- **BEV** (`RenderView.BEV_AGENT_OBS`): agent-centric ~100 m × 178 m
  window, ego at the center facing up

## Prerequisites

Apt (Ubuntu/Debian):

```bash
sudo apt install -y libvulkan-dev glslang-tools vulkan-tools spirv-tools ffmpeg
```

Each package is needed for:

- `libvulkan-dev` — Vulkan headers used at compile time
- `glslang-tools` — `glslangValidator`, the GLSL → SPIR-V compiler that
  trajviz invokes when compiling its shaders
- `vulkan-tools` — `vulkaninfo` for diagnostics (optional)
- `spirv-tools` — SPIR-V utilities (optional)
- `ffmpeg` — runtime; trajviz pipes raw RGBA frames to ffmpeg for h264 encoding

You also need a Vulkan-capable GPU and ICD. On NVIDIA, the proprietary
driver provides this automatically (`/usr/share/vulkan/icd.d/nvidia_icd.json`).
Verify with:

```bash
vulkaninfo --summary
```

You should see your GPU listed with `deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`.

## Build

Trajviz is an **opt-in** CPython extension built via `setup.py`. Pass
`TRAJVIZ=1` to enable it:

```bash
TRAJVIZ=1 python setup.py build_ext --inplace
```

This compiles the shaders (via `glslangValidator`), embeds them as SPIR-V
blobs in a generated `shaders.c`, and builds
`pufferlib.ocean.drive.trajviz._native` into the source tree.

Without `TRAJVIZ=1`, the trajviz extension is **not** built and the rest
of pufferlib (including the drive sim) builds normally — so users who
don't need trajviz aren't forced to install Vulkan.

## Usage

### Python API

```python
from pufferlib.ocean.drive.trajviz import Renderer

with Renderer(width=1280, height=720) as r:
    r.render_episode(
        road_xy=road_xy,            # (V, 2) float32, mean-centered
        road_offsets=road_offsets,  # (P+1,) uint32 CSR
        road_types=road_types,      # (P,)   uint32 — TVZ_ROAD_* type ids
        traj_xyh=traj,              # (T, A, 3) float32  (x, y, heading)
        agent_lengths=lengths,      # (A,)   int32  valid step counts
        ego_idx=-1,                 # -1 = first agent with length >= 2
        fps=30,
        out_topdown="td.mp4",
        out_bev="bev.mp4",
    )
```

### Batched (multi-episode) API

```python
with Renderer(width=1280, height=720) as r:
    r.render_batch([
        dict(road_xy=..., road_offsets=..., road_types=...,
             traj_xyh=..., agent_lengths=..., ego_idx=-1,
             out_topdown="ep0_td.mp4", out_bev="ep0_bev.mp4"),
        dict(road_xy=..., road_offsets=..., road_types=...,
             traj_xyh=..., agent_lengths=..., ego_idx=-1,
             out_topdown="ep1_td.mp4", out_bev="ep1_bev.mp4"),
        # ... up to 16 episodes per batch
    ], fps=30)
```

The Renderer is reusable across batches. Pay the Vulkan startup cost
(~50 ms) and the BatchRenderer atlas allocation (~20 ms) once for an
entire run of episodes.

### From a saved trajectories_*.npz

```python
from pufferlib.ocean.drive.trajviz import render_npz

render_npz(
    "data/runs/.../trajectories_000010.npz",
    maps_dir="pufferlib/resources/drive/binaries/training",
    out_dir="videos/",
)
```

### CLI

```bash
python -m pufferlib.ocean.drive.trajviz \
    data/runs/foo/trajectories_*.npz \
    --maps-dir pufferlib/resources/drive/binaries/training \
    --out videos/
```

Multiple input files or directories are supported. The Vulkan context is
created once and reused across all inputs.

### Random-rollout smoke test

A small tool spins up a Drive sim, runs a 90-step random-action episode,
and renders both views. Useful for verifying that trajviz works end-to-end
without depending on saved trajectories:

```bash
python -m pufferlib.ocean.drive.trajviz.tools.random_rollout \
    --map pufferlib/resources/drive/binaries/map_001.bin \
    --out-dir /tmp
```

Outputs `/tmp/random_topdown.mp4` and `/tmp/random_bev.mp4`. Defaults to
2 controllable agents (matches the typical WOSAC `tracks_to_predict`
count); use `--num-agents N` to override.

## Performance tuning

On an RTX 4080 (16-core CPU), the current pipeline reaches **~3.7
episodes per second** at `batch_size ≥ 4` for 90-frame 1280×720 episodes
with both views. Per-episode breakdown:

- Pure GPU + readback: **~30 ms / episode** (the floor — what trajviz
  achieves with `TRAJVIZ_NO_WRITE=1`)
- + ffmpeg encoding (libx264 `-preset veryfast`): **~270 ms / episode**

The encoder is the dominant cost beyond the GPU work; everything below
is squeezed.

### Bumping kernel pipe limits

Trajviz pipes 3.6 MB raw RGBA frames per view to ffmpeg via Unix pipes.
Default Linux pipe buffers (64 KB) force many round-trips per frame; the
trajviz `ffmpeg_pipe_open` automatically calls `fcntl(F_SETPIPE_SZ, ...)`
to bump them, but the per-process maximum is `/proc/sys/fs/pipe-max-size`
(default 1 MB on most kernels). Raise it for better throughput:

```bash
sudo sysctl fs.pipe-max-size=16777216
```

There is also a per-user *total* page budget,
`/proc/sys/fs/pipe-user-pages-soft` (default 64 MB shared across all
your pipes). For batches >= 8 with 16 large pipes, raise this too:

```bash
sudo sysctl fs.pipe-user-pages-soft=262144   # 1 GB total per user
```

Both settings revert on reboot. Persist via `/etc/sysctl.d/99-trajviz.conf`
if you want them permanent.

### Why HOST_CACHED matters (NVIDIA)

The single biggest win in trajviz's throughput came from requesting
`HOST_CACHED` for the readback buffers (see `vk_batch_renderer.c`).
NVIDIA's default `HOST_VISIBLE | HOST_COHERENT` memory type is
write-combined PCIe BAR — fast for the GPU to write to, but
**~250 MB/s for the CPU to read** because every read is uncached over
PCIe. With `HOST_CACHED`, reads hit RAM at >5 GB/s. This is a 6-7×
speedup on its own.

If your device doesn't expose `HOST_CACHED` host-visible memory, trajviz
falls back to plain `HOST_COHERENT` and prints no warning, so the only
visible symptom is slower wall-clock per frame.

### Choosing a batch size

| batch_size | latency / batch | per-ep | ep/s |
|---|---|---|---|
| 1 | ~345 ms | 345 ms | 2.9 |
| 2 | ~596 ms | 298 ms | 3.4 |
| 4 | ~1.1 s | 274 ms | 3.7 |
| 8 | ~2.1 s | 267 ms | 3.7 |

The curve plateaus at `batch_size = 4` — past that, the CPU encoders
(N parallel libx264 instances) saturate ~16 cores. Going to
`batch_size = 16` doesn't help and consumes more pipe memory. Pick the
smallest size that gives you the throughput you need.

### Debugging knobs

The C side honors a few env vars for benchmarking:

- `TRAJVIZ_NO_FFMPEG=1` — replace the ffmpeg subprocess with `cat > /dev/null`.
  Skips encoding cost; useful for measuring "render + readback + pipe write" alone.
- `TRAJVIZ_NO_WRITE=1` — skip the `write()` to the pipe entirely. The
  output mp4 will be empty/invalid; useful for measuring the pure
  Vulkan + readback path.
- `TRAJVIZ_FFMPEG=/path/to/ffmpeg` — override the ffmpeg binary used.

## Architecture

```
              .npz / numpy arrays
                      │
              ┌───────▼───────┐
              │  __init__.py  │   Renderer wrapper, npz loader,
              └───────┬───────┘   numpy padding, batching shim
                      │
                      ▼
              ┌───────────────┐
              │   _native.c   │   CPython extension boundary,
              └───────┬───────┘   numpy → raw pointers, GIL release
                      │
                      ▼
              ┌───────────────┐
              │   trajviz.c   │   public API: render_episode,
              └───┬─────────┬─┘   render_episodes_batch
                  │         │
                  ▼         ▼
        ┌──────────┐  ┌──────────────────┐
        │vk_renderer│  │vk_batch_renderer │
        │ (1 ep)    │  │   (N eps tiled)  │
        └─────┬────┘  └────────┬─────────┘
              │                │
              ├────────────────┘
              │
              ▼
   ┌──────────────────┐    ┌──────────────────┐
   │ vk_pipeline.c    │    │ ffmpeg_pipe.c    │
   │ vk_context.c     │    │ + writer thread  │
   │ (Vulkan setup)   │    │ (per pipe)       │
   └──────────────────┘    └──────────────────┘
              │                       │
              ▼                       ▼
        Vulkan 1.3 driver        ffmpeg subprocess
```

Key design points:

- **Tiled atlas for batching**: the batched renderer allocates one large
  color attachment image per view, sized `tile_w × (batch_size * tile_h)`.
  Tiles are stacked **vertically** so each tile's bytes are row-contiguous
  in the host readback buffer — one `write()` per tile per frame, no row
  stitching.
- **Threaded writers**: each ffmpeg pipe gets its own background writer
  thread. The renderer's per-frame "submit all → wait all" loop pays
  `max(single fwrite)` per frame instead of `sum(fwrites)`, which is the
  threading win.
- **Push-constant cameras**: per-frame and per-view MVP matrices are pushed
  via `vkCmdPushConstants`, no descriptor sets. Each view has its own
  camera matrix per slot per frame.
- **LINE_STRIP polylines**: roads are drawn as `VK_PRIMITIVE_TOPOLOGY_LINE_STRIP`
  with one `vkCmdDraw` per polyline (not per segment), so a 268-polyline
  Waymo intersection is 268 draw calls per view, not ~2400.
- **Instanced agent boxes**: the agent vertex shader expands a unit quad
  by per-instance `(x, y, heading, length, width, color)`. One
  `vkCmdDrawIndexed` per slot draws all of that slot's agents.

## Known limitations / future work

- **Uniform `num_steps` in batch**: all episodes in a batch share the
  same length cap. The Python wrapper pads shorter episodes with zeros
  and uses `agent_lengths` to mark valid steps. Episodes with very
  different lengths waste GPU work on the trailing zeros.
- **No NPC / expert-replay agents**: trajviz only renders the
  controlled agents from `get_sim_trajectories`. The other 18 vehicles
  in a typical Waymo scenario (the WOSAC "context" tracks) are not
  shown. Adding them requires a separate Drive API to expose expert
  trajectories.
- **No 3D follow-cam**: the `RenderView.AGENT_PERSP` view from the
  raylib visualizer (3D car meshes from `.glb`) is not implemented.
- **CPU-bound by libx264**: the encoder is the wall once batching is
  amortized. NVENC integration (Vulkan video encode or libnvidia-encode)
  would close the remaining 12% gap to the pure-GPU ceiling but adds a
  CUDA dependency.
- **batch_size cap of 16**: enforced by `TRAJVIZ_BATCH_MAX` in
  `trajviz.c`. The atlas image height grows linearly with batch_size,
  and 16 × 720 = 11520 px is well under Vulkan's 16384 limit. Raising
  it further requires either a 2-D tile grid layout or multiple atlas
  passes.

## Troubleshooting

**ImportError on `from pufferlib.ocean.drive.trajviz import Renderer`** —
the extension wasn't built. Run `TRAJVIZ=1 python setup.py build_ext --inplace`.

**`vulkan/vulkan.h: No such file or directory`** during build —
`libvulkan-dev` not installed. `sudo apt install libvulkan-dev`.

**`glslangValidator: command not found`** during build — `glslang-tools`
not installed. `sudo apt install glslang-tools`.

**`no Vulkan-capable physical device found`** at runtime — your driver
isn't exposing a Vulkan ICD. Check `vulkaninfo --summary`. On a remote
node, ensure the GPU device files (`/dev/nvidia*`) are accessible to
your user.

**`ffmpeg topdown write failed at slot N`** — the ffmpeg subprocess
crashed or was killed. Check the ffmpeg stderr in your terminal output.
A common cause is the output path containing a single quote (we reject
those for shell-quoting safety).

**`pipe size 1048576 B < frame size 3686400 B — fwrites may block`** —
informational warning that the kernel pipe buffer is smaller than one
frame. Bump it via `sudo sysctl fs.pipe-max-size=16777216` (see
performance section above).

## Files

```
pufferlib/ocean/drive/
├── map_io.py                        Map .bin parser (extracted from notebook)
└── trajviz/
    ├── __init__.py                  Python Renderer wrapper, render_npz
    ├── __main__.py                  CLI entry point
    ├── _native.c                    CPython extension shell (numpy unwrap)
    ├── trajviz.{h,c}                Public C API: render_episode, render_episodes_batch
    ├── vk_context.{h,c}             VkInstance, VkDevice, queues, command pool
    ├── vk_pipeline.{h,c}            Graphics pipelines (line + box)
    ├── vk_renderer.{h,c}            Single-episode renderer
    ├── vk_batch_renderer.{h,c}      Batched multi-episode renderer (tiled atlas)
    ├── vk_math.h                    Mat4 helpers (header-only)
    ├── ffmpeg_pipe.{h,c}            Pipe to ffmpeg + writer thread
    ├── shaders.h                    Externs for embedded SPIR-V blobs
    ├── shaders.c                    GENERATED — do not commit
    ├── shaders/
    │   ├── polyline.{vert,frag}     GLSL source for road polylines
    │   ├── agent_box.{vert,frag}    GLSL source for instanced agent quads
    │   └── build_shaders.sh         Compiles GLSL → embedded shaders.c
    ├── tests/
    │   └── test_main.c              Standalone C test harness (no Python)
    └── tools/
        └── random_rollout.py        Random-policy rollout → MP4 smoke test
```
