# PufferDrive

<div class="hero">
  <img src="images/pufferdrive.gif" alt="PufferDrive logo">
  <div>
    <p>PufferDrive is a high-throughput autonomous driving simulator built on <a href="https://puffer.ai">PufferLib</a>. Train and evaluate multi-agent driving policies with fast vectorized stepping, streamlined data conversion, and ready-made benchmarks.</p>
    <div class="cta">
      <a class="primary" href="getting-started/">Start here: install & build</a>
      <a href="#workflow">See the workflow</a>
    </div>
  </div>
</div>

## Highlights

- Multi-agent drive environment that trains agents at 300K steps per second.
- Scripts to convert Waymo Open Motion Dataset JSON into lightweight binaries (CARLA sample map included).
- Benchmarks for distributional realism and human compatibility.
- Raylib-based visualizer for local or headless render/export.

## Quick start

- Follow [Getting Started](getting-started.md) to install, build the C extensions, and run `puffer train puffer_drive`.
- Consult [Simulator](simulator.md) for how actions/observations, rewards, and `.ini` settings map to the underlying C environment and Torch policy.
- Prepare drive map binaries with the steps in [Data](data.md).
- Evaluate a policy with the commands in [Evaluation](evaluation.md) and preview runs with the [Visualizer](visualizer.md).

## Workflow

<div class="workflow">
  <div class="step-card">
    <div class="badge">Step 1</div>
    <h3>Install & Build</h3>
    <p>Set up the environment, install dependencies, and compile the native extensions.</p>
    <a href="getting-started/">Open guide</a>
  </div>
  <div class="step-card">
    <div class="badge">Step 2</div>
    <h3>Prepare Data</h3>
    <p>Download WOMD/GPUDrive datasets from Hugging Face and convert to map binaries.</p>
    <a href="data/">Open guide</a>
  </div>
  <div class="step-card">
    <div class="badge">Step 3</div>
    <h3>Train & Evaluate</h3>
    <p>Train agents and evaluate them with WOSAC and human-replay benchmarks.</p>
    <a href="evaluation/">Open guide</a>
  </div>
</div>

## Repository layout
- `pufferlib/ocean/drive`: Drive environment implementation and map processing utilities.
- `resources/drive/binaries`: Expected location for compiled map binaries (outputs of the data conversion step).
- `scripts/build_ocean.sh`: Helper for building the Raylib visualizer and related binaries.
- `examples`, `tests`, `experiments`: Reference usage, checks, and research scripts that pair with the docs pages.
