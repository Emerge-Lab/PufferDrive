# PufferDrive


## Installation

Clone the repo
```bash
https://github.com/Emerge-Lab/PufferDrive.git
```

Make a venv (`uv venv`), activate the venv
```
source .venv/bin/activate
```

Inside the venv, install the dependencies
```
uv pip install -e .
```

Compile the C code
```
python setup.py build_ext --inplace --force
```

To test your setup, you can run
```
puffer train puffer_drive
```
See also the [puffer docs](https://puffer.ai/docs.html).


## Quick start

Start a training run
```
puffer train puffer_drive
```

## Dataset

<details>
<summary>Downloading and using data</summary>

### Data preparation

To train with PufferDrive, you need to convert JSON files to map binaries. Run the following command with the path to your data folder:

```bash
python pufferlib/ocean/drive/drive.py
```

### Downloading Waymo Data

You can download the WOMD data from Hugging Face in two versions:

- **Mini Dataset**: [GPUDrive_mini](https://huggingface.co/datasets/EMERGE-lab/GPUDrive_mini) contains 1,000 training files and 300 test/validation files
- **Full Dataset**: [GPUDrive](https://huggingface.co/datasets/EMERGE-lab/GPUDrive) contains 100,000 unique scenes

**Note**: Replace 'GPUDrive_mini' with 'GPUDrive' in your download commands if you want to use the full dataset.

### Additional Data Sources

For more training data compatible with PufferDrive, see [ScenarioMax](https://github.com/valeoai/ScenarioMax). The GPUDrive data format is fully compatible with PufferDrive.
</details>


## Visualizer

<details>
<summary>Dependencies and usage</summary>

## Headless server setup

Run the Raylib visualizer on a headless server and export as .mp4. This will rollout the pre-trained policy in the env.

### Install dependencies

```bash
sudo apt update
sudo apt install ffmpeg xvfb
```

For HPC (There are no root privileges), so install into the conda environment
```bash
conda install -c conda-forge xorg-x11-server-xvfb-cos6-x86_64
conda install -c conda-forge ffmpeg
```

- `ffmpeg`: Video processing and conversion
- `xvfb`: Virtual display for headless environments

### Build and run

1. Build the application:
```bash
bash scripts/build_ocean.sh visualize local
```

2. Run with virtual display:
```bash
xvfb-run -s "-screen 0 1280x720x24" ./visualize
```

The `-s` flag sets up a virtual screen at 1280x720 resolution with 24-bit color depth.

---

> To force a rebuild, you can delete the cached compiled executable binary using `rm ./visualize`.

---

</details>


## Benchmarks

### Distributional realism

We provide a PufferDrive implementation of the [Waymo Open Sim Agents Challenge (WOSAC)](https://waymo.com/open/challenges/2025/sim-agents/) for fast, easy evaluation of how well your trained agent matches distributional properties of human behavior. See details [here](https://github.com/Emerge-Lab/PufferDrive/main/pufferlib/ocean/benchmark).

WOSAC evaluation with random policy:
```bash
puffer eval puffer_drive --eval.wosac-realism-eval True
```

WOSAC evaluation with your checkpoint (must be .pt file):
```bash
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path <your-trained-policy>.pt
```

### Human-compatibility

You may be interested in how compatible your agent is with human partners. For this purpose, we support an eval where your policy only controls the self-driving car (SDC). The rest of the agents in the scene are stepped using the logs. While it is not a perfect eval since the human partners here are static, it will still give you a sense of how closely aligned your agent's behavior is to how people drive. You can run it like this:
```bash
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path <your-trained-policy>.pt
```
