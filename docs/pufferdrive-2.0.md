# PufferDrive 2.0: A fast and friendly driving simulator for training and evaluating RL agents

**Daphne Cornelisse**¹·$*$, **Spencer Cheng**²·$*$, Pragnay Mandavilli¹, Julian Hunt¹, Kevin Joseph¹, Waël Doulazmi³, Eugene Vinitsky¹

¹ Emerge Lab at NYU | ² [Puffer.ai](https://puffer.ai/) | ³ Valeo | $*$ Shared first contributor

*December 26, 2025*

---

We introduce **PufferDrive 2.0**, a fast, easy-to-use driving simulator for reinforcement learning. Built on [PufferLib](https://puffer.ai/), it supports training at **300,000 steps per second** on a single GPU, allowing agents to reach strong performance in just a few hours. Evaluation and visualization run directly in the browser.

This post highlights the main features and provides some history. We conclude with a brief roadmap.

---

## Highlights

- **Super-fast self-play RL:** Reach ~1.0 score on 10K multi-agent Waymo scenarios in under an hour (single episode, 91 steps). [Earlier results](https://arxiv.org/abs/2502.14706) took 24 hours; you can now get close in **~15 minutes** on a single consumer GPU.
- **Long-horizon driving:** Train agents via self-play RL to reach goals indefinitely on large Carla maps. The demo agents are trained this way. Try it yourself and drive with trained agents in the browser.
- **Built-in evaluation:** Integrated eval support for the [Waymo Open Sim Agent Challenge (WOSAC)](https://emerge-lab.github.io/PufferDrive/wosac/) and a [human compatibility benchmark](https://emerge-lab.github.io/PufferDrive/evaluation/#human-compatibility-benchmark).
- **Easy scenario creation:** Edit or design custom scenarios in minutes, including long-tail and stress-test cases, using the [interactive scenario editor](https://emerge-lab.github.io/PufferDrive/scene-editor/).
- **And more:** See the docs for details.


## Drive together with trained agents

<iframe src="/assets/game.html" title="PufferDrive Demo" width="1280" height="720" style="border: none; display: block; margin: 2rem auto;"></iframe>

<p style="text-align: center; color: #888; margin-top: 1rem;">
  Hold <strong>Left Shift</strong> and use arrow keys or <strong>WASD</strong> to control the vehicle. Hold <strong>space</strong> for first-person view and <strong>ctrl</strong> to see what your agent is seeing :)
</p>


## Introduction and history

Deep reinforcement learning algorithms, such as [PPO](https://arxiv.org/abs/1707.06347), are highly effective in the billion-sample regime. Across domains, a consistent finding is that RL can optimize precisely specified objectives even under sparse rewards, provided there are occasional successes and sufficient scale.

This shifts the primary bottleneck to simulation. The faster we can generate high-quality experience, the more reliably we can apply RL to hard real-world problems, such as autonomous navigation in dynamic, unstructured environments.[^1]

Over the past few years, several simulators have demonstrated that large-scale self-play can be effective for driving. Below, we summarize this progression and explain how it led to PufferDrive 2.0.

[^1]: A useful parallel comes from the early days of computing. In the 1970s and 1980s, advances in semiconductor manufacturing and microprocessor design—such as Intel’s 8080 and 80286 chips—dramatically reduced computation costs and increased speed. This made iterative software development accessible and enabled entirely new ecosystems of applications, ultimately giving rise to the personal computer. Multi-agent RL faces a similar bottleneck today: progress is limited by the cost and speed of experience collection. Fast, affordable simulation with integrated RL algorithms may play a similar catalytic role, enabling solutions that were previously out of reach.

## Early results with self-play RL in autonomous driving

[**Nocturne**](https://arxiv.org/abs/2206.09889) was the first paper to show that self-play RL could work for driving at scale. Using maps from the [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/), PPO agents achieved around an 80% goal-reaching rate without any human data.

The main limitation was speed. Nocturne ran at roughly 2,000 steps per second, leading to multi-day training times and a complex setup process.

The results were promising, but it was clear that scale was a major constraint.

## Scaling up

Subsequent work showed what becomes possible when scale is no longer the bottleneck.

* [**Gigaflow**](https://arxiv.org/abs/2501.00678) demonstrated that large-scale self-play alone can produce robust, naturalistic driving. Using a highly batched simulator, it trained on the equivalent of **decades of driving experience per hour** and achieved state-of-the-art performance across multiple autonomous driving benchmarks—without using any human data.
* [**GPUDrive**](https://arxiv.org/abs/2408.01584), built on [Madrona](https://madrona-engine.github.io/), showed that [similar levels of controllability could be achieved](https://arxiv.org/abs/2502.14706) in about one day on a single consumer GPU, using a simpler reward function and a standard PPO implementation.

These empirical results support the hypothesis that robust autonomous driving policies can be trained in the billion-sample regime _without any human demonstrations_.

![Sanity map gallery placeholder](images/sim-comparison.png)
**Figure 1:** *Progression of RL-based driving simulators. Left: end-to-end training throughput on an NVIDIA RTX 4080, counting only transitions collected by learning policy agents (excluding padding agents). Right: wall-clock time (log scale) required to reach an 80% goal-reaching rate. This metric captures both simulation speed and algorithmic efficiency.*

## From GPUDrive to PufferDrive

While GPUDrive delivered impressive raw simulation speed, end-to-end training throughput of around 50K steps per second remained a limiting factor. This was particularly true on large maps such as [CARLA](https://carla.org/). Memory layout and batching overheads, rather than simulation fidelity, became the dominant constraints.

Faster end-to-end training is critical because it enables tighter debugging loops, broader experimentation, and faster scientific and engineering progress. This led directly to the development of **PufferDrive**.

We partnered with Spencer Cheng from [Puffer.ai](https://puffer.ai/) to rebuild the system around the design of [**PufferLib**](https://arxiv.org/abs/2406.12905). Spencer reimplemented **GPUDrive**. The result was **PufferDrive 1.0**, reaching approximately 200,000 steps per second on a single GPU and scaling linearly across multiple GPUs. Training agents to solve 10,000 maps from the Waymo datset took roughly 24 hours with GPUDrive. [With PufferDrive, the same results could now be reproduced in about 2 hours](https://x.com/spenccheng/status/1959665036483350994).


## Roadmap: What is next?
TODO


## Citation

If you use PufferDrive in your research, please cite:
```bibtex
@software{pufferdrive2025github,
  author = {Daphne Cornelisse* and Spencer Cheng* and Pragnay Mandavilli and Julian Hunt and Kevin Joseph and Waël Doulazmi and Eugene Vinitsky},
  title = {{PufferDrive}: A Fast and Friendly Driving Simulator for Training and Evaluating {RL} Agents},
  url = {https://github.com/Emerge-Lab/PufferDrive},
  version = {2.0.0},
  year = {2025},
}
```
*\*Equal contribution*
