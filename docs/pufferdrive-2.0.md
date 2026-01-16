# PufferDrive 2.0: A fast and friendly driving simulator for training and evaluating RL agents

**Daphne Cornelisse¹·*, Spencer Cheng²·*, Pragnay Mandavilli¹, Julian Hunt¹, Kevin Joseph¹, Waël Doulazmi³, Eugene Vinitsky¹**

¹Emerge Lab at NYU | ²[Puffer.ai](https://puffer.ai/) | ³Valeo | *Shared first authorship

*December 12, 2025*

---

We introduce **PufferDrive 2.0**, a fast and easy-to-use driving simulator for reinforcement learning. It supports training at up to **300,000 steps per second** on a single GPU, enabling agents to reach strong performance in just a few hours. Evaluation and visualization run directly in the browser.

This post outlines the design goals, highlights the main features, and shows what works out of the box. We conclude with a brief roadmap.

---

## Introduction and history

Deep reinforcement learning algorithms, such as [PPO](https://arxiv.org/abs/1707.06347), are highly effective in the billion-sample regime. A consistent finding across domains is that, given sufficient environmental signal and enough data, any precisely specified objective can, in principle, be optimized.

This shifts the primary bottleneck to simulation. The faster we can generate high-quality experience, the more reliably we can apply RL to hard real-world problems, such as autonomous navigation in dynamic, unstructured environments.[^1]

Over the past few years, several simulators have demonstrated that large-scale self-play can work for driving. Below, we summarize this progression and explain how it led to PufferDrive 2.0.

[^1]: A useful parallel comes from the early days of computing. In the 1970s and 1980s, advances in semiconductor manufacturing and microprocessor design—such as Intel’s 8080 and 80286 chips—dramatically reduced computation costs and increased speed. This made iterative software development accessible and enabled entirely new ecosystems of applications, ultimately giving rise to the personal computer. Multi-agent RL faces a similar bottleneck today: progress is limited by the cost and speed of experience collection. Fast, affordable simulation with integrated RL algorithms may play a similar catalytic role, enabling solutions that were previously out of reach.

## Early results with self-play RL in autonomous driving

[**Nocturne**](https://arxiv.org/abs/2206.09889) was the first paper to show that self-play RL could work for driving at scale. Using maps from the [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/), PPO agents achieved around an 80% goal-reaching rate without any human data.

The main limitation was speed. Nocturne ran at roughly 2,000 steps per second, leading to multi-day training times and a complex setup process.

The results were promising, but it was clear that scale was a major constraint.

## Scaling up

Subsequent work showed what becomes possible when scale is no longer the bottleneck.

* [**Gigaflow**](https://arxiv.org/abs/2501.00678) demonstrated that large-scale self-play alone can produce robust, naturalistic driving. Using a highly batched simulator, it trained on the equivalent of **decades of driving experience per hour** and achieved state-of-the-art performance across multiple autonomous driving benchmarks—without using any human data.
* [**GPUDrive**](https://arxiv.org/abs/2408.01584), built on [Madrona](https://madrona-engine.github.io/), showed that [similar behavior could be learned](https://arxiv.org/abs/2502.14706) in about one day on a single consumer GPU, using a simple reward function and a standard PPO implementation.

These empirical results support the hypothesis that robust autonomous driving policies can be trained in the billion-sample regime _without any human data_.


![Sanity map gallery placeholder](images/sim-comparison.png)
**Figure 1:** *Progression of RL-based driving simulators. Left: end-to-end training throughput on an NVIDIA RTX 4080, counting only transitions collected by learning policy agents (excluding padding agents). Right: wall-clock time (log scale) required to reach an 80% goal-reaching rate. This metric captures both simulation speed and algorithmic efficiency.*

## From GPUDrive to PufferDrive

While GPUDrive delivered impressive raw simulation speed, end-to-end training throughput of around 50K steps per second remained a limiting factor. This was particularly true on large maps such as [CARLA](https://carla.org/). Memory layout and batching overheads, rather than simulation fidelity, became the dominant constraints.

Faster end-to-end training is critical because it enables tighter debugging loops, broader experimentation, and faster scientific and engineering progress. This led directly to the development of **PufferDrive**.

We partnered with Spencer Cheng from [Puffer.ai](https://puffer.ai/) to rebuild the system around the principles of [**PufferLib**](https://arxiv.org/abs/2406.12905). Spencer reimplemented **GPUDrive**. The result was **PufferDrive 1.0**, reaching approximately 200,000 steps per second on a single GPU and scaling linearly across multiple GPUs. Training agents to solve 10,000 maps from the Waymo datset took roughly 24 hours with GPUDrive. [With PufferDrive, the same results could now be reproduced in about 2 hours](https://x.com/spenccheng/status/1959665036483350994).

## PufferDrive 2.0

PufferDrive 2.0 builds on this foundation and [TODO]:

* Built-in evaluations, including a standard benchmark
* Support for multiple real-world datasets (WOMD, Carla)
* Speed improvement (200K -> 300K)
* Extended browser-based visualization and analysis tools

To our knowledge, PufferDrive 2.0 is among the fastest open-source driving simulators available today, while remaining accessible to new users.

## Highlights
TODO

## Roadmap
TODO


## Citation

If you use PufferDrive in your research, please cite:
```bibtex
@software{pufferdrive2024github,
  author = {Daphne Cornelisse* and Spencer Cheng* and Pragnay Mandavilli and Julian Hunt and Kevin Joseph and Waël Doulazmi and Eugene Vinitsky},
  title = {{PufferDrive}: A Fast and Friendly Driving Simulator for Training and Evaluating {RL} Agents},
  url = {https://github.com/Emerge-Lab/PufferDrive},
  version = {2.0.0},
  year = {2025},
}
```
*\*Equal contribution*
