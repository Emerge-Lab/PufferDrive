"""Per-rank seed decorrelation for multi-GPU / multi-node training.

Under DDP every rank holds identical policy weights; if ranks also shared
identical torch and env seeds they would collect (near-)duplicate experience
and multi-node training would degenerate to single-node. train() derives
per-rank seeds via `pufferl.derive_rank_seeds(vec_seed, train_seed,
world_size, global_rank)`; these tests pin its contract:

  - every GPU (rank) gets a distinct torch seed and env seed
  - ranks on different nodes differ too (seeding keys off the global RANK;
    LOCAL_RANK values repeat on every node)
  - single-process runs keep torch.manual_seed(train_seed) exactly
  - the derivation is a pure function: same config -> same seeds every run
  - sweeping train.seed varies the env scenario stream, not just torch RNG
  - vec.seed=None (explicitly unseeded envs) passes through as None

No env/C-sim/GPU needed.
"""

import pytest

from pufferlib.pufferl import derive_rank_seeds

VEC_SEED = 42
TRAIN_SEED = 42


def _seeds_for_all_ranks(world_size, vec_seed=VEC_SEED, train_seed=TRAIN_SEED):
    return [derive_rank_seeds(vec_seed, train_seed, world_size, rank) for rank in range(world_size)]


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_multi_gpu_ranks_get_distinct_seeds(world_size):
    seeds = _seeds_for_all_ranks(world_size)
    torch_seeds = [torch_seed for torch_seed, _ in seeds]
    env_seeds = [env_seed for _, env_seed in seeds]
    assert len(set(torch_seeds)) == world_size, f"torch seed collision: {torch_seeds}"
    assert len(set(env_seeds)) == world_size, f"env seed collision: {env_seeds}"


def test_multi_node_ranks_with_same_local_rank_differ():
    # 2 nodes x 8 GPUs: LOCAL_RANK is 0-7 on both nodes, so any seeding keyed
    # off the local rank would replay node 0's seeds on node 1. Global ranks
    # r and r + 8 share a local rank and must still get distinct seeds.
    gpus_per_node = 8
    world_size = 2 * gpus_per_node
    seeds = _seeds_for_all_ranks(world_size)
    for local_rank in range(gpus_per_node):
        node0_torch, node0_env = seeds[local_rank]
        node1_torch, node1_env = seeds[local_rank + gpus_per_node]
        assert node0_torch != node1_torch, f"local rank {local_rank}: torch seed repeats across nodes"
        assert node0_env != node1_env, f"local rank {local_rank}: env seed repeats across nodes"


def test_single_process_torch_seed_unchanged():
    # Backward compatibility: a non-distributed run must seed torch with the
    # plain train seed, exactly as before the per-rank derivation existed.
    torch_seed, _ = derive_rank_seeds(VEC_SEED, 123, world_size=1, global_rank=0)
    assert torch_seed == 123


def test_derivation_is_deterministic():
    # SeedSequence is a deterministic hash, not an entropy source: the same
    # config must map to the same seeds on every run of every process, or
    # runs would not be reproducible. The pinned value guards the mixing
    # scheme itself; update it only on a deliberate scheme change (which
    # breaks scenario-stream comparability with older runs).
    assert derive_rank_seeds(42, 42, 8, 0) == (336, 1921063561)
    assert derive_rank_seeds(42, 42, 8, 3) == derive_rank_seeds(42, 42, 8, 3)


def test_train_seed_sweep_varies_env_seed():
    # Sweeping train.seed alone must vary the env scenario stream; before the
    # per-rank derivation, sweeps only changed network init and sampling.
    env_seeds = {derive_rank_seeds(VEC_SEED, train_seed, 1, 0)[1] for train_seed in range(5)}
    assert len(env_seeds) == 5


def test_unseeded_envs_pass_through():
    # vec.seed=None means "do not seed the envs"; the derivation must not
    # manufacture a seed for them.
    _, env_seed = derive_rank_seeds(None, TRAIN_SEED, 8, 3)
    assert env_seed is None


def test_no_env_seed_collisions_across_sweep_by_rank_grid():
    # A realistic experiment grid (5 sweep seeds x 64 ranks) must produce
    # all-distinct env seeds; any collision means two runs/ranks replay the
    # same scenario sequence.
    grid = [derive_rank_seeds(VEC_SEED, train_seed, 64, rank)[1] for train_seed in range(5) for rank in range(64)]
    assert len(set(grid)) == len(grid)
