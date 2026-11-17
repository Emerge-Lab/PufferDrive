"""rl.derive_rank_seeds: distinct, reproducible torch and env seeds per DDP rank, keyed off the global RANK."""

import pytest

from opengigaflow.rl import derive_rank_seeds


VEC_SEED = 42
TRAIN_SEED = 42


def _seeds_for_all_ranks(world_size):
    return [derive_rank_seeds(VEC_SEED, TRAIN_SEED, world_size, rank) for rank in range(world_size)]


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_multi_gpu_ranks_get_distinct_seeds(world_size):
    seeds = _seeds_for_all_ranks(world_size)
    torch_seeds = [torch_seed for torch_seed, _ in seeds]
    env_seeds = [env_seed for _, env_seed in seeds]
    assert len(set(torch_seeds)) == world_size, f"torch seed collision: {torch_seeds}"
    assert len(set(env_seeds)) == world_size, f"env seed collision: {env_seeds}"


def test_multi_node_ranks_with_same_local_rank_differ():
    # 2 nodes x 8 GPUs: global ranks r and r + 8 share a LOCAL_RANK
    gpus_per_node = 8
    world_size = 2 * gpus_per_node
    seeds = _seeds_for_all_ranks(world_size)
    for local_rank in range(gpus_per_node):
        node0_torch, node0_env = seeds[local_rank]
        node1_torch, node1_env = seeds[local_rank + gpus_per_node]
        assert node0_torch != node1_torch, f"local rank {local_rank}: torch seed repeats across nodes"
        assert node0_env != node1_env, f"local rank {local_rank}: env seed repeats across nodes"


def test_single_process_torch_seed_unchanged():
    torch_seed, _ = derive_rank_seeds(VEC_SEED, 123, world_size=1, global_rank=0)
    assert torch_seed == 123


def test_derivation_is_deterministic():
    # a deliberate change of the mixing scheme breaks scenario-stream comparability with older runs
    assert derive_rank_seeds(42, 42, 8, 0) == (336, 1921063561)


def test_train_seed_sweep_varies_env_seed():
    env_seeds = {derive_rank_seeds(VEC_SEED, train_seed, 1, 0)[1] for train_seed in range(5)}
    assert len(env_seeds) == 5


def test_unseeded_envs_pass_through():
    _, env_seed = derive_rank_seeds(None, TRAIN_SEED, 8, 3)
    assert env_seed is None


def test_no_env_seed_collisions_across_sweep_by_rank_grid():
    grid = [derive_rank_seeds(VEC_SEED, train_seed, 64, rank)[1] for train_seed in range(5) for rank in range(64)]
    assert len(set(grid)) == len(grid)
