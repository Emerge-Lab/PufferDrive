"""get_global_agent_state(include_static=True) returns every static agent of a replay scene without overrunning its buffers."""

import numpy as np

from opengigaflow.ocean.drive import binding
from opengigaflow.ocean.drive.drive import Drive

MAP_DIR = "opengigaflow/resources/sdc_replay_test"


def test_replay_scene_static_agents_fit_the_returned_arrays():
    env = Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        goal_source="gt",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=20,
        resample_frequency=1_000_000,
        termination_mode=False,
    )
    env.reset(seed=1)
    static_agent_count = binding.vec_get_static_agent_count(env.c_envs)
    assert static_agent_count > 0
    active_state = env.get_global_agent_state()
    full_state = env.get_global_agent_state(include_static=True)
    for key, values in full_state.items():
        assert values.shape == (env.num_agents + static_agent_count,)
        np.testing.assert_array_equal(values[: env.num_agents], active_state[key])
    assert np.all(np.isfinite(full_state["x"])) and np.all(np.isfinite(full_state["y"]))
    env.close()


def test_undersized_output_arrays_are_rejected():
    env = Drive(
        num_agents=8,
        min_agents_per_env=8,
        max_agents_per_env=8,
        num_maps=1,
        map_dir="opengigaflow/resources/carla",
        scenario_length=20,
        resample_frequency=0,
        termination_mode=False,
    )
    env.reset(seed=1)
    short_float = np.zeros(env.num_agents - 1, dtype=np.float32)
    short_int = np.zeros(env.num_agents - 1, dtype=np.int32)
    try:
        binding.vec_get_global_agent_state(
            env.c_envs, short_float, short_float, short_float, short_float, short_int, short_float, short_float, 0
        )
        raised = False
    except ValueError:
        raised = True
    env.close()
    assert raised
