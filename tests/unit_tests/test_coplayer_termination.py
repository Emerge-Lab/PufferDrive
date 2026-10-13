"""termination_mode=1 under the co-player layout (log_sdc_only): only the SDC's own stop or removal ends an episode
early. Co-players stopped by collision_behavior/offroad_behavior="stop" stay masked while the episode goes on.

With inactive_agent_threshold=0.0 any counted stop would end the episode, so a masked co-player row proves
co-players are not counted. With 0.4 a lone SDC among 16 rows would not reach the threshold, so an SDC that is
never seen masked after stopping proves its stop alone ends the episode.
"""

import os

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import COPLAYER_ENV_SETUP

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUPLAN_MAP = os.path.join(
    REPO_ROOT, "pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin"
)
STEP_COUNT = 1000
JERK_ACTION_COUNT = 12


@pytest.mark.parametrize("inactive_agent_threshold", [0.0, 0.4])
def test_only_the_sdc_ends_a_coplayer_episode(inactive_agent_threshold):
    env = Drive(
        map_dir=NUPLAN_MAP,
        num_maps=1,
        simulation_mode="replay",
        goal_source="gt",
        scenario_length=91,
        resample_frequency=0,
        num_agents=64,
        max_agents_per_env=64,
        min_agents_per_env=1,
        obs_slots_lane_n=8,
        obs_slots_boundary_n=8,
        obs_slots_partners_n=4,
        dynamics_model="jerk",
        action_type="discrete",
        termination_mode=1,
        inactive_agent_threshold=inactive_agent_threshold,
        collision_behavior="stop",
        offroad_behavior="stop",
        terminate_on_goal=0,
        seed=0,
        **COPLAYER_ENV_SETUP,
    )
    sdc_rows = np.flatnonzero(env.learner_masks)
    coplayer_rows = np.flatnonzero(~env.learner_masks[: env.agent_offsets[-1]])
    assert len(sdc_rows) > 1 and len(coplayer_rows) >= 15 * len(sdc_rows)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    sdc_stops = masked_sdc_steps = masked_coplayer_steps = 0
    try:
        for step in range(STEP_COUNT):
            _, _, terminals, _, _ = env.step(rng.integers(0, JERK_ACTION_COUNT, size=(env.num_agents, 1)))
            # masks are written as a step starts, so they show the stops of the previous step
            if step > 0:
                masked_sdc_steps += int(np.count_nonzero(~env.masks[sdc_rows]))
                masked_coplayer_steps += int(np.count_nonzero(~env.masks[coplayer_rows]))
            sdc_stops += int(np.count_nonzero(terminals[sdc_rows]))
    finally:
        env.close()
    assert sdc_stops > 0, "random actions never stopped an SDC"
    assert masked_sdc_steps == 0, "a stopped SDC kept its episode running"
    assert masked_coplayer_steps > 0, "a co-player stop ended the episode"
