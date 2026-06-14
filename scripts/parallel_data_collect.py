import os
import sys
from pathlib import Path
import numpy as np

# Run from the PufferDrive project root so relative resource/config paths work.
working_dir = Path.cwd()
while not (working_dir / "pufferlib").exists():
    if working_dir == working_dir.parent:
        raise FileNotFoundError("Could not find the PufferDrive project root containing 'pufferlib'")
    working_dir = working_dir.parent
os.chdir(working_dir)
sys.path.append(str(working_dir / "pufferlib" / "ocean" / "drive"))

from pufferlib.ocean.drive.drive import Drive, RenderView

def main():
    MAP_DIR = "resources/drive/binaries/training"

    if not Path(MAP_DIR).exists():
        print(f"Warning: {MAP_DIR} not found.")
        return

    # PufferDrive samples vectorized envs from the first NUM_MAPS map files.
    # num_agents is the target total number of controlled rows in the batch.
    NUM_ENVS = 10
    NUM_MAPS = 100
    CONTROLLED_AGENTS_PER_ENV = 1
    NUM_AGENTS = NUM_ENVS * CONTROLLED_AGENTS_PER_ENV
    EPISODE_LEN = 91

    print("Initializing PufferDrive Environment...")
    env = Drive(
        map_dir=MAP_DIR,
        num_maps=NUM_MAPS,
        num_agents=NUM_AGENTS,
        # max_controlled_agents is enforced for control_mixed_play in the C env.
        control_mode="control_mixed_play",
        init_mode="create_all_valid",
        goal_behavior=2,
        action_type="continuous",
        episode_length=EPISODE_LEN,
        render_mode=RenderView.FULL_SIM_STATE,
        max_controlled_agents=CONTROLLED_AGENTS_PER_ENV,
    )

    obs, _ = env.reset()
    print("env obs space:", env.observation_space)
    print("env action space:", env.action_space)
    print("single env obs shape:", env.single_observation_space)
    print("single env action shape:", env.single_action_space)

    num_controlled_agents = obs.shape[0]
    if env.num_envs != NUM_ENVS or num_controlled_agents != NUM_AGENTS:
        env.close()
        raise RuntimeError(
            f"Expected {NUM_ENVS} envs/{NUM_AGENTS} controlled agents, "
            f"got {env.num_envs} envs/{num_controlled_agents} controlled agents."
        )

    print(f"Environment initialized. Batch size (controlled agents): {num_controlled_agents}")
    print(f"Vectorized env count: {env.num_envs}")
    print(f"Sampled map ids: {env.map_ids}")
    print(f"Scenario ids: {env.scenario_ids}")

    # 2. Initialize lists to store our data
    collected_obs = []
    collected_actions = []
    collected_rewards = []
    collected_terminals = []
    collected_truncations = []

    print("Starting Parallel Native Render Rollout...")
    for t in range(EPISODE_LEN):
        # 3. Store the observation.
        # CRITICAL: Use .copy() because PufferLib writes to the same C-memory buffer in-place.
        collected_obs.append(obs.copy())

        # Generate actions for all parallel agents simultaneously
        rl_actions = np.random.uniform(-1.0, 1.0, size=(num_controlled_agents, 2)).astype(np.float32)

        # Store the actions
        collected_actions.append(rl_actions.copy())

        # Step the environment
        obs, rewards, terminals, truncations, infos = env.step(rl_actions)
        collected_rewards.append(rewards.copy())
        collected_terminals.append(terminals.copy())
        collected_truncations.append(truncations.copy())

        # Render only the first environment (env_id=0) out of the 10
        # env.render(view_mode=RenderView.FULL_SIM_STATE, draw_traces=True, env_id=0)

        if t % 10 == 0:
            print(f"Processed step {t:02d}/{EPISODE_LEN}...")

        if terminals.all() or truncations.all():
            print(f"All environments terminated at step {t}.")
            break

    env.close()

    # 4. Stack the lists into final numpy arrays for easy saving/manipulation
    final_obs_array = np.stack(collected_obs)          # Shape: (T, 10, obs_dim)
    final_actions_array = np.stack(collected_actions)  # Shape: (T, 10, 2)
    final_rewards_array = np.stack(collected_rewards)  # Shape: (T, 10)
    final_terminals_array = np.stack(collected_terminals)
    final_truncations_array = np.stack(collected_truncations)

    print("\n=== Data Collection Complete ===")
    print(f"Collected Obs Shape: {final_obs_array.shape} -> (Time, Agents, Features)")
    print(f"Collected Actions Shape: {final_actions_array.shape} -> (Time, Agents, Actions)")
    print(f"Collected Rewards Shape: {final_rewards_array.shape} -> (Time, Agents)")
    print(f"Collected Terminals Shape: {final_terminals_array.shape} -> (Time, Agents)")
    print(f"Collected Truncations Shape: {final_truncations_array.shape} -> (Time, Agents)")

if __name__ == "__main__":
    main()
