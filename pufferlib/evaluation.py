import argparse
import itertools
import json
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import pufferlib
import pufferlib.vector
from pufferlib.ocean import env_creator
from pufferlib.ocean.torch import Drive, Recurrent


def evaluate_oracle_policy(
    policy_path,
    co_player_policy_path=None,
    num_maps=10,
    num_rollouts=10,
    num_workers=8,
    oracle_mode=True,
    condition_type="none",
    device="cuda",
    ego_preset="cautious",
):

    num_agents = 64

    ego_presets = {
        "cautious": {"collision": -1.0, "offroad": -0.4, "goal": 0.0, "entropy": 0.001, "discount": 0.95},
        "aggressive": {"collision": 0.0, "offroad": 0.0, "goal": 1.0, "entropy": 0.001, "discount": 0.95},
    }
    ego_config = ego_presets[ego_preset]

    collision_weights = [-1.0, 0.0]
    offroad_weights = [-0.4, 0.0]
    goal_weights = [0.0, 1.0]
    entropy_weights = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    discount_weights = list(np.linspace(0.1, 1.0, 10))  # [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    cautious = [-1.0, -0.4, 0.0]
    aggressive = [0.0, 0.0, 1.0]
    combos = list(itertools.product(
        # collision_weights,
        # offroad_weights,
        # goal_weights,
        entropy_weights,
        discount_weights,
    ))
    combos = [tuple(mode) + combo for mode in (cautious, aggressive) for combo in combos]
    print(combos)

    print(f"Evaluation Configuration:")
    print(f"  Policy: {policy_path}")
    print(f"  Device: {device}")
    print(f"  Num maps: {num_maps}")
    print(f"  Num agents per env: {num_agents}")
    print(f"  Num workers: {num_workers}")
    print(f"  Rollouts per combo (parallel): {num_rollouts}")
    print(f"  Total parallel agents: {num_agents * num_rollouts}")
    print(f"  Total combinations: {len(combos)}")
    print(f"  Oracle mode: {oracle_mode}")

    policy_dir = os.path.dirname(policy_path)
    policy_name = os.path.basename(policy_path).replace(".pt", "")
    out_path = os.path.join(policy_dir, policy_name, "eval_results.jsonl")
    print(f"  Output: {out_path}\n")

    print("Loading policy...")
    make_env = env_creator("puffer_drive")

    temp_env = make_env(num_agents=1, num_maps=1, oracle_mode=oracle_mode, condition_type=condition_type)
    base_policy = Drive(temp_env, input_size=64, hidden_size=256)
    policy = Recurrent(temp_env, base_policy, input_size=256, hidden_size=256).to(device)
    state_dict = torch.load(policy_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()

    co_player_policy = None
    if co_player_policy_path is not None:
        print("Loading co-player policy...")
        co_player_base_policy = Drive(temp_env, input_size=64, hidden_size=256)
        co_player_policy = Recurrent(temp_env, co_player_base_policy, input_size=256, hidden_size=256).to(device)
        co_player_state_dict = torch.load(co_player_policy_path, map_location=device)
        co_player_state_dict = {k.replace("module.", ""): v for k, v in co_player_state_dict.items()}
        co_player_policy.load_state_dict(co_player_state_dict)
        co_player_policy.eval()
        print("Co-player policy loaded successfully")

    temp_env.close()

    print("Policy loaded successfully\n")

    with open(out_path, "w") as f:
        t0 = time.time()

        print("Starting evaluation sweep...\n")

        pbar = tqdm(combos, desc="Evaluating", ncols=120,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for combo_idx, (cw, ow, gw, ew, dw) in enumerate(pbar):
            # Update progress bar description with current combo
            combo_str = f"C:{cw:+.1f} O:{ow:+.1f} G:{gw:+.1f} E:{ew:.4f} D:{dw:.2f}"
            pbar.set_description(f"[{combo_str}]")

            # Run num_rollouts environments in parallel
            population_play = co_player_policy is not None
            env_kwargs = {
                "num_agents": num_agents,
                "num_maps": num_maps,
                "collision_weight_lb": cw,
                "collision_weight_ub": cw,  # lb=ub: all agents same
                "offroad_weight_lb": ow,
                "offroad_weight_ub": ow,
                "goal_weight_lb": gw,
                "goal_weight_ub": gw,
                "entropy_weight_lb": ew,
                "entropy_weight_ub": ew,
                "discount_weight_lb": dw,
                "discount_weight_ub": dw,
                "oracle_mode": oracle_mode,
                "condition_type": condition_type,
                "population_play": population_play,
                "report_interval": 1,  # Report every step for metrics
            }
            if co_player_policy is not None:
                env_kwargs["co_player_policy"] = co_player_policy

            vecenv = pufferlib.vector.make(
                make_env,
                env_kwargs=env_kwargs,
                backend=pufferlib.vector.Multiprocessing,
                num_envs=num_rollouts,
                num_workers=min(num_workers, num_rollouts),
            )

            obs, _ = vecenv.reset()

            driver_env = vecenv.driver_env

            if population_play:
                # In population play, set all ego agents to fixed conditioning
                for ego_id in driver_env.ego_ids:
                    # Find which env this ego belongs to
                    for env_idx in range(num_rollouts):
                        env_start = driver_env.agent_offsets[env_idx]
                        env_end = driver_env.agent_offsets[env_idx + 1]
                        if env_start <= ego_id < env_end:
                            active_idx = ego_id - env_start
                            driver_env.set_agent_weights(env_idx, active_idx, ego_config["collision"],
                                ego_config["offroad"], ego_config["goal"], ego_config["entropy"], ego_config["discount"])
                            break
            # else:
            #     # In single policy mode, set one random agent per env to fixed conditioning
            #     agent_offsets = driver_env.agent_offsets
            #     for env_idx in range(num_rollouts):
            #         env_active_count = agent_offsets[env_idx + 1] - agent_offsets[env_idx]
            #         if env_active_count > 0:
            #             ego_idx = np.random.randint(0, env_active_count)
            #             driver_env.set_agent_weights(env_idx, ego_idx, ego_config["collision"],
            #                 ego_config["offroad"], ego_config["goal"], ego_config["entropy"], ego_config["discount"])

            total_agents = obs.shape[0]
            # print("Total Agents", total_agents)

            state = {
                "lstm_h": torch.zeros(total_agents, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(total_agents, policy.hidden_size, device=device),
            }

            total_reward = np.zeros(total_agents)
            all_infos = []

            with torch.no_grad():
                for t in range(91):
                    obs_t = torch.as_tensor(obs, device=device)
                    logits, _ = policy.forward_eval(obs_t, state)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits)

                    obs, reward, done, trunc, info = vecenv.step(action.cpu().numpy())
                    total_reward += reward

                    # Collect all infos for metrics
                    if info:
                        all_infos.extend(info)

            # Compute per-rollout returns
            rollout_rewards = total_reward.reshape(num_rollouts, num_agents)
            rollout_returns = rollout_rewards.mean(axis=1)

            # Extract performance metrics (always available)
            # Sum up all metrics across all infos
            perf_sum = sum(info.get('perf', 0) for info in all_infos)
            score_sum = sum(info.get('score', 0) for info in all_infos)
            collision_rate_sum = sum(info.get('collision_rate', 0) for info in all_infos)
            offroad_rate_sum = sum(info.get('offroad_rate', 0) for info in all_infos)
            completion_rate_sum = sum(info.get('completion_rate', 0) for info in all_infos)
            dnf_rate_sum = sum(info.get('dnf_rate', 0) for info in all_infos)
            num_infos = len(all_infos) or 1

            # Average across all logged episodes
            perf_avg = perf_sum / num_infos
            score_avg = score_sum / num_infos
            collision_rate_avg = collision_rate_sum / num_infos
            offroad_rate_avg = offroad_rate_sum / num_infos
            completion_rate_avg = completion_rate_sum / num_infos
            dnf_rate_avg = dnf_rate_sum / num_infos

            vecenv.close()

            avg_ret = float(np.mean(rollout_returns))
            std_ret = float(np.std(rollout_returns))
            min_ret = float(np.min(rollout_returns))
            max_ret = float(np.max(rollout_returns))

            # Update progress bar with key metrics
            pbar.set_postfix({
                'Ret': f'{avg_ret:.1f}±{std_ret:.1f}',
                'Perf': f'{perf_avg:.3f}',
                'Coll': f'{collision_rate_avg:.3f}',
                'Comp': f'{completion_rate_avg:.3f}'
            })

            result = {
                "collision_weight": cw,
                "offroad_weight": ow,
                "goal_weight": gw,
                "entropy_weight": ew,
                "discount_weight": dw,
                "num_maps": num_maps,
                "num_agents": num_agents,
                "horizon": 91,
                "rollouts": num_rollouts,
                "avg_return": float(np.mean(rollout_returns)),
                "std_return": float(np.std(rollout_returns)),
                "min_return": float(np.min(rollout_returns)),
                "max_return": float(np.max(rollout_returns)),
                "perf": perf_avg,
                "score": score_avg,
                "collision_rate": collision_rate_avg,
                "offroad_rate": offroad_rate_avg,
                "completion_rate": completion_rate_avg,
                "dnf_rate": dnf_rate_avg,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

    elapsed = time.time() - t0
    print(f"\n✓ Evaluation complete!")
    print(f"  Total time: {elapsed:.0f}s ({len(combos) / elapsed:.2f} combos/sec)")
    print(f"  Results written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate oracle policy in homogeneous behavioral environments"
    )
    parser.add_argument("--policy-name", type=str, required=True, help="Path to trained model checkpoint (.pt file)",)
    parser.add_argument("--co-player-policy-name", type=str, default=None, help="Path to co-player model checkpoint (.pt file) for population play",)
    parser.add_argument("--ego-preset", type=str, default="cautious", choices=["cautious", "aggressive"], help="Ego agent conditioning preset",)
    parser.add_argument("--num-maps", type=int, default=100, help="Number of maps to evaluate on",)
    parser.add_argument("--num-rollouts", type=int, default=16, help="Number of independent rollouts per conditioning combination",)
    parser.add_argument("--num-workers", type=int, default=16, help="Number of parallel workers for environment vectorization",)
    parser.add_argument("--oracle", action="store_true", default=False, help="Whether policy was trained with oracle conditioning",)
    parser.add_argument("--condition-type", type=str, default="none", help="Whether policy was trained with oracle conditioning",)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on",)
    args = parser.parse_args()

    evaluate_oracle_policy(
        policy_path=args.policy_name,
        co_player_policy_path=args.co_player_policy_name,
        num_maps=args.num_maps,
        num_rollouts=args.num_rollouts,
        num_workers=args.num_workers,
        oracle_mode=args.oracle,
        condition_type=args.condition_type,
        device=args.device,
        ego_preset=args.ego_preset,
    )
