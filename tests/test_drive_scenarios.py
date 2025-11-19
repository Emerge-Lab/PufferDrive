#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy


def run_training_test(env_name, config_overrides, target_steps=10000, test_name=""):
    print(f"\nTesting: {test_name}")

    try:
        args = load_config(env_name)

        args["train"].update({
            "device": "cpu",
            "compile": False,
            "total_timesteps": 100000,
            "batch_size": 128,
            "bptt_horizon": 8,
            "minibatch_size": 128,
            "max_minibatch_size": 128,
            "update_epochs": 1,
            "render": False,
            "checkpoint_interval": 999999,
            "learning_rate": 0.001,
        })

        args["vec"].update({
            "num_workers": 1,
            "num_envs": 1,
            "batch_size": 1,
        })

        args["env"].update({
            "num_agents": 8,
            "action_type": "discrete",
            "num_maps": 1,
        })

        args["policy"].update({
            "input_size": 64,
            "hidden_size": 64,
        })

        args["rnn"].update({
            "input_size": 64,
            "hidden_size": 64,
        })

        args["wandb"] = False
        args["neptune"] = False
        args["eval"] = {
            "eval_interval": 10000,
            "num_episodes": 4,
            "wosac_realism_eval": False,
            "human_replay_eval": False,
            "human_replay_num_agents": 8,
        }

        for section, updates in config_overrides.items():
            if section not in args:
                args[section] = {}
            args[section].update(updates)

        vecenv = load_env(env_name, args)
        policy = load_policy(args, vecenv, env_name)

        train_config = dict(**args["train"], env=env_name, eval=args.get("eval", {}))
        pufferl = PuffeRL(train_config, vecenv, policy, logger=None)

        start_time = time.time()
        last_step = 0
        last_progress_time = start_time

        while pufferl.global_step < target_steps:
            try:
                pufferl.evaluate()
                pufferl.train()

                current_time = time.time()
                progress = pufferl.global_step / target_steps * 100
                print(f"Training... {pufferl.global_step}/{target_steps} ({progress:.1f}%)")

                if pufferl.global_step > last_step:
                    last_step = pufferl.global_step
                    last_progress_time = current_time
                elif current_time - last_progress_time > 60:
                    raise RuntimeError("Training stuck")

            except Exception as e:
                if hasattr(pufferl.vecenv, "pool") and pufferl.vecenv.pool:
                    for worker in pufferl.vecenv.pool._pool:
                        if not worker.is_alive():
                            raise RuntimeError(f"Worker died: {e}")
                raise RuntimeError(f"Training failed: {e}")

        print(f"{test_name} completed!")

        try:
            if hasattr(pufferl, "utilization") and hasattr(pufferl.utilization, "stop"):
                pufferl.utilization.stop()
        except:
            pass

        os._exit(0)

    except Exception as e:
        print(f"{test_name} failed: {e}")
        sys.exit(1)


def test_scenario_1_normal_training():
    run_training_test(
        env_name="puffer_drive",
        config_overrides={
            "co_player_policy": {"enabled": False},
            "policy.conditioning": {"type": "none"},
        },
        test_name="Scenario 1: Normal Training (Baseline)"
    )


def test_scenario_2_conditioned_self_play():
    run_training_test(
        env_name="puffer_drive",
        config_overrides={
            "co_player_policy": {"enabled": False},
            "policy.conditioning": {
                "type": "reward",
                "collision_weight_lb": -1.0,
                "collision_weight_ub": 0.0,
                "offroad_weight_lb": -0.4,
                "offroad_weight_ub": 0.0,
                "goal_weight_lb": 0.0,
                "goal_weight_ub": 1.0,
            },
        },
        test_name="Scenario 2: Conditioned Self-Play"
    )


def test_scenario_3_adaptive_self_play():
    run_training_test(
        env_name="puffer_adaptive_drive",
        config_overrides={
            "env": {"k_scenarios": 2},
            "co_player_policy": {"enabled": False},
            "policy.conditioning": {"type": "none"},
        },
        test_name="Scenario 3: Adaptive Agent Self-Play"
    )


def test_scenario_4_population_play_normal():
    run_training_test(
        env_name="puffer_drive",
        config_overrides={
            "env": {"num_agents": 16},
            "co_player_policy": {
                "enabled": True,
                "num_ego_agents": 8,
                "policy_name": "Drive",
                "rnn_name": "Recurrent",
                "policy_path": "resources/drive/policies/varied_discount.pt",
                "input_size": 64,
                "hidden_size": 64,
            },
            "co_player_rnn": {
                "input_size": 64,
                "hidden_size": 64,
            },
            "policy.conditioning": {"type": "none"},
            "co_player_policy.conditioning": {
                "type": "all",
                "collision_weight_lb": -1.0,
                "collision_weight_ub": 0.0,
                "offroad_weight_lb": -0.4,
                "offroad_weight_ub": 0.0,
                "goal_weight_lb": 0.0,
                "goal_weight_ub": 1.0,
                "entropy_weight_lb": 0.0,
                "entropy_weight_ub": 0.001,
                "discount_weight_lb": 0.98,
                "discount_weight_ub": 0.80,
            },
        },
        test_name="Scenario 4: Population Play (Normal + Conditioned Co-players)"
    )


def test_scenario_5_population_play_adaptive():
    run_training_test(
        env_name="puffer_adaptive_drive",
        config_overrides={
            "env": {"num_agents": 16, "k_scenarios": 2},
            "co_player_policy": {
                "enabled": True,
                "num_ego_agents": 8,
                "policy_name": "Drive",
                "rnn_name": "Recurrent",
                "policy_path": "resources/drive/policies/varied_discount.pt",
                "input_size": 64,
                "hidden_size": 64,
            },
            "co_player_rnn": {
                "input_size": 64,
                "hidden_size": 64,
            },
            "policy.conditioning": {"type": "none"},
            "co_player_policy.conditioning": {
                "type": "all",
                "collision_weight_lb": -1.0,
                "collision_weight_ub": 0.0,
                "offroad_weight_lb": -0.4,
                "offroad_weight_ub": 0.0,
                "goal_weight_lb": 0.0,
                "goal_weight_ub": 1.0,
                "entropy_weight_lb": 0.0,
                "entropy_weight_ub": 0.001,
                "discount_weight_lb": 0.98,
                "discount_weight_ub": 0.80,
            },
        },
        test_name="Scenario 5: Population Play (Adaptive + Conditioned Co-players)"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        if scenario == "1" or scenario == "normal":
            test_scenario_1_normal_training()
        elif scenario == "2" or scenario == "conditioned":
            test_scenario_2_conditioned_self_play()
        elif scenario == "3" or scenario == "adaptive":
            test_scenario_3_adaptive_self_play()
        elif scenario == "4" or scenario == "population":
            test_scenario_4_population_play_normal()
        elif scenario == "5" or scenario == "population_adaptive":
            test_scenario_5_population_play_adaptive()
        else:
            print(f"Unknown scenario: {scenario}")
            sys.exit(1)
    else:
        print("Running all scenarios...")
        test_scenario_1_normal_training()
        test_scenario_2_conditioned_self_play()
        test_scenario_3_adaptive_self_play()
        test_scenario_4_population_play_normal()
        test_scenario_5_population_play_adaptive()
