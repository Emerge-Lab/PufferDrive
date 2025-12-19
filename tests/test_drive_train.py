#!/usr/bin/env python3
"""
Test script for PufferDrive training functionality on CPU.
Runs a 10s training session to verify the end-to-end setup works.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy


def test_drive_training():
    """Test PufferDrive training until reaching 50K global steps on CPU."""
    print("Testing PufferDrive training on CPU...")

    try:
        env_name = "puffer_drive"
        args = load_config(env_name)

        args["train"].update(
            {
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
            }
        )

        args["vec"].update(
            {
                "num_workers": 1,
                "num_envs": 1,
                "batch_size": 1,
            }
        )

        args["env"].update(
            {
                "num_agents": 8,  # 1 env * 8 agents = 8 total <= 16 segments
                "action_type": "discrete",
                "num_maps": 1,
                "map_dir": "resources/drive/binaries",
                "init_mode": "create_all_valid",
                "control_mode": "control_agents",
            }
        )

        args["policy"].update(
            {
                "input_size": 64,
                "hidden_size": 64,  # Smaller than your 256
            }
        )

        args["rnn"].update(
            {
                "input_size": 64,
                "hidden_size": 64,
            }
        )
        args["wandb"] = False
        args["neptune"] = False
        args["eval"] = {
            "eval_interval": 10000,
            "num_episodes": 4,
            "wosac_realism_eval": False,
            "human_replay_eval": True,
            "human_replay_num_agents": 8,
        }

        # Load components
        print("Loading environment and policy...")
        vecenv = load_env(env_name, args)
        policy = load_policy(args, vecenv, env_name)

        # Initialize training
        train_config = dict(**args["train"], env=env_name, eval=args.get("eval", {}))
        pufferl = PuffeRL(train_config, vecenv, policy, logger=None)

        # Train until reaching 50K steps
        target_steps = 50000
        print(f"Starting training until {target_steps} global steps...")
        start_time = time.time()
        last_step = 0
        last_progress_time = start_time

        while pufferl.global_step < target_steps:
            try:
                pufferl.evaluate()
                pufferl.train()

                current_time = time.time()
                elapsed = current_time - start_time
                progress = pufferl.global_step / target_steps * 100

                print(
                    f"Training... {pufferl.global_step}/{target_steps} steps ({progress:.1f}%), {elapsed:.0f}s elapsed"
                )

                # Check if training is making progress (allow 60 seconds without progress)
                if pufferl.global_step > last_step:
                    last_step = pufferl.global_step
                    last_progress_time = current_time
                elif current_time - last_progress_time > 60:
                    raise RuntimeError("Training appears stuck - no progress for 60+ seconds")

            except Exception as e:
                # Check if multiprocessing workers crashed
                if hasattr(pufferl.vecenv, "pool") and pufferl.vecenv.pool:
                    for worker in pufferl.vecenv.pool._pool:
                        if not worker.is_alive():
                            raise RuntimeError(f"Training worker process died: {e}")

                # Re-raise any other exceptions
                raise RuntimeError(f"Training failed: {e}")

        # Success - reached target steps
        final_time = time.time() - start_time
        print(f"Successfully reached {pufferl.global_step} steps in {final_time:.0f}s!")

        # Attempt minimal cleanup to stop background threads
        try:
            # Stop the utilization monitoring thread
            if hasattr(pufferl, "utilization") and hasattr(pufferl.utilization, "stop"):
                pufferl.utilization.stop()
        except:
            pass

        print("Training test completed successfully!")

        # Force exit to avoid hanging due to background threads
        import os

        os._exit(0)

    except Exception as e:
        print(f"Training test failed: {e}")
        return False


if __name__ == "__main__":
    if test_drive_training():
        print("Test passed!")
        sys.exit(0)
    else:
        print("Test failed!")
        sys.exit(1)
