#!/usr/bin/env python3
"""
Test script for PufferDrive training functionality on CPU.
Runs a 5-minute training session to verify the end-to-end setup works.
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy


def test_drive_training():
    """Test PufferDrive training for 5 minutes on CPU."""
    print("Testing PufferDrive training on CPU...")

    try:
        # Load and configure for CPU
        env_name = "puffer_drive"
        args = load_config(env_name)

        # CPU-friendly settings
        args["train"].update(
            {
                "device": "cpu",
                "compile": False,
                "total_timesteps": 5000,
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
                "num_agents": 8,
                "action_type": "discrete",
                "num_maps": 1,
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

        # Load components
        print("Loading environment and policy...")
        vecenv = load_env(env_name, args)
        policy = load_policy(args, vecenv, env_name)

        # Initialize training
        train_config = dict(**args["train"], env=env_name)
        pufferl = PuffeRL(train_config, vecenv, policy, logger=None)

        # Train for 5 minutes
        print("Starting 5-minute training...")
        start_time = time.time()
        max_time = 300  # 5 minutes
        last_step = 0

        while time.time() - start_time < max_time:
            try:
                pufferl.evaluate()
                pufferl.train()

                elapsed = time.time() - start_time
                print(f"Training... {elapsed:.0f}s elapsed, {pufferl.global_step} steps")

                # Check if training is making progress
                if pufferl.global_step == last_step and elapsed > 30:
                    raise RuntimeError("Training appears stuck - no progress for 30+ seconds")
                last_step = pufferl.global_step

            except Exception as e:
                # Check if multiprocessing workers crashed
                if hasattr(pufferl.vecenv, "pool") and pufferl.vecenv.pool:
                    for worker in pufferl.vecenv.pool._pool:
                        if not worker.is_alive():
                            raise RuntimeError(f"Training worker process died: {e}")

                # Re-raise any other exceptions
                raise RuntimeError(f"Training failed: {e}")

        # Cleanup
        pufferl.close()
        vecenv.close()

        print("Training test completed successfully!")
        return True

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
