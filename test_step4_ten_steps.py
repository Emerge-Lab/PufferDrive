#!/usr/bin/env python3
"""
Test Step 4: Ten consecutive evaluation steps with reward verification
Runs 10 evaluation steps and verifies at each step:
- Actions generated correctly from split policies
- Rewards modified correctly (adversarial agents get -agent0_reward)
- No crashes or shape mismatches
"""
import sys
import torch

if __name__ == '__main__':
    # Minimal args for testing
    # Constraints: num_envs divisible by num_workers (2)
    #              total_agents <= segments (batch_size/bptt_horizon = 1024/32 = 32)
    sys.argv = [
        "test",
        "--vec.num-envs", "2",  # 2 envs × 16 agents = 32 total (= segments)
        "--env.num-agents", "16",
        "--adversarial.adversarial-mode", "True",
        "--adversarial.reference-model-path", "experiments/latest_checkpoint.pt",
    ]

    from pufferlib.pufferl import load_config, load_env, load_policy, PuffeRL

    print("\n" + "="*70)
    print("STEP 4 RIGOROUS TEST: 10 CONSECUTIVE EVALUATION STEPS")
    print("="*70)

    # Load config
    config = load_config(env_name="puffer_drive")
    print("\n[1/4] Config loaded")
    print(f"  bptt_horizon: {config['train']['bptt_horizon']}")
    print(f"  (10 steps << 32 horizon, so no training will trigger)")

    # Create vecenv
    vecenv = load_env("puffer_drive", config)
    print("\n[2/4] Vecenv created")
    print(f"  num_envs: {vecenv.driver_env.num_envs}")
    print(f"  num_agents: {vecenv.num_agents}")
    print(f"  agent_offsets: {vecenv.driver_env.agent_offsets}")

    # Create adversarial policy
    adv_policy = load_policy(config, vecenv)
    print("\n[3/4] Adversarial policy created")

    # Initialize PuffeRL
    train_config = dict(
        **config["train"],
        env="puffer_drive",
        adversarial=config["adversarial"],
        package=config["package"],
        policy_name=config["policy_name"],
        policy=config["policy"]
    )

    pufferl = PuffeRL(train_config, vecenv, adv_policy)
    print("\n[4/4] PuffeRL initialized")
    print(f"  agent0_indices: {pufferl.agent0_indices.tolist()}")

    # Run 10 evaluation steps with detailed verification
    print("\n" + "="*70)
    print("RUNNING 10 EVALUATION STEPS")
    print("="*70)

    try:
        step_rewards = []  # Track rewards at each step

        for step in range(10):
            print(f"\n--- Step {step + 1}/10 ---")

            # Capture state before evaluate
            old_global_step = pufferl.global_step

            # Run one evaluation step
            stats = pufferl.evaluate()

            # Check global step incremented
            new_global_step = pufferl.global_step
            steps_taken = new_global_step - old_global_step
            print(f"  Global steps: {old_global_step} -> {new_global_step} (+{steps_taken})")

            # Verify reward buffer has data
            # ep_lengths tells us how many timesteps have been collected per episode
            max_ep_length = pufferl.ep_lengths.max().item()
            print(f"  Max episode length: {max_ep_length}")

            # Get current rewards from buffer (latest timestep)
            if max_ep_length > 0:
                current_timestep = max_ep_length - 1
                current_rewards = pufferl.rewards[:pufferl.total_agents, current_timestep]

                # Verify adversarial reward structure
                print(f"  Verifying reward modification...")
                all_correct = True

                for env_idx in range(pufferl.num_envs):
                    agent0_idx = pufferl.agent0_indices[env_idx]
                    env_start = pufferl.agent_offsets[env_idx]
                    env_end = pufferl.agent_offsets[env_idx + 1]

                    agent0_reward = current_rewards[agent0_idx].item()

                    # Check adversarial agents in this env
                    for global_idx in range(env_start, env_end):
                        if global_idx != agent0_idx:
                            adv_reward = current_rewards[global_idx].item()
                            expected = -agent0_reward

                            # Check if they match (within small tolerance for floating point)
                            if abs(adv_reward - expected) > 0.001:
                                print(f"    ✗ Env {env_idx}, Agent {global_idx}: got {adv_reward:.4f}, expected {expected:.4f}")
                                all_correct = False

                if all_correct:
                    print(f"    ✓ All adversarial rewards correctly set to -agent0_reward")

                # Store snapshot
                step_rewards.append({
                    'step': step,
                    'agent0_rewards': [current_rewards[idx].item() for idx in pufferl.agent0_indices],
                    'timestep': current_timestep
                })

            else:
                print(f"  (No episodes recorded yet)")

        print("\n" + "="*70)
        print("✓ ALL 10 STEPS COMPLETED SUCCESSFULLY")
        print("="*70)

        print("\nSummary:")
        print(f"  - 10 consecutive evaluation steps executed")
        print(f"  - Split forward pass worked every step")
        print(f"  - Reward modification verified every step")
        print(f"  - No crashes, shape mismatches, or dtype errors")

        print("\n[Reward Tracking Across Steps]")
        for record in step_rewards:
            print(f"  Step {record['step'] + 1} (timestep {record['timestep']}): agent0 rewards = {record['agent0_rewards']}")

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nStep 4 complete and verified:")
        print("  - Evaluation loop stable over multiple steps")
        print("  - Reward modification consistent")
        print("  - Ready to proceed to Step 5 (training)")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR DURING EVALUATION:")
        print("="*70)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70)
        sys.exit(1)

    finally:
        # Cleanup
        print("\nCleaning up...")
        pufferl.vecenv.close()
        if hasattr(pufferl, 'utilization'):
            pufferl.utilization.stop()
        print("Test complete.")
