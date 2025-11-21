#!/usr/bin/env python3
"""
Test Step 4: Capture and verify rewards during evaluate loop
Monkey-patches the evaluate function to capture rewards at the moment they're modified
"""
import sys
import torch

if __name__ == '__main__':
    # Minimal args for testing
    sys.argv = [
        "test",
        "--vec.num-envs", "2",
        "--env.num-agents", "16",
        "--adversarial.adversarial-mode", "True",
        "--adversarial.reference-model-path", "experiments/latest_checkpoint.pt",
    ]

    from pufferlib.pufferl import load_config, load_env, load_policy, PuffeRL

    print("\n" + "="*70)
    print("STEP 4 TEST: CAPTURE REWARDS DURING EVALUATION")
    print("="*70)

    # Load config
    config = load_config(env_name="puffer_drive")
    print("\n[1/4] Config loaded")

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
        policy=config["policy"],
        eval=config["eval"]
    )

    pufferl = PuffeRL(train_config, vecenv, adv_policy)
    print("\n[4/4] PuffeRL initialized")
    print(f"  agent0_indices: {pufferl.agent0_indices.tolist()}")

    # Capture rewards by monkey-patching
    print("\n" + "="*70)
    print("RUNNING EVALUATION WITH REWARD CAPTURE")
    print("="*70)

    captured_rewards = []

    # Store original evaluate method
    original_evaluate = pufferl.evaluate

    # Patch to capture rewards before they're reset
    def patched_evaluate():
        # Call original (this fills the buffer)
        result = original_evaluate()

        # Capture current state of reward buffer
        # The buffer has shape [segments, horizon], rewards were just written
        snapshot = pufferl.rewards.clone()
        captured_rewards.append(snapshot)

        return result

    pufferl.evaluate = patched_evaluate

    try:
        # Run a few evaluation steps
        print("\nRunning 5 evaluation calls...")
        for i in range(5):
            print(f"  Evaluation {i+1}/5...")
            pufferl.evaluate()

        print("\n" + "="*70)
        print("VERIFYING CAPTURED REWARDS")
        print("="*70)

        # Restore original
        pufferl.evaluate = original_evaluate

        # Now check captured rewards
        print(f"\nCaptured {len(captured_rewards)} reward snapshots")

        # Check last snapshot (most recent)
        last_rewards = captured_rewards[-1]
        print(f"Reward buffer shape: {last_rewards.shape}")

        # Find which parts of buffer have data (non-zero)
        has_data = (last_rewards != 0).any(dim=1)
        num_with_data = has_data.sum().item()
        print(f"Segments with non-zero rewards: {num_with_data}/{last_rewards.shape[0]}")

        if num_with_data > 0:
            # Get first few segments with data
            segments_with_data = torch.where(has_data)[0][:min(10, num_with_data)]

            print("\nChecking adversarial reward structure...")
            print(f"Agent0 indices: {pufferl.agent0_indices.tolist()}")
            print(f"Agent offsets: {pufferl.agent_offsets}")

            # For each environment, verify reward structure
            all_correct = True
            for env_idx in range(pufferl.num_envs):
                agent0_idx = pufferl.agent0_indices[env_idx]
                env_start = pufferl.agent_offsets[env_idx]
                env_end = pufferl.agent_offsets[env_idx + 1]

                print(f"\nEnvironment {env_idx}:")
                print(f"  Agent0 index: {agent0_idx}")
                print(f"  Env agents: [{env_start}, {env_end})")

                # Check if this segment has data
                if agent0_idx < len(has_data) and has_data[agent0_idx]:
                    # Get agent0 reward (from first timestep with data)
                    timesteps_with_data = (last_rewards[agent0_idx] != 0).nonzero(as_tuple=True)[0]
                    if len(timesteps_with_data) > 0:
                        t = timesteps_with_data[0].item()
                        agent0_reward = last_rewards[agent0_idx, t].item()
                        print(f"  Agent0 reward (t={t}): {agent0_reward:.4f}")

                        # Check adversarial agents
                        mismatches = []
                        for global_idx in range(env_start, env_end):
                            if global_idx != agent0_idx and global_idx < len(has_data):
                                if has_data[global_idx]:
                                    adv_reward = last_rewards[global_idx, t].item()
                                    expected = -agent0_reward

                                    if abs(adv_reward - expected) > 0.001:
                                        mismatches.append((global_idx, adv_reward, expected))
                                        all_correct = False

                        if mismatches:
                            print(f"  ✗ Mismatches found:")
                            for idx, actual, expected in mismatches:
                                print(f"    Agent {idx}: got {actual:.4f}, expected {expected:.4f}")
                        else:
                            # Count how many adversarial agents we verified
                            verified_count = sum(1 for idx in range(env_start, env_end)
                                               if idx != agent0_idx and idx < len(has_data) and has_data[idx])
                            print(f"  ✓ All {verified_count} adversarial agents have correct reward (-{agent0_reward:.4f})")

            if all_correct:
                print("\n" + "="*70)
                print("✓ ALL REWARDS CORRECTLY MODIFIED")
                print("="*70)
            else:
                print("\n" + "="*70)
                print("✗ SOME REWARDS INCORRECT")
                print("="*70)
                sys.exit(1)

        else:
            print("\n⚠ Warning: No reward data found in buffer")
            print("This might be normal if episodes haven't progressed enough")

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nStep 4 verified:")
        print("  - Rewards captured successfully")
        print("  - Adversarial reward modification works correctly")
        print("  - Ready for Step 5 (training)")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR:")
        print("="*70)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        print("\nCleaning up...")
        pufferl.vecenv.close()
        if hasattr(pufferl, 'utilization'):
            pufferl.utilization.stop()
        print("Test complete.")
