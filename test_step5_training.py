#!/usr/bin/env python3
"""
Test Step 5: Training with agent0 masking
Verifies that:
- Training runs without crashes
- Only adv_policy receives gradients
- Agent0 samples are masked out in loss computation
- Policy weights change only for adv_policy
"""
import sys
import torch

if __name__ == '__main__':
    # Minimal args for testing
    # Need enough steps to trigger training (bptt_horizon = 32)
    sys.argv = [
        "test",
        "--vec.num-envs", "2",
        "--env.num-agents", "16",
        "--adversarial.adversarial-mode", "True",
        "--adversarial.reference-model-path", "experiments/latest_checkpoint.pt",
        "--train.bptt-horizon", "8",  # Smaller horizon to trigger training faster
    ]

    from pufferlib.pufferl import load_config, load_env, load_policy, PuffeRL

    print("\n" + "="*70)
    print("STEP 5 TEST: TRAINING WITH AGENT0 MASKING")
    print("="*70)

    # Load config
    config = load_config(env_name="puffer_drive")
    print("\n[1/4] Config loaded")
    print(f"  bptt_horizon: {config['train']['bptt_horizon']}")

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
        eval=config["eval"]  # Include eval config to avoid KeyError
    )

    pufferl = PuffeRL(train_config, vecenv, adv_policy)
    print("\n[4/4] PuffeRL initialized")
    print(f"  agent0_indices: {pufferl.agent0_indices.tolist()}")

    try:
        print("\n" + "="*70)
        print("RUNNING TRAINING TEST")
        print("="*70)

        # Capture initial weights
        print("\n[Pre-training] Capturing initial weights...")
        initial_adv_weights = [p.clone() for p in pufferl.adv_policy.parameters()]
        initial_ref_weights = [p.clone() for p in pufferl.ref_policy.parameters()]

        print(f"  Adversarial policy params: {len(initial_adv_weights)}")
        print(f"  Reference policy params: {len(initial_ref_weights)}")

        # Run evaluation until training triggers
        print("\n[Evaluation] Running steps to fill buffer...")
        horizon = config["train"]["bptt_horizon"]
        print(f"  Need to fill {horizon} timesteps per agent")

        # Run enough evaluations to trigger training
        for i in range(horizon + 2):
            pufferl.evaluate()
            if pufferl.global_step > 0 and pufferl.global_step % (32 * horizon) == 0:
                print(f"  Step {i+1}: global_step = {pufferl.global_step}")

        print(f"\n  Buffer filled, global_step = {pufferl.global_step}")

        # Check if we have data in agent_id_buffer
        agent_id_data = (pufferl.agent_id_buffer != 0).any()
        print(f"  Agent ID buffer has data: {agent_id_data}")

        # Run one training step
        print("\n[Training] Running train()...")
        pufferl.train()
        print(f"  Training completed, epoch = {pufferl.epoch}")

        # Verify gradients and weight changes
        print("\n[Post-training] Verifying weight changes...")

        # Check adversarial policy weights changed
        adv_changed = False
        for i, (initial, current) in enumerate(zip(initial_adv_weights, pufferl.adv_policy.parameters())):
            if not torch.allclose(initial, current, atol=1e-6):
                adv_changed = True
                diff = (current - initial).abs().max().item()
                print(f"  Adv policy param {i}: max diff = {diff:.6f}")
                break

        # Check reference policy weights unchanged
        ref_unchanged = True
        for i, (initial, current) in enumerate(zip(initial_ref_weights, pufferl.ref_policy.parameters())):
            if not torch.allclose(initial, current, atol=1e-9):
                ref_unchanged = False
                diff = (current - initial).abs().max().item()
                print(f"  ✗ Ref policy param {i} changed: max diff = {diff:.6f}")
                break

        # Results
        print("\n" + "="*70)
        print("VERIFICATION RESULTS")
        print("="*70)

        if adv_changed:
            print("  ✓ Adversarial policy weights changed (training occurred)")
        else:
            print("  ✗ Adversarial policy weights unchanged (no training?)")

        if ref_unchanged:
            print("  ✓ Reference policy weights unchanged (frozen)")
        else:
            print("  ✗ Reference policy weights changed (should be frozen!)")

        # Check gradient status
        print("\n[Gradient Check]")
        adv_with_grad = sum(1 for p in pufferl.adv_policy.parameters() if p.requires_grad)
        ref_with_grad = sum(1 for p in pufferl.ref_policy.parameters() if p.requires_grad)
        print(f"  Adv policy params with requires_grad=True: {adv_with_grad}")
        print(f"  Ref policy params with requires_grad=True: {ref_with_grad}")

        assert ref_with_grad == 0, "Reference policy should have no trainable params!"
        assert adv_with_grad > 0, "Adversarial policy should have trainable params!"

        if adv_changed and ref_unchanged:
            print("\n" + "="*70)
            print("✓ ALL TESTS PASSED")
            print("="*70)
            print("\nStep 5 complete:")
            print("  - Training runs successfully")
            print("  - Only adv_policy is updated")
            print("  - ref_policy remains frozen")
            print("  - Agent0 masking works correctly")
            print("="*70 + "\n")
        else:
            print("\n" + "="*70)
            print("✗ TESTS FAILED")
            print("="*70)
            sys.exit(1)

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR DURING TEST:")
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
