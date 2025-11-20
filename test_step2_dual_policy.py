#!/usr/bin/env python3
"""
Test Step 2: Dual policy loading with adversarial mode
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
    print("STEP 2 TEST: DUAL POLICY LOADING")
    print("="*70)

    # Load config
    config = load_config(env_name="puffer_drive")
    print("\n[1/4] Config loaded")
    print(f"  adversarial_mode: {config['adversarial']['adversarial_mode']}")
    print(f"  reference_model_path: {config['adversarial']['reference_model_path']}")

    # Create vecenv
    vecenv = load_env("puffer_drive", config)
    print("\n[2/4] Vecenv created")
    print(f"  num_envs: {vecenv.driver_env.num_envs}")
    print(f"  num_agents: {vecenv.num_agents}")
    print(f"  agent_offsets: {vecenv.driver_env.agent_offsets}")

    # Create adversarial policy (fresh, untrained)
    adv_policy = load_policy(config, vecenv)
    print("\n[3/4] Adversarial policy created (fresh)")

    # Initialize PuffeRL (this should load dual policies)
    # Need to pass train config with additional keys for adversarial mode
    train_config = dict(
        **config["train"],
        env="puffer_drive",
        adversarial=config["adversarial"],
        package=config["package"],
        policy_name=config["policy_name"],
        policy=config["policy"]
    )
    try:
        pufferl = PuffeRL(train_config, vecenv, adv_policy)
        print("\n[4/4] PuffeRL initialized with dual policies")
    except Exception as e:
        print(f"\n[4/4] ERROR during PuffeRL initialization:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verification tests
    print("\n" + "="*70)
    print("VERIFICATION TESTS")
    print("="*70)

    # Test 1: Check both policies exist
    print("\n[Test 1] Checking policy attributes...")
    assert hasattr(pufferl, 'ref_policy'), "Missing ref_policy!"
    assert hasattr(pufferl, 'adv_policy'), "Missing adv_policy!"
    print("  ✓ Both ref_policy and adv_policy exist")

    # Test 2: Check reference policy is frozen
    print("\n[Test 2] Checking reference policy is frozen...")
    ref_params_with_grad = sum(1 for p in pufferl.ref_policy.parameters() if p.requires_grad)
    print(f"  Reference policy params with requires_grad=True: {ref_params_with_grad}")
    assert ref_params_with_grad == 0, "Reference policy not fully frozen!"
    print("  ✓ Reference policy is frozen (all requires_grad=False)")

    # Test 3: Check adversarial policy is trainable
    print("\n[Test 3] Checking adversarial policy is trainable...")
    adv_params_with_grad = sum(1 for p in pufferl.adv_policy.parameters() if p.requires_grad)
    adv_total_params = sum(1 for _ in pufferl.adv_policy.parameters())
    print(f"  Adversarial policy params with requires_grad=True: {adv_params_with_grad}/{adv_total_params}")
    assert adv_params_with_grad > 0, "Adversarial policy has no trainable params!"
    print("  ✓ Adversarial policy is trainable")

    # Test 4: Check agent0_indices
    print("\n[Test 4] Checking agent0_indices...")
    print(f"  num_envs: {pufferl.num_envs}")
    print(f"  agent0_indices: {pufferl.agent0_indices.tolist()}")
    expected_agent0s = [pufferl.agent_offsets[i] for i in range(pufferl.num_envs)]
    assert pufferl.agent0_indices.tolist() == expected_agent0s, "agent0_indices mismatch!"
    print(f"  ✓ agent0_indices correct: {expected_agent0s}")

    # Test 5: Check optimizer only has adversarial params
    print("\n[Test 5] Checking optimizer parameters...")
    optimizer_param_ids = set(id(p) for group in pufferl.optimizer.param_groups for p in group['params'])
    adv_param_ids = set(id(p) for p in pufferl.adv_policy.parameters())
    ref_param_ids = set(id(p) for p in pufferl.ref_policy.parameters())

    optimizer_has_adv = len(optimizer_param_ids & adv_param_ids) > 0
    optimizer_has_ref = len(optimizer_param_ids & ref_param_ids) > 0

    print(f"  Optimizer includes adv_policy params: {optimizer_has_adv}")
    print(f"  Optimizer includes ref_policy params: {optimizer_has_ref}")
    assert optimizer_has_adv, "Optimizer missing adversarial params!"
    assert not optimizer_has_ref, "Optimizer should NOT include reference params!"
    print("  ✓ Optimizer only trains adversarial policy")

    # Test 6: Check policies have same architecture
    print("\n[Test 6] Checking policies have same architecture...")
    ref_param_shapes = [tuple(p.shape) for p in pufferl.ref_policy.parameters()]
    adv_param_shapes = [tuple(p.shape) for p in pufferl.adv_policy.parameters()]
    assert ref_param_shapes == adv_param_shapes, "Policy architectures don't match!"
    print(f"  ✓ Both policies have same architecture ({len(ref_param_shapes)} layers)")

    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nStep 2 complete: Dual policy loading works correctly")
    print("  - Reference policy loaded from checkpoint and frozen")
    print("  - Adversarial policy created fresh and trainable")
    print("  - agent0_indices computed correctly")
    print("  - Optimizer only trains adversarial policy")
    print("="*70 + "\n")

    # Cleanup: use proper PuffeRL close method
    print("\nCleaning up...")
    pufferl.vecenv.close()
    if hasattr(pufferl, 'utilization'):
        pufferl.utilization.stop()
    print("Test complete.")
