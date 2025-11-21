#!/usr/bin/env python3
"""
Test Step 6: Checkpoint save/load for dual policies
Verifies that:
- Checkpoints save both ref_policy and adv_policy
- Loading from dual-policy checkpoint works
- Resuming training preserves weights correctly
"""
import sys
import torch
import os
import tempfile
import shutil

if __name__ == '__main__':
    # Use a temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()

    try:
        # Minimal args for testing
        sys.argv = [
            "test",
            "--vec.num-envs", "2",
            "--env.num-agents", "16",
            "--adversarial.adversarial-mode", "True",
            "--adversarial.reference-model-path", "experiments/latest_checkpoint.pt",
            "--train.data-dir", temp_dir,  # Use temp directory
        ]

        from pufferlib.pufferl import load_config, load_env, load_policy, PuffeRL

        print("\n" + "="*70)
        print("STEP 6 TEST: CHECKPOINT SAVE/LOAD FOR DUAL POLICIES")
        print("="*70)

        # Load config
        config = load_config(env_name="puffer_drive")
        print("\n[1/4] Config loaded")
        print(f"  Checkpoint dir: {temp_dir}")

        # Create vecenv
        vecenv = load_env("puffer_drive", config)
        print("\n[2/4] Vecenv created")

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

        print("\n" + "="*70)
        print("TEST 1: SAVE DUAL-POLICY CHECKPOINT")
        print("="*70)

        # Capture weights before saving
        print("\n[Pre-save] Capturing policy weights...")
        ref_weights_before = [p.clone() for p in pufferl.ref_policy.parameters()]
        adv_weights_before = [p.clone() for p in pufferl.adv_policy.parameters()]

        # Save checkpoint
        print("\n[Save] Calling save_checkpoint()...")
        checkpoint_path = pufferl.save_checkpoint()
        print(f"  Checkpoint saved to: {checkpoint_path}")

        # Verify file exists
        assert os.path.exists(checkpoint_path), "Checkpoint file not created!"
        print(f"  ✓ Checkpoint file exists")

        # Load and inspect checkpoint
        print("\n[Inspect] Loading checkpoint to verify structure...")
        saved_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  Checkpoint type: {type(saved_checkpoint)}")

        if isinstance(saved_checkpoint, dict):
            print(f"  Checkpoint keys: {list(saved_checkpoint.keys())}")
            assert 'ref_policy' in saved_checkpoint, "Missing ref_policy in checkpoint!"
            assert 'adv_policy' in saved_checkpoint, "Missing adv_policy in checkpoint!"
            print(f"  ✓ Contains ref_policy ({len(saved_checkpoint['ref_policy'])} params)")
            print(f"  ✓ Contains adv_policy ({len(saved_checkpoint['adv_policy'])} params)")
        else:
            print("  ✗ Checkpoint is not a dict (should be for adversarial mode)")
            sys.exit(1)

        print("\n" + "="*70)
        print("TEST 2: LOAD FROM DUAL-POLICY CHECKPOINT")
        print("="*70)

        # Close current pufferl
        print("\n[Cleanup] Closing current PuffeRL instance...")
        pufferl.vecenv.close()
        if hasattr(pufferl, 'utilization'):
            pufferl.utilization.stop()

        # Create new vecenv for fresh start
        print("\n[Reload] Creating fresh environment and policies...")
        vecenv2 = load_env("puffer_drive", config)

        # Load policy from the dual-policy checkpoint we just saved
        config["load_model_path"] = checkpoint_path
        adv_policy2 = load_policy(config, vecenv2)
        print(f"  Policy loaded from: {checkpoint_path}")

        # Initialize new PuffeRL
        pufferl2 = PuffeRL(train_config, vecenv2, adv_policy2)
        print(f"  New PuffeRL initialized")

        # Verify loaded weights match original
        print("\n[Verify] Comparing loaded weights with original...")
        ref_weights_after = list(pufferl2.ref_policy.parameters())
        adv_weights_after = list(pufferl2.adv_policy.parameters())

        # Check reference policy weights match
        ref_match = all(torch.allclose(w1, w2, atol=1e-6)
                       for w1, w2 in zip(ref_weights_before, ref_weights_after))

        # Check adversarial policy weights match
        adv_match = all(torch.allclose(w1, w2, atol=1e-6)
                       for w1, w2 in zip(adv_weights_before, adv_weights_after))

        if ref_match:
            print("  ✓ Reference policy weights match")
        else:
            print("  ✗ Reference policy weights differ")

        if adv_match:
            print("  ✓ Adversarial policy weights match")
        else:
            print("  ✗ Adversarial policy weights differ")

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        if ref_match and adv_match:
            print("\n✓ ALL TESTS PASSED")
            print("\nStep 6 complete:")
            print("  - Dual-policy checkpoints save correctly")
            print("  - Both ref_policy and adv_policy preserved")
            print("  - Loading from checkpoint restores weights")
            print("  - Ready for resuming training")
            print("="*70 + "\n")
        else:
            print("\n✗ TESTS FAILED")
            print("  Weights do not match after save/load")
            print("="*70 + "\n")
            sys.exit(1)

        # Cleanup second instance
        pufferl2.vecenv.close()
        if hasattr(pufferl2, 'utilization'):
            pufferl2.utilization.stop()

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR DURING TEST:")
        print("="*70)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup temp directory
        print("\nCleaning up temporary directory...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("Test complete.")
