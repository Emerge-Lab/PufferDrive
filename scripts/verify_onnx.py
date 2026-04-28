"""Verify an exported ONNX model against saved example input/output tensors.

Accepts either the .onnx or the .onnx.data file for --onnx; the .onnx file
is used in both cases.

Args:
    --onnx    Path to the exported .onnx file (or .onnx.data — suffix is stripped automatically)
    --input   Path to the example_input.pt file saved by export_onnx.py
    --output  Path to the example_output.pt file saved by export_onnx.py

Usage:
    python -m scripts.verify_onnx \
        --onnx   pufferlib/resources/drive/onnx_files/model_puffer_drive_011500.onnx \
        --input  pufferlib/resources/drive/onnx_files/example_input.pt \
        --output pufferlib/resources/drive/onnx_files/example_output.pt
"""

import argparse
import os
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx.external_data_helper import load_external_data_for_model


def verify(args):
    # Allow passing either the .onnx or the .onnx.data file
    onnx_path = args.onnx
    if onnx_path.endswith(".onnx.data"):
        onnx_path = onnx_path[: -len(".data")]
    input_path = args.input
    output_path = args.output

    print(f"ONNX model:     {onnx_path}")
    print(f"Example input:  {input_path}")
    print(f"Example output: {output_path}")

    # Load example tensors
    example_input = torch.load(input_path, map_location="cpu")
    saved = torch.load(output_path, map_location="cpu")
    ref_logits = saved["logits"]
    ref_value = saved["value"]

    print(f"\nInput shape:   {example_input.shape}")
    print(f"Ref logits:    {ref_logits.shape}  sample: {ref_logits[0].tolist()}")
    print(f"Ref value:     {ref_value[0].item():.6f}")

    # Load ONNX model — merge external data into memory so ORT can run it
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    proto = onnx.load(onnx_path, load_external_data=False)
    load_external_data_for_model(proto, onnx_dir)
    model_bytes = proto.SerializeToString()

    sess = ort.InferenceSession(model_bytes)
    ort_outs = sess.run(None, {"observation": example_input.numpy()})
    ort_logits, ort_value = ort_outs[0], ort_outs[1]

    print(f"\nONNX logits:   {ort_logits.shape}  sample: {ort_logits[0].tolist()}")
    print(f"ONNX value:    {ort_value[0, 0]:.6f}")

    atol, rtol = 1e-4, 1e-3
    passed = True

    try:
        np.testing.assert_allclose(ref_logits.numpy(), ort_logits, rtol=rtol, atol=atol)
        print("\n  logits:     MATCH")
    except AssertionError as e:
        print("\n  logits:     MISMATCH")
        print(f"  {e}")
        passed = False

    try:
        np.testing.assert_allclose(ref_value.numpy(), ort_value, rtol=rtol, atol=atol)
        print("  value:      MATCH")
    except AssertionError as e:
        print("  value:      MISMATCH")
        print(f"  {e}")
        passed = False

    if "trajectory" in saved:
        ref_traj = saved["trajectory"]
        print(f"\nRef trajectory: {ref_traj.shape}  first point: {ref_traj[0, 0].tolist()}")
        if len(ort_outs) < 3:
            print("  trajectory: MISMATCH — ONNX model has no trajectory output")
            passed = False
        else:
            ort_traj = ort_outs[2]
            print(f"ONNX trajectory:{ort_traj.shape}  first point: {ort_traj[0, 0].tolist()}")
            try:
                np.testing.assert_allclose(ref_traj.numpy(), ort_traj, rtol=rtol, atol=atol)
                print("  trajectory: MATCH")
            except AssertionError as e:
                print("  trajectory: MISMATCH")
                print(f"  {e}")
                passed = False
    else:
        if len(ort_outs) > 2:
            print(f"\n  WARNING: ONNX model has {len(ort_outs)} outputs but example_output.pt has no trajectory key")

    if passed:
        print("\nAll outputs verified successfully.")
    else:
        print("\nVerification FAILED — see mismatches above.")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify exported ONNX model")
    parser.add_argument(
        "--onnx",
        default="pufferlib/resources/drive/onnx_files/model_puffer_drive_011500.onnx",
    )
    parser.add_argument(
        "--input",
        default="pufferlib/resources/drive/onnx_files/example_input.pt",
    )
    parser.add_argument(
        "--output",
        default="pufferlib/resources/drive/onnx_files/example_output.pt",
    )
    verify(parser.parse_args())
