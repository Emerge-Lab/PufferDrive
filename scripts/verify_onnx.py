import argparse
import numpy as np
import onnx
import onnxruntime as ort


def verify_onnx():
    parser = argparse.ArgumentParser(description="Verify ONNX model")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx file")
    args = parser.parse_args()

    print(f"Checking model: {args.model}")

    # 1. Verify Model Structure with ONNX
    try:
        model = onnx.load(args.model)
        onnx.checker.check_model(model)
        print("ONNX Model Check: Passed (Structure is valid)")
    except Exception as e:
        print(f"ONNX Model Check: Failed\n{e}")
        return

    # 2. Verify Execution with ONNX Runtime
    try:
        print("Starting ONNX Runtime session...")
        ort_session = ort.InferenceSession(args.model)

        print("Session created successfully")

        # Prepare Dummy Inputs based on model expectation
        inputs = {}
        batch_size = 1

        for input_meta in ort_session.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            dtype = input_meta.type

            # Handle dynamic axes (often represented as strings or -1)
            processed_shape = []
            for dim in shape:
                if isinstance(dim, str) or dim is None or dim < 0:
                    processed_shape.append(batch_size)
                else:
                    processed_shape.append(dim)

            print(f"  Input: {name}, Shape: {shape} -> Using: {processed_shape}, Type: {dtype}")

            # Create random input data
            if "float" in dtype:
                data = np.random.randn(*processed_shape).astype(np.float32)
            else:
                data = np.zeros(processed_shape).astype(np.int64)  # Fallback

            inputs[name] = data

        # Run Inference
        outputs = ort_session.run(None, inputs)

        print("Inference Verification: Passed")
        print("\nOutput Shapes:")
        for output_meta, output_data in zip(ort_session.get_outputs(), outputs):
            print(f"  Output: {output_meta.name}, Shape: {output_data.shape}")

    except Exception as e:
        print(f"Inference Verification: Failed\n{e}")


if __name__ == "__main__":
    verify_onnx()
