# Exporting PufferDrive Models to ONNX

PufferDrive provides a utility script to export trained PyTorch models to the ONNX format. This is useful for deployment, inference optimization, or using the model in environments that support ONNX Runtime.

## Usage

The export script is located at `scripts/export_onnx.py`. You can run it from the root of the repository.

### Basic Usage

To export a model using default settings (assuming you have a checkpoint at the default path or specify one):

```bash
python scripts/export_onnx.py --checkpoint path/to/your/checkpoint.pt
```

This will create an `.onnx` file in the same directory as the checkpoint, with the same name (e.g., `checkpoint.onnx`).

### Specifying Output Path

You can specify a custom output path for the ONNX file:

```bash
python scripts/export_onnx.py \
    --checkpoint experiments/my_experiment/model_000100.pt \
    --output models/my_model.onnx
```

### Specifying Environment

If you are using a specific environment configuration, you can specify it with `--env`:

```bash
python scripts/export_onnx.py --env puffer_drive --checkpoint ...
```

### ONNX Opset Version

You can specify the ONNX opset version (default is 18):

```bash
python scripts/export_onnx.py --opset 17 ...
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--env` | str | `puffer_drive` | The environment name to load configuration for. |
| `--checkpoint` | str | (required/default example path) | Path to the PyTorch `.pt` checkpoint file. |
| `--output` | str | `None` (derived from checkpoint) | Path where the `.onnx` file will be saved. |
| `--opset` | int | `18` | ONNX opset version to use for export. |

## Verification

The script automatically verifies the exported ONNX model by running a forward pass on both the PyTorch model and the ONNX model with dummy inputs and comparing the outputs. It checks for:
- Logits
- Value
- LSTM hidden states (if applicable)

If verification passes, it will print match confirmations. If there are mismatches, it will raise an error or print a mismatch warning.

# Exporting Model Weights to .bin

You can also export the model weights to a binary format (`.bin`) which can be loaded by the C backend of PufferDrive. This is done using `scripts/export_model_bin.py`.

## Usage

```bash
python scripts/export_model_bin.py --checkpoint path/to/your/checkpoint.pt
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--env` | str | `puffer_drive` | The environment name to load configuration for. |
| `--checkpoint` | str | (required) | Path to the PyTorch `.pt` checkpoint file. |
| `--output` | str | `pufferlib/resources/drive/model_puffer_drive_000100.bin` | Output path for the binary weights file. |

This script flattens all model parameters into a single contiguous binary file.
