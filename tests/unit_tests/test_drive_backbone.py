import pytest
import torch
from torch import nn

from pufferlib.ocean.fused_slot_encoder import validate_fused_encoder
from pufferlib.ocean.torch import DriveBackbone


def slot_encoder(in_features, width):
    return nn.Sequential(nn.Linear(in_features, width), nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, width))


@pytest.mark.parametrize("in_features,width", [(16, 256), (32, 128), (32, 64)])
def test_fused_encoder_accepts_the_shapes_at_the_kernel_limits(in_features, width):
    validate_fused_encoder(slot_encoder(in_features, width))


@pytest.mark.parametrize("in_features,width", [(17, 256), (33, 128), (10, 512), (10, 96)])
def test_fused_encoder_rejects_shapes_the_kernels_cannot_run(in_features, width):
    with pytest.raises(ValueError, match="fused_slot_encoder supports"):
        validate_fused_encoder(slot_encoder(in_features, width))


def test_encode_and_pool_masks_padded_objects():
    backbone = object.__new__(DriveBackbone)

    objects = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 3.0], [100.0, 100.0], [200.0, 200.0]],
            [[300.0, 300.0], [400.0, 400.0], [500.0, 500.0], [600.0, 600.0]],
            [[4.0, 0.0], [-3.0, 7.0], [5.0, 1.0], [999.0, 999.0]],
        ]
    )
    valid_counts = torch.tensor([2, 0, 3])
    encoded_inputs = []

    def encoder(x):
        encoded_inputs.append(x)
        return x

    pooled = backbone._encode_and_pool(objects, valid_counts, encoder)

    # Dense encode (static shapes for torch.compile): every row reaches the encoder once.
    assert len(encoded_inputs) == 1
    assert encoded_inputs[0].shape == objects.shape
    # Padded rows never win the max-pool; an all-padded row pools to zeros.
    torch.testing.assert_close(
        pooled,
        torch.tensor(
            [
                [2.0, 10.0],
                [0.0, 0.0],
                [5.0, 7.0],
            ]
        ),
    )
