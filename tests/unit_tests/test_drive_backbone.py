import torch

from pufferlib.ocean.torch import DriveBackbone


def test_encode_and_pool_masks_padded_objects():
    backbone = object.__new__(DriveBackbone)
    backbone.mask_padded_features = True

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


def test_encode_and_pool_without_masking_pools_every_row():
    backbone = object.__new__(DriveBackbone)
    backbone.mask_padded_features = False
    objects = torch.tensor([[[1.0, 10.0], [2.0, 3.0], [100.0, 100.0]]])
    pooled = backbone._encode_and_pool(objects, torch.tensor([2]), lambda x: x)
    torch.testing.assert_close(pooled, torch.tensor([[100.0, 100.0]]))
