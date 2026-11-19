"""The fused slot encoder matches the eager bf16 _encode_and_pool and lies no further from the fp32 truth than it."""

import pytest
import torch
from torch import nn

import opengigaflow.pytorch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused slot encoder needs CUDA")

# the last two sit on the kernels' limits: 16 features at width 256, the wide input tile at width 128
ENCODER_SHAPES = [(35, 10, 256), (30, 6, 256), (20, 11, 256), (4, 14, 128), (8, 16, 256), (8, 32, 128)]
BATCH = 1024
BF16_ULP_AT_FOUR = 2.0**-6


def make_encoder(in_features, width):
    layers = [
        opengigaflow.pytorch.layer_init(nn.Linear(in_features, width)),
        nn.LayerNorm(width),
        nn.ReLU(),
        opengigaflow.pytorch.layer_init(nn.Linear(width, width)),
    ]
    return nn.Sequential(*layers).cuda()


def reference_encode_and_pool(objects, valid_counts, encoder):
    valid_mask = torch.arange(objects.shape[1], device=objects.device) < valid_counts.unsqueeze(1)
    encoded = encoder(objects)
    masked = encoded.masked_fill(~valid_mask.unsqueeze(2), torch.finfo(encoded.dtype).min)
    pooled = masked.max(dim=1).values
    return torch.where(valid_counts.unsqueeze(1) == 0, pooled.new_zeros(()), pooled)


def make_inputs(slots, features, width, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    obs_width = 100 + slots * features
    obs = torch.zeros(BATCH, obs_width, device="cuda")
    counts = torch.randint(0, slots + 1, (BATCH,), device="cuda", generator=generator)
    counts[: BATCH // 8] = 0
    counts[BATCH // 8 : BATCH // 4] = slots
    objects = obs[:, 100:].view(BATCH, slots, features)
    data = torch.rand(BATCH, slots, features, device="cuda", generator=generator) * 1.6 - 0.8
    valid = torch.arange(slots, device="cuda")[None, :] < counts[:, None]
    objects.copy_(data * valid[:, :, None])
    duplicated = counts >= 2
    objects[duplicated, 1] = objects[duplicated, 0]
    grad_pooled = (torch.randn(BATCH, width, device="cuda", generator=generator) * 0.1).to(torch.bfloat16)
    return objects, counts.long(), grad_pooled


def run(encoder, objects, counts, grad_pooled, path):
    from opengigaflow.ocean.fused_slot_encoder import fused_encode_and_pool

    for parameter in encoder.parameters():
        parameter.grad = None
    if path == "fp32":
        pooled = reference_encode_and_pool(objects, counts, encoder)
        pooled.backward(grad_pooled.float())
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if path == "eager":
                pooled = reference_encode_and_pool(objects, counts, encoder)
            else:
                pooled = fused_encode_and_pool(objects, counts, encoder)
        pooled.backward(grad_pooled)
    grads = {name: parameter.grad.detach().clone() for name, parameter in encoder.named_parameters()}
    return pooled.detach().float(), grads


def relative_error(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("slots,features,width", ENCODER_SHAPES)
def test_fused_matches_autocast_reference(slots, features, width):
    torch.manual_seed(0)
    encoder = make_encoder(features, width)
    objects, counts, grad_pooled = make_inputs(slots, features, width, seed=slots)
    truth_pooled, truth_grads = run(encoder, objects, counts, grad_pooled, "fp32")
    eager_pooled, eager_grads = run(encoder, objects, counts, grad_pooled, "eager")
    fused_pooled, fused_grads = run(encoder, objects, counts, grad_pooled, "fused")

    assert fused_pooled.dtype == torch.float32 and fused_pooled.shape == eager_pooled.shape
    pooled_diff = (fused_pooled - eager_pooled).abs()
    assert pooled_diff.max().item() <= BF16_ULP_AT_FOUR
    assert (pooled_diff > 0).float().mean().item() < 5e-3
    assert (fused_pooled - truth_pooled).abs().max().item() <= 1.05 * (
        eager_pooled - truth_pooled
    ).abs().max().item() + 1e-3
    for name in eager_grads:
        assert relative_error(fused_grads[name], eager_grads[name]) < 2e-2, name
        fused_truth_error = relative_error(fused_grads[name], truth_grads[name])
        eager_truth_error = relative_error(eager_grads[name], truth_grads[name])
        assert fused_truth_error <= 1.05 * eager_truth_error + 2e-3, name


@pytest.mark.parametrize("slots,features,width", ENCODER_SHAPES[:1])
def test_fused_inference_matches_training_forward(slots, features, width):
    from opengigaflow.ocean.fused_slot_encoder import fused_encode_and_pool

    torch.manual_seed(0)
    encoder = make_encoder(features, width)
    objects, counts, _ = make_inputs(slots, features, width, seed=7)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pooled_train = fused_encode_and_pool(objects, counts, encoder)
        with torch.no_grad():
            pooled_eval = fused_encode_and_pool(objects, counts, encoder)
    assert torch.equal(pooled_train.detach(), pooled_eval)
    assert pooled_eval.dtype == torch.bfloat16


@pytest.mark.parametrize("slots,features,width", [ENCODER_SHAPES[0], ENCODER_SHAPES[3], (3, 5, 64)])
def test_second_layer_weight_gradient_matches_dense_reference(slots, features, width):
    """The arg-max-only kernel must equal grad_encoded^T @ normed_relu computed densely in float64."""
    from opengigaflow.ocean.cuda_slot_encoder import cuda_slot_encoder_w2_grad

    generator = torch.Generator(device="cuda").manual_seed(slots)
    batch = BATCH + slots  # off the kernel's row-block size
    normed_relu = torch.rand(batch * slots, width, device="cuda", generator=generator).to(torch.bfloat16)
    counts = torch.randint(0, slots + 1, (batch,), device="cuda", generator=generator)
    winner = torch.randint(0, slots, (batch, width), device="cuda", generator=generator)
    grad_pooled = (torch.randn(batch, width, device="cuda", generator=generator) * 0.1).to(torch.bfloat16)
    grad_w2 = cuda_slot_encoder_w2_grad(normed_relu, counts, winner.to(torch.uint8), grad_pooled, slots)

    winner_grad = grad_pooled.double() * (winner < counts[:, None])
    grad_encoded = torch.zeros(batch, slots, width, dtype=torch.float64, device="cuda")
    grad_encoded.scatter_(1, winner[:, None, :], winner_grad[:, None, :])
    reference = grad_encoded.view(batch * slots, width).t() @ normed_relu.double()
    rounded_reference = reference.to(torch.bfloat16).float()
    assert (grad_w2 != rounded_reference).float().mean().item() < 1e-3
    assert (grad_w2 - rounded_reference).abs().max().item() <= 2.0**-7 * reference.abs().max().item()


def test_fused_handles_strided_upstream_gradient():
    """The cat backward hands the encoder a strided slice of the feature gradient."""
    from opengigaflow.ocean.fused_slot_encoder import fused_encode_and_pool

    torch.manual_seed(0)
    slots, features, width = ENCODER_SHAPES[0]
    encoder = make_encoder(features, width)
    head = opengigaflow.pytorch.layer_init(nn.Linear(2 * width, 8)).cuda()
    objects, counts, _ = make_inputs(slots, features, width, seed=11)
    other_features = torch.randn(BATCH, width, device="cuda")
    grads = {}
    for path in ("eager", "fused"):
        for parameter in encoder.parameters():
            parameter.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if path == "eager":
                pooled = reference_encode_and_pool(objects, counts, encoder)
            else:
                pooled = fused_encode_and_pool(objects, counts, encoder)
            output = head(torch.cat([other_features.to(pooled.dtype), pooled], dim=1))
        output.float().square().mean().backward()
        grads[path] = {name: parameter.grad.detach().clone() for name, parameter in encoder.named_parameters()}
    for name in grads["eager"]:
        assert relative_error(grads["fused"][name], grads["eager"][name]) < 2e-2, name
