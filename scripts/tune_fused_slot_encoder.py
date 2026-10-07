"""Time fused_slot_encoder kernel configurations on the current GPU and print one KernelConfig for its GPU class.

Usage (needs CUDA + Triton; one process per GPU type):
    python scripts/tune_fused_slot_encoder.py                      # all encoder shapes, 131072 transitions
    python scripts/tune_fused_slot_encoder.py --batch 65536 --quick
Paste the printed KernelConfig into CONFIGS_BY_CAPABILITY in pufferlib/ocean/fused_slot_encoder.py.
"""

import argparse
import dataclasses
import itertools
import json
import time

import torch
import triton
from torch import nn

import pufferlib.ocean.fused_slot_encoder as fse
import pufferlib.pytorch


ENCODER_SHAPES = {"lane": (35, 10, 256), "boundary": (30, 6, 256), "partner": (20, 11, 256), "traffic": (4, 14, 128)}
OBS_PREFIX = 100
TIMING_ITERS = 15
WARMUP_ITERS = 3
POOLED_TOLERANCE = 2.0**-6
GRAD_TOLERANCE = 1e-3
FWD_GRID = {"block_batch": (16, 32, 64), "block_k": (32, 64), "num_warps": (4, 8), "num_stages": (2, 3)}
BWD_GRID = {"block_rows": (32, 64, 128), "block_k": (32, 64), "num_warps": (4, 8), "num_stages": (2, 3)}
POOL_GRID = {"block_rows": (16, 32, 64), "num_warps": (4, 8)}
QUICK_FWD_GRID = {"block_batch": (32,), "block_k": (32, 64), "num_warps": (4,), "num_stages": (2,)}
QUICK_BWD_GRID = {"block_rows": (64,), "block_k": (64,), "num_warps": (8,), "num_stages": (2,)}
QUICK_POOL_GRID = {"block_rows": (32,), "num_warps": (4,)}
LAUNCH_ERRORS = (triton.runtime.errors.OutOfResources, triton.compiler.errors.CompilationError, RuntimeError)
PARTS = ("fwd", "bwd", "pool")


def make_case(slots, features, width, batch, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    encoder = nn.Sequential(
        pufferlib.pytorch.layer_init(nn.Linear(features, width)),
        nn.LayerNorm(width),
        nn.ReLU(),
        pufferlib.pytorch.layer_init(nn.Linear(width, width)),
    ).cuda()
    obs = torch.zeros(batch, OBS_PREFIX + slots * features, device="cuda")
    counts = torch.randint(0, slots + 1, (batch,), device="cuda", generator=generator)
    objects = obs[:, OBS_PREFIX:].view(batch, slots, features)
    valid = torch.arange(slots, device="cuda")[None, :] < counts[:, None]
    objects.copy_(
        (torch.rand(batch, slots, features, device="cuda", generator=generator) * 1.6 - 0.8) * valid[:, :, None]
    )
    grad_pooled = (torch.randn(batch, width, device="cuda", generator=generator) * 0.1).to(torch.bfloat16)
    return encoder, objects, counts.long(), grad_pooled


def time_call(fn):
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(TIMING_ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / TIMING_ITERS * 1e3


def with_part(base, part, params):
    fields = {name: getattr(base, name) for name in base.__dataclass_fields__}
    fields.update({f"{part}_{key}": value for key, value in params.items()})
    return fse.KernelConfig(**fields)


class ShapeCase:
    def __init__(self, name, batch):
        slots, features, width = ENCODER_SHAPES[name]
        self.name = name
        self.encoder, self.objects, self.counts, self.grad_pooled = make_case(slots, features, width, batch, seed=slots)
        fse.CONFIG_OVERRIDE = fse.GB202_CONFIG
        self.reference_fwd = self.forward()
        self.reference_bwd = self.backward(self.reference_fwd)
        fse.CONFIG_OVERRIDE = None

    def forward(self):
        linear_in, norm, _, linear_out = self.encoder
        return fse.fused_slot_encoder(
            self.objects,
            self.counts,
            linear_in.weight,
            linear_in.bias,
            norm.weight,
            norm.bias,
            linear_out.weight,
            linear_out.bias,
            True,
        )

    def backward(self, fwd):
        pooled, tie_count, normed_relu, encoded, w1_padded, b1_values, w2_bf16 = fwd
        _, norm, _, _ = self.encoder
        return fse.fused_slot_encoder_bwd(
            self.objects,
            self.counts,
            w1_padded,
            b1_values,
            norm.weight,
            norm.bias,
            w2_bf16,
            normed_relu,
            encoded,
            pooled,
            tie_count,
            self.grad_pooled,
        )

    def measure(self, part, config):
        """Milliseconds for one part under `config`, or None when its numerics differ from the default config."""
        fse.CONFIG_OVERRIDE = config
        try:
            if part == "fwd":
                pooled = self.forward()[0]
                if (pooled.float() - self.reference_fwd[0].float()).abs().max().item() > POOLED_TOLERANCE:
                    return None
                return time_call(self.forward)
            grads = self.backward(self.reference_fwd)
            for candidate, reference in zip(grads, self.reference_bwd):
                if ((candidate - reference).norm() / reference.norm().clamp_min(1e-12)).item() > GRAD_TOLERANCE:
                    return None
            return time_call(lambda: self.backward(self.reference_fwd))
        finally:
            fse.CONFIG_OVERRIDE = None


def sweep(cases, part, grid, base):
    totals = {}
    per_shape_best = {case.name: (float("inf"), None) for case in cases}
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        config = with_part(base, part, params)
        total = 0.0
        for case in cases:
            try:
                ms = case.measure(part, config)
            except LAUNCH_ERRORS as exc:
                ms = None
                print(f"   {part} {params} on {case.name}: skipped ({str(exc).splitlines()[0][:70]})", flush=True)
            if ms is None:
                total = None
                break
            total += ms
            if ms < per_shape_best[case.name][0]:
                per_shape_best[case.name] = (ms, params)
        if total is not None:
            totals[tuple(params.items())] = total
            print(f"   {part} {params}: {total:7.3f} ms over all shapes", flush=True)
    best_key = min(totals, key=totals.get)
    return dict(best_key), totals[best_key], per_shape_best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default="all", help="comma-separated encoder names or 'all'")
    parser.add_argument("--batch", type=int, default=131072, help="transitions per minibatch")
    parser.add_argument("--quick", action="store_true", help="tiny grids, for checking that the script runs")
    parser.add_argument("--write-config", help="write the recommended KernelConfig as JSON to this path")
    args = parser.parse_args()
    names = list(ENCODER_SHAPES) if args.shapes == "all" else args.shapes.split(",")
    grids = {"fwd": FWD_GRID, "bwd": BWD_GRID, "pool": POOL_GRID}
    if args.quick:
        grids = {"fwd": QUICK_FWD_GRID, "bwd": QUICK_BWD_GRID, "pool": QUICK_POOL_GRID}
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    device_line = (
        f"{properties.name}, capability {capability}, {properties.multi_processor_count} SMs, "
        f"{properties.shared_memory_per_block_optin // 1024} KB shared memory per block, "
        f"torch {torch.__version__}, triton {triton.__version__}"
    )
    print(device_line, flush=True)
    cases = [ShapeCase(name, args.batch) for name in names]
    base = fse.GB202_CONFIG
    recommended = base
    report_lines = []
    for part in PARTS:
        best_params, best_total, per_shape = sweep(cases, part, grids[part], base)
        baseline_total = sum(case.measure(part, base) for case in cases)
        recommended = with_part(recommended, part, best_params)
        report_lines.append(f"{part}: GB202_CONFIG {baseline_total:.3f} ms -> best {best_total:.3f} ms  {best_params}")
        for name, (ms, params) in per_shape.items():
            report_lines.append(f"   {part} best for {name}: {ms:.3f} ms  {params}")
    combined_total = sum(case.measure(part, recommended) for case in cases for part in PARTS)
    baseline_combined = sum(case.measure(part, base) for case in cases for part in PARTS)
    print("\n=== fused_slot_encoder tuning result (paste this block back) ===")
    print(device_line)
    print(f"batch {args.batch}, shapes {','.join(names)}")
    print("\n".join(report_lines))
    print(f"all parts, all shapes: GB202_CONFIG {baseline_combined:.3f} ms -> recommended {combined_total:.3f} ms")
    print(f"RECOMMENDED for capability {capability}: {recommended}")
    print("=== end of tuning result ===")
    if args.write_config:
        with open(args.write_config, "w") as handle:
            json.dump(dataclasses.asdict(recommended), handle)


if __name__ == "__main__":
    main()
