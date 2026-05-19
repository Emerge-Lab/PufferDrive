"""Parameter-efficient finetuning primitives for PufferLib.

Activated by [finetune].enabled = True in the env config (drive.ini) or the
overlay passed via --finetune-config. Three strategies:

  - full   : every parameter trainable; nothing here applies.
  - freeze : params matching [finetune].freeze_regex get requires_grad=False.
  - lora   : nn.Linear submodules whose dotted name matches
             [finetune].lora_target get wrapped with LoRALinear (frozen base
             weight + trainable rank-r adapter B@A).

Ordering inside load_policy:
    policy = build()
    policy.load_state_dict(base_pt)   # base weights present, NO lora keys
    apply_freeze(policy, ...)         # freeze the layers the user pinned
    wrap_lora(policy, ...)            # swap nn.Linear -> LoRALinear in place
    # ... then DDP wrap, torch.compile, optimizer
"""

import math
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a frozen base weight and a
    trainable rank-r adapter.

        forward(x) = F.linear(x, W, b) + (alpha / r) * F.linear(F.linear(x, A), B)

    State-dict layout:
        Saves under {prefix}weight (MERGED: W + scaling * B @ A) and {prefix}bias.
        Does NOT save lora_A / lora_B — by design. Every LoRA finetune restarts
        its adapter from scratch on resume; the merged weight carries forward
        the previous run's learning so progress isn't lost. This keeps saved
        .pt files load-compatible with a vanilla nn.Linear at the same path,
        which is what subprocess evals and downstream finetunes expect.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Copy the base weight + bias as own parameters (NOT a nested
        # submodule) so the state_dict key layout matches a vanilla Linear.
        self.weight = nn.Parameter(base.weight.data.clone())
        self.weight.requires_grad = False
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone())
            self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 0.0

        # Kaiming-uniform init for A (standard LoRA practice).
        # B is zero so at step 0 the adapter contributes nothing — the
        # wrapped model is mathematically identical to the base policy
        # before any gradient step.
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            out = out + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out

    def merged_weight(self) -> torch.Tensor:
        if self.rank > 0:
            return self.weight + self.scaling * (self.lora_B @ self.lora_A)
        return self.weight

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        merged = self.merged_weight()
        destination[prefix + "weight"] = merged if keep_vars else merged.detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


def _resolve_optional_str(value) -> Optional[str]:
    """drive.ini may carry the literal string 'None' for unset values when
    routed through ast.literal_eval; treat that the same as Python None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in ("", "None"):
        return None
    return value


def apply_freeze(policy: nn.Module, regex) -> int:
    """Set requires_grad=False on every named_parameter whose name matches
    `regex` (re.search semantics). Returns the count of frozen tensors.
    No-op if regex is None / empty.
    """
    regex = _resolve_optional_str(regex)
    if not regex:
        return 0
    pattern = re.compile(regex)
    count = 0
    for name, p in policy.named_parameters():
        if pattern.search(name):
            p.requires_grad = False
            count += 1
    return count


def wrap_lora(policy: nn.Module, target_regex, rank: int, alpha: float) -> int:
    """Replace nn.Linear submodules whose dotted module-name matches
    `target_regex` (re.search semantics) with LoRALinear. Returns the count
    of wrapped modules. No-op if target_regex is None / empty or rank <= 0.

    Note: regex matches MODULE NAMES (e.g. 'actor_backbone.backbone.1'), not
    parameter names. Type filtering to nn.Linear is enforced internally, so
    your regex doesn't need to include 'Linear'.
    """
    target_regex = _resolve_optional_str(target_regex)
    if not target_regex or rank <= 0:
        return 0
    pattern = re.compile(target_regex)
    targets = []
    for name, module in policy.named_modules():
        # Skip LoRALinear instances so re-runs are idempotent.
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and pattern.search(name):
            targets.append((name, module))
    for name, module in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = policy.get_submodule(parent_name) if parent_name else policy
        wrapped = LoRALinear(module, rank=rank, alpha=alpha).to(
            device=module.weight.device, dtype=module.weight.dtype
        )
        setattr(parent, child_name, wrapped)
    return len(targets)


def get_lora_params(policy: nn.Module) -> List[nn.Parameter]:
    """Collect every LoRALinear.lora_A / lora_B in the policy. Used to build
    a separate optimizer parameter group with its own LR."""
    params: List[nn.Parameter] = []
    for module in policy.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def build_param_groups(policy: nn.Module, base_lr: float, lora_lr: float) -> list:
    """Build optimizer parameter groups. LoRA adapter params get `lora_lr`;
    every other trainable param gets `base_lr`. Frozen params are excluded.
    Returns a list of dicts suitable for torch.optim.Optimizer.
    """
    lora_params = get_lora_params(policy)
    lora_param_ids = {id(p) for p in lora_params}
    base_params = [
        p for p in policy.parameters()
        if p.requires_grad and id(p) not in lora_param_ids
    ]
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    active_lora = [p for p in lora_params if p.requires_grad]
    if active_lora:
        groups.append({"params": active_lora, "lr": lora_lr})
    return groups


def trainable_summary(policy: nn.Module) -> str:
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    lora_modules = sum(1 for m in policy.modules() if isinstance(m, LoRALinear))
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return (
        f"[finetune] trainable {trainable:,} / {total:,} ({pct:.2f}%) "
        f"| LoRA modules wrapped: {lora_modules}"
    )
