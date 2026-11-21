"""LoRA with several named adapters on one frozen backbone (the 'Specialize' step).

    from pose_lib import lora
    lora.inject(model, lora.hmr2_targets(n_blocks=4), names=["E0", "E1", "E2", "E3"], r=16)
    lora.set_active(model, "E1")          # route everything through expert E1
    params = lora.adapter_params(model, "E1")
    lora.save_adapter(model, "E1", "E1.pt"); lora.load_adapter(model, "E1", "E1.pt")

The generalist baseline gets ONE adapter whose rank is the sum of the expert ranks, so its trainable
parameter count equals the four experts together (LoRA size is linear in r): `generalist_rank(16, 4) = 64`.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W x + b + sum over active adapters of (alpha/r) * B(A(x)). Base weights stay frozen."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.scale = {}
        self.active = []                       # list of (name, weight)

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):                          # some code reads .weight directly (e.g. dtype checks)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def add(self, name, r, alpha):
        dev, dt = self.base.weight.device, self.base.weight.dtype
        a = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.A[name] = a
        self.B[name] = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=dt))
        self.scale[name] = alpha / r

    def forward(self, x):
        y = self.base(x)
        for name, w in self.active:
            A, B = self.A[name], self.B[name]
            y = y + (w * self.scale[name]) * ((x.to(A.dtype) @ A.t()) @ B.t()).to(y.dtype)
        return y


def hmr2_targets(n_blocks=4, head=True):
    """Predicate for HMR2.0: every nn.Linear in the last `n_blocks` ViT blocks (qkv, proj, fc1, fc2)
    and, if head=True, every nn.Linear in the SMPL head (transformer decoder + pose/shape/cam outputs)."""
    def pred(name, module, model):
        if not isinstance(module, nn.Linear):
            return False
        if head and name.startswith("smpl_head."):
            return True
        if name.startswith("backbone.blocks."):
            i = int(name.split(".")[2])
            return i >= len(model.backbone.blocks) - n_blocks
        return False
    return pred


def inject(model, predicate, names, r=16, alpha=None):
    """Wrap matching nn.Linear modules and add one adapter per name. Freezes the base model."""
    alpha = alpha or r
    model.requires_grad_(False)
    targets = [(n, m) for n, m in model.named_modules() if predicate(n, m, model)]
    assert targets, "no modules matched the LoRA predicate"
    for name, mod in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = LoRALinear(mod)
        for nm in names:
            wrapped.add(nm, r, alpha)
        setattr(parent, child, wrapped)
    print(f"[lora] {len(targets)} linear layers x {len(names)} adapters (r={r}); "
          f"{adapter_numel(model, names[0]) / 1e6:.2f}M params per adapter")
    return model


def add_adapter(model, name, r, alpha=None):
    alpha = alpha or r
    for m in lora_modules(model):
        m.add(name, r, alpha)


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_active(model, names):
    """names: None (base model only), 'E1', ['E0','E1'] (summed), or {'E0': 0.3, 'E1': 0.7} (weighted).
    Weighted mixing of adapter outputs is 'merging in weight space'; fusion.py fuses outputs instead."""
    if names is None:
        act = []
    elif isinstance(names, str):
        act = [(names, 1.0)]
    elif isinstance(names, dict):
        act = list(names.items())
    else:
        act = [(n, 1.0) for n in names]
    for m in lora_modules(model):
        m.active = act


def adapter_params(model, name, train=True):
    ps = []
    for m in lora_modules(model):
        for p in (m.A[name], m.B[name]):
            p.requires_grad_(train)
            ps.append(p)
    return ps


def trainable_fn(name):
    """For hmr2_ft.train(trainable=...): freeze all, unfreeze only this adapter, route through it."""
    def fn(model):
        model.requires_grad_(False)
        set_active(model, name)
        ps = adapter_params(model, name, train=True)
        print(f"[lora] training adapter {name}: {sum(p.numel() for p in ps) / 1e6:.2f}M params")
        return ps
    return fn


def adapter_numel(model, name):
    return sum(m.A[name].numel() + m.B[name].numel() for m in lora_modules(model))


def adapter_state(model, name):
    sd = {}
    for mod_name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{mod_name}.A"] = m.A[name].detach().half().cpu()
            sd[f"{mod_name}.B"] = m.B[name].detach().half().cpu()
    return sd


def save_adapter(model, name, path):
    torch.save(adapter_state(model, name), path)


def load_adapter(model, name, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    mods = dict(model.named_modules())
    for k, v in sd.items():
        mod_name, _, which = k.rpartition(".")
        m = mods[mod_name]
        tgt = (m.A if which == "A" else m.B)[name]
        assert tgt.shape == v.shape, (k, tgt.shape, v.shape)
        tgt.data.copy_(v.to(tgt.dtype))
    return model


def generalist_rank(expert_rank, n_experts):
    return expert_rank * n_experts
