"""RQ3: does choosing hard training data adaptively beat random choice and hard-example mining at the
same data budget?

The progress report proposes an RL agent that picks the Blender conditions that make the current model
fail most. Here that agent is a multi-armed bandit (multiplicative weights / Hedge): a one-step RL
problem with no state, where each arm is a condition bin (e.g. occluded-body fraction 20-40%, or quality = motion blur
at severity 0.6). Its reward is the current model's loss on a small probe batch from that bin, so bins
the model still gets wrong are chosen more often. This runs entirely on existing data (the D_occ3d pool,
binned by hidden fraction) or on cheap image-space conditions (degrade.quality), so the whole comparison
fits a Colab session. Swapping the pool for live Blender rendering only changes `ArmPool.sample`.

Strategies, all with identical budget (rounds x round_size images) and identical training steps:
    random   uniform over the pool
    hem      hard-example mining: score a random candidate set with the current model, keep the hardest
    bandit   Hedge over arms, reward = probe loss normalised to [0,1] (probing costs forward passes only)
"""

import math

import numpy as np


class ArmPool:
    """arms: list of index arrays into one dataset (or list of datasets, one per arm)."""

    def __init__(self, arms, names):
        self.arms = [np.asarray(a) for a in arms]
        self.names = names
        self.all = np.concatenate(self.arms)

    def sample(self, arm, n, rng):
        a = self.arms[arm]
        return rng.choice(a, size=n, replace=len(a) < n)

    def sample_any(self, n, rng):
        return rng.choice(self.all, size=n, replace=False)


def occfrac_arms(occfrac, edges=(0.05, 0.2, 0.4, 0.6, 1.01)):
    """Bin a pool (e.g. D_occ3d train) by hidden-body fraction."""
    occfrac = np.asarray(occfrac)
    arms, names = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.nonzero((occfrac >= lo) & (occfrac < hi))[0]
        if len(idx):
            arms.append(idx)
            names.append(f"occ {lo:.0%}-{min(hi, 1):.0%}")
    return arms, names


class Hedge:
    """Multiplicative weights over arms. Every arm is probed every round, so all rewards are observed
    (full information) and no importance weighting is needed (that would be EXP3, for when only the
    chosen arm's reward is seen). gamma mixes in uniform exploration so no condition is starved."""

    def __init__(self, k, eta=1.0, gamma=0.1, seed=0):
        self.k, self.eta, self.gamma = k, eta, gamma
        self.logw = np.zeros(k)
        self.rng = np.random.default_rng(seed)
        self.history = []

    def probs(self):
        p = np.exp(self.logw - self.logw.max())
        p /= p.sum()
        return (1 - self.gamma) * p + self.gamma / self.k

    def choose(self, n):
        """Allocate n draws -> counts per arm."""
        return self.rng.multinomial(n, self.probs())

    def update(self, rewards):
        """rewards: dict arm -> reward in [0,1]."""
        for a, r in rewards.items():
            self.logw[a] += self.eta * r
        self.history.append(self.probs().tolist())


def run(strategy, pool, *, rounds, round_size, train_on, loss_of, seed=0, probe=32, hem_factor=4, log=print):
    """Generic loop. The caller supplies the model through two callbacks:
        train_on(indices)  -> trains the current model on these pool indices (fixed number of steps)
        loss_of(indices)   -> per-sample loss (np.array) of the current model on these indices
    Returns the list of chosen indices per round and, for the bandit, the arm probabilities."""
    rng = np.random.default_rng(seed)
    bandit = Hedge(len(pool.arms), seed=seed) if strategy == "bandit" else None
    chosen = []
    for r in range(rounds):
        if strategy == "random":
            idx = pool.sample_any(round_size, rng)
        elif strategy == "hem":
            cand = pool.sample_any(min(len(pool.all), round_size * hem_factor), rng)
            l = loss_of(cand)
            idx = cand[np.argsort(-l)[:round_size]]
        elif strategy == "bandit":
            # probe every arm a little, reward = mean loss scaled to [0,1] across arms
            losses = {a: float(np.mean(loss_of(pool.sample(a, probe, rng)))) for a in range(len(pool.arms))}
            lo, hi = min(losses.values()), max(losses.values())
            rewards = {a: (v - lo) / (hi - lo + 1e-9) for a, v in losses.items()}
            bandit.update(rewards)
            counts = bandit.choose(round_size)
            idx = np.concatenate([pool.sample(a, int(c), rng) for a, c in enumerate(counts) if c > 0])
        else:
            raise ValueError(strategy)
        train_on(idx)
        chosen.append(idx)
        if log:
            msg = f"[sel] {strategy} round {r + 1}/{rounds}: {len(idx)} samples"
            if bandit is not None:
                msg += " | p=" + " ".join(f"{p:.2f}" for p in bandit.probs())
            log(msg)
    return {"chosen": chosen, "bandit_probs": bandit.history if bandit else None,
            "arm_names": pool.names}


class AdapterTrainer:
    """Incremental training of one LoRA adapter for the RQ3 loop (works with hmr2_ft's model glue).

    Each round: add the chosen indices to the training set, then take `steps_per_round` optimiser steps
    sampling from everything chosen so far (replay). Every strategy gets the same steps and images."""

    def __init__(self, model, adapter, pool_ds, fk, lr=1e-4, batch_size=16, steps_per_round=50, device="cuda",
                 amp=True, seed=0, workers=2, score_ds=None):
        import torch
        from . import lora
        self.torch, self.lora = torch, lora
        self.model, self.adapter, self.pool, self.fk = model, adapter, pool_ds, fk.to(device)
        self.bs, self.steps, self.device, self.amp, self.workers = batch_size, steps_per_round, device, amp, workers
        self.params = lora.trainable_fn(adapter)(model)
        model.to(device).eval()
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler(enabled=amp and device == "cuda")
        self.seen = []
        self.g = torch.Generator().manual_seed(seed)
        self.pos = {int(k): i for i, k in enumerate(pool_ds.idx)}   # dataset label index -> position
        self.score = score_ds or pool_ds              # scoring (probe / mining) uses un-augmented images
        assert np.array_equal(self.score.idx, pool_ds.idx)

    def _subset(self, idx, ds=None):
        return self.torch.utils.data.Subset(ds or self.pool, [self.pos[int(i)] for i in idx])

    def train_on(self, idx):
        from .hmr2_ft import compute_loss, predict
        torch = self.torch
        self.seen.extend(int(i) for i in idx)
        sub = self._subset(self.seen)
        sampler = torch.utils.data.RandomSampler(sub, replacement=True, num_samples=self.steps * self.bs,
                                                 generator=self.g)
        dl = torch.utils.data.DataLoader(sub, batch_size=self.bs, sampler=sampler, num_workers=self.workers,
                                         drop_last=True)
        self.lora.set_active(self.model, self.adapter)
        for batch in dl:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp and self.device == "cuda"):
                pred = predict(self.model, batch["img"])
            loss, _ = compute_loss(pred, batch, self.fk)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()

    def loss_of(self, idx):
        from .hmr2_ft import compute_loss, predict
        torch = self.torch
        dl = torch.utils.data.DataLoader(self._subset(idx, self.score), batch_size=self.bs * 2, shuffle=False,
                                         num_workers=self.workers)
        self.lora.set_active(self.model, self.adapter)
        out = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=self.amp and self.device == "cuda"):
                    pred = predict(self.model, batch["img"])
                out.append(compute_loss(pred, batch, self.fk, reduce=False).float().cpu())
        return torch.cat(out).numpy()
