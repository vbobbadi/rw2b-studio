"""The 'Diagnose' step: a learned problem detector trained on Blender images, whose labels come free.

Inputs: the same 256x256 person crop the pose model sees.
Outputs: condition probabilities [occlusion, quality, framing], per-joint visibility (24) and the hidden
fraction of the body. Router weights for the experts come from the condition probabilities:
    w(E1) = p_occ, w(E2) = p_quality, w(E3) = p_framing, w(E0) = (1-p_occ)(1-p_quality)(1-p_framing),
then normalised. Several problems at once give several non-zero weights.

Training data is a mixture built from notebook 04's folders (no new rendering):
    B_clean                   -> all conditions 0
    D_occ3d (+ C_paste2d)     -> occlusion 1 (if hidden fraction >= 5%), per-joint visibility labels
    B_clean + quality()       -> quality 1
    B_clean + framing()       -> framing 1, out-of-view joints marked not visible
    D_occ3d + quality()       -> both (so the detector learns combinations)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COND_NAMES = ["occlusion", "quality", "framing"]
EXPERTS = ["E0", "E1", "E2", "E3"]


class Detector(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            import torchvision
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.resnet18(weights=w)
            feat = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == "tiny":                       # for CPU tests only
            net = nn.Sequential(nn.Conv2d(3, 16, 5, 4, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat = 32
        else:
            raise ValueError(backbone)
        self.net = net
        self.cond = nn.Linear(feat, 3)
        self.vis = nn.Linear(feat, 24)
        self.occfrac = nn.Linear(feat, 1)

    def forward(self, img):
        """img: (B,3,256,256) normalised like the pose model input. Uses the central 192 columns too,
        so it judges exactly what HMR2.0 sees."""
        f = self.net(img[:, :, :, 32:-32])
        return {"cond_logits": self.cond(f), "vis_logits": self.vis(f), "occfrac": torch.sigmoid(self.occfrac(f))[:, 0]}


def targets_from_batch(batch, occ_thresh=0.05):
    occ = (batch["occfrac"] >= occ_thresh).float()
    q = (batch["cond_quality"] > 0).float()
    fr = (batch["cond_framing"] > 0).float()
    return torch.stack([occ, q, fr], 1)


def loss_fn(out, batch, w_vis=0.5, w_frac=1.0):
    t = targets_from_batch(batch)
    l_cond = F.binary_cross_entropy_with_logits(out["cond_logits"], t)
    l_vis = F.binary_cross_entropy_with_logits(out["vis_logits"], batch["vis"].float())
    l_frac = F.l1_loss(out["occfrac"], batch["occfrac"].float())
    return l_cond + w_vis * l_vis + w_frac * l_frac, {"cond": l_cond.item(), "vis": l_vis.item(), "frac": l_frac.item()}


def router_weights(cond_probs, floor=0.0):
    """(B,3) probabilities -> (B,4) expert weights for [E0, E1, E2, E3]."""
    p = cond_probs.clamp(0, 1)
    w0 = torch.prod(1 - p, dim=1, keepdim=True)
    w = torch.cat([w0, p], 1) + floor
    return w / w.sum(1, keepdim=True).clamp(min=1e-8)


def oracle_weights(batch, occ_thresh=0.05):
    """Router weights from the true condition labels (upper bound for the detector)."""
    return router_weights(targets_from_batch(batch, occ_thresh))


@torch.no_grad()
def predict(det, img):
    o = det(img)
    return {"cond": torch.sigmoid(o["cond_logits"]), "vis": torch.sigmoid(o["vis_logits"]), "occfrac": o["occfrac"]}


def train(det, train_ds, val_ds, out_path, epochs=3, batch_size=64, lr=3e-4, device="cuda", workers=2, seed=0,
          log_every=50):
    torch.manual_seed(seed)
    det.to(device).train()
    dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                                     drop_last=True)
    opt = torch.optim.AdamW(det.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    best, hist, step = -1.0, [], 0
    for ep in range(epochs):
        det.train()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, parts = loss_fn(det(batch["img"]), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0:
                print(f"[det] ep {ep + 1} step {step}/{steps} loss {loss.item():.3f} {parts}", flush=True)
        m = evaluate(det, val_ds, device=device, batch_size=batch_size, workers=workers)
        hist.append({"epoch": ep + 1, **m})
        print(f"[det] epoch {ep + 1}: {m}", flush=True)
        if m["mean_auc"] > best:
            best = m["mean_auc"]
            torch.save(det.state_dict(), out_path)
    return hist


@torch.no_grad()
def evaluate(det, ds, device="cuda", batch_size=64, workers=2, return_raw=False):
    """AUC per condition, per-joint visibility accuracy, hidden-fraction MAE."""
    det.to(device).eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    P, T, V, VT, OF, OFT = [], [], [], [], [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        o = predict(det, batch["img"])
        P.append(o["cond"].cpu()); T.append(targets_from_batch(batch).cpu())
        V.append(o["vis"].cpu()); VT.append(batch["vis"].cpu())
        OF.append(o["occfrac"].cpu()); OFT.append(batch["occfrac"].cpu())
    P, T, V, VT = torch.cat(P).numpy(), torch.cat(T).numpy(), torch.cat(V).numpy(), torch.cat(VT).numpy()
    OF, OFT = torch.cat(OF).numpy(), torch.cat(OFT).numpy()
    out = {}
    for i, n in enumerate(COND_NAMES):
        out[f"auc_{n}"] = auc(T[:, i], P[:, i])
    aucs = [v for v in out.values() if np.isfinite(v)]
    out["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["vis_acc"] = float(((V > 0.5) == VT).mean())
    out["occfrac_mae"] = float(np.abs(OF - OFT).mean())
    if return_raw:
        out["_raw"] = {"cond": P, "cond_true": T, "vis": V, "vis_true": VT, "occfrac": OF, "occfrac_true": OFT}
    return out


def auc(y, s):
    """ROC AUC without sklearn (rank-based). NaN if only one class present."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class MixtureDataset(torch.utils.data.Dataset):
    """Concatenate datasets with fixed per-source sizes (sampled once, reproducibly)."""

    def __init__(self, sources, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.items = []
        for si, (ds, n) in enumerate(zip(sources, sizes)):
            pick = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
            self.items += [(si, int(k)) for k in pick]
        self.sources = sources

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        si, k = self.items[i]
        b = self.sources[si][k]
        b.pop("selfvis", None)              # not every source has it; keep batches collatable
        return b
