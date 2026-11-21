"""Fusing expert outputs into one pose per frame (RQ5).

All three methods fuse the same things: per-joint local rotations (24 incl. the root), betas and the
camera. Rotations are averaged in matrix space and projected back to SO(3) (chordal L2 mean), which is
the standard, well-behaved weighted rotation average.

    weighted   one weight per expert per frame (router weights)
    per_joint  router weights, re-weighted per joint by the detector's visibility for that joint:
               E1 (occlusion) gets more say on joints predicted hidden, the others on visible joints
    learned    a small MLP maps [router probs, visibility probs] -> per-joint expert weights; trained on
               a held-out synthetic split with precomputed expert predictions (cheap, CPU is fine)
    top1       hard routing to the highest-weight expert (reference point)
"""

import torch
import torch.nn as nn

from .rotations import gram_schmidt_t, project_to_so3_t


def _stack(preds):
    """list of K expert dicts -> R (B,K,24,3,3), betas (B,K,10), cam (B,K,3)."""
    R = torch.stack([torch.cat([p["global_orient"][:, None], p["body_pose"]], 1) for p in preds], 1)
    betas = torch.stack([p["betas"] for p in preds], 1)
    cam = torch.stack([p["cam"] for p in preds], 1)
    return R.float(), betas.float(), cam.float()


def _out(R, betas, cam):
    s, tx, ty = cam[:, 0], cam[:, 1], cam[:, 2]
    cam_t = torch.stack([tx, ty, 2 * 5000.0 / (256 * s + 1e-9)], -1)
    return {"global_orient": R[:, 0], "body_pose": R[:, 1:], "betas": betas, "cam": cam, "cam_t": cam_t}


def fuse_joint_weights(preds, wj, w_frame=None, project="svd"):
    """wj: (B,K,24) per-joint weights (rotations); w_frame (B,K) for betas/cam (default: mean of wj).
    project: 'svd' (chordal mean, exact) or 'gs' (Gram-Schmidt, differentiable; used by learned fusion)."""
    R, betas, cam = _stack(preds)
    wj = wj.float()
    wj = wj / wj.sum(1, keepdim=True).clamp(min=1e-8)
    M = (wj[..., None, None] * R).sum(1)                        # (B,24,3,3)
    Rf = project_to_so3_t(M) if project == "svd" else gram_schmidt_t(M)
    wf = wj.mean(2) if w_frame is None else w_frame / w_frame.sum(1, keepdim=True).clamp(min=1e-8)
    return _out(Rf, (wf[..., None] * betas).sum(1), (wf[..., None] * cam).sum(1))


def fuse_weighted(preds, w):
    """w: (B,K) router weights."""
    return fuse_joint_weights(preds, w[..., None].expand(-1, -1, 24), w)


def fuse_top1(preds, w):
    hard = torch.zeros_like(w).scatter_(1, w.argmax(1, keepdim=True), 1.0)
    return fuse_weighted(preds, hard)


def per_joint_weights(w, vis, occ_index=1, gain=2.0):
    """w (B,K) router weights, vis (B,24) predicted visibility -> (B,K,24).
    For joint j: occlusion expert weight x (1 + gain*(1-vis_j)); every other expert x (1 + gain*vis_j)."""
    B, K = w.shape
    wj = w[..., None].expand(B, K, 24).clone()
    hid = (1 - vis)[:, None]                                   # (B,1,24)
    boost = 1 + gain * vis[:, None].expand(B, K, 24).clone()
    boost[:, occ_index] = 1 + gain * hid[:, 0]
    return wj * boost


def fuse_per_joint(preds, w, vis, occ_index=1, gain=2.0):
    return fuse_joint_weights(preds, per_joint_weights(w, vis, occ_index, gain), w)


class LearnedFusion(nn.Module):
    """[cond probs (3), vis probs (24), occfrac (1)] -> per-joint weights over K experts (softmax)."""

    def __init__(self, k=4, hidden=128):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(3 + 24 + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 24 * k))

    def forward(self, cond, vis, occfrac):
        x = torch.cat([cond, vis, occfrac[:, None]], 1)
        return torch.softmax(self.net(x).view(-1, 24, self.k), -1).transpose(1, 2)    # (B,K,24)


def train_learned_fusion(cache, fk, epochs=200, lr=1e-3, val_frac=0.2, seed=0, device="cpu"):
    """cache: dict of tensors over N held-out samples:
        R (N,K,24,3,3), betas (N,K,10), cam (N,K,3), cond (N,3), vis (N,24), occfrac (N,), joints3d (N,24,3)
    Trains on root-relative 3D joint error after FK of the fused rotations."""
    from .hmr2_ft import EVAL_JOINTS
    g = torch.Generator().manual_seed(seed)
    N = cache["R"].shape[0]
    perm = torch.randperm(N, generator=g)
    nv = int(N * val_frac)
    va, tr = perm[:nv], perm[nv:]
    net = LearnedFusion(k=cache["R"].shape[1]).to(device)
    fk = fk.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    c = {k: v.to(device) for k, v in cache.items()}

    def err(idx):
        wj = net(c["cond"][idx], c["vis"][idx], c["occfrac"][idx])
        M = (wj[..., None, None] * c["R"][idx]).sum(1)
        Rf = gram_schmidt_t(M)
        wf = wj.mean(2)
        betas = (wf[..., None] * c["betas"][idx]).sum(1)
        j = fk(Rf[:, 0], Rf[:, 1:], betas)
        j = j - j[:, :1]
        gt = c["joints3d"][idx] - c["joints3d"][idx][:, :1]
        return (j - gt)[:, EVAL_JOINTS].norm(dim=-1).mean() * 1000

    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        for b in torch.split(tr[torch.randperm(len(tr), generator=g)], 256):
            loss = err(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(err(va))
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f"[fusion] learned fusion: best held-out MPJPE {best:.1f} mm")
    return net.cpu(), best
