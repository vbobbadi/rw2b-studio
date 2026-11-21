"""Fine-tune / evaluate HMR2.0 on the synthetic occlusion data.

Ported from notebook 05's finetune_lib.py. With default arguments, train() and evaluate() do exactly what
notebook 05 did (same data split, augmentation, losses, schedule), so seeds 1 and 2 are comparable with
the existing seed-0 results in results/ep4_b4_s0. Additions (all opt-in):
  - `transform` on SynthDataset: image + label transforms (E2 quality, E3 framing; see degrade.py)
  - `selfvis_joint` passed through when the labels have it (self-occlusion split)
  - train(..., trainable=callable) for LoRA experts; train(..., max_steps=...) to equalise budgets
  - evaluate(..., return_preds=True) keeps rotations/betas/cam for fusion
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .rotations import aa_to_rotmat_t as aa_to_rotmat
from .smpl import EVAL_JOINTS, smpl_joints_torch

FOCAL = 5000.0
IMG = 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def SMPLJoints(smpl_npz):  # noqa: N802  (keeps notebook 05's name)
    return smpl_joints_torch(smpl_npz)


# ---------------------------------------------------------------- data
class SynthDataset(torch.utils.data.Dataset):
    """One condition folder (images/ + labels.npz) from notebook 04.

    transform: optional fn(img_uint8 (H,W,3), labels: dict, rng) -> (img_uint8, label_updates dict).
               Applied before photometric augmentation. With deterministic=True the rng is seeded by
               (transform_seed, sample index), so a degraded TEST set is identical every time it is loaded.
    """

    def __init__(self, cond_dir, indices=None, augment=False, only_ok=True, transform=None,
                 transform_seed=0, deterministic=True):
        self.dir = cond_dir
        L = dict(np.load(os.path.join(cond_dir, "labels.npz")))
        keep = np.ones(len(L["names"]), bool)
        if only_ok and "occ_ok" in L:
            keep &= L["occ_ok"]
        idx = np.nonzero(keep)[0]
        if indices is not None:
            idx = np.intersect1d(idx, indices)
        self.idx = idx
        self.L = L
        self.augment = augment
        self.transform = transform
        self.transform_seed = transform_seed
        self.deterministic = deterministic

    def __len__(self):
        return len(self.idx)

    def _raw(self, i):
        from PIL import Image
        return np.asarray(Image.open(os.path.join(self.dir, "images", f"{self.L['names'][i]}.jpg")).convert("RGB"))

    def _augment(self, x):
        g = torch.Generator().manual_seed(int(torch.randint(0, 2**31, (1,))))
        b = 1 + (torch.rand(1, generator=g) - 0.5) * 0.4
        c = 1 + (torch.rand(1, generator=g) - 0.5) * 0.4
        s = 1 + (torch.rand(1, generator=g) - 0.5) * 0.6
        gray = x.mean(0, keepdim=True)
        x = (x - gray) * s + gray
        x = (x - x.mean()) * c + x.mean()
        x = x * b
        if torch.rand(1, generator=g) < 0.3:
            k = 3 if torch.rand(1, generator=g) < 0.5 else 5
            x = F.avg_pool2d(x[None], k, 1, k // 2, count_include_pad=False)[0]
        x = x + torch.randn(x.shape, generator=g) * float(torch.rand(1, generator=g)) * 0.03
        return x.clamp(0, 1)

    def item_labels(self, i):
        L = self.L
        d = {"joints2d": L["joints2d"][i].astype(np.float32), "vis": L["vis"][i].astype(bool),
             "occfrac": float(L["occfrac"][i])}
        if "selfvis_joint" in L:
            d["selfvis"] = L["selfvis_joint"][i].astype(bool)
        return d

    def __getitem__(self, k):
        i = int(self.idx[k])
        L = self.L
        img = self._raw(i)
        lab = self.item_labels(i)
        meta = {"cond_quality": 0.0, "cond_framing": 0.0}
        if self.transform is not None:
            rng = (np.random.default_rng([self.transform_seed, i]) if self.deterministic
                   else np.random.default_rng())
            img, upd = self.transform(img, lab, rng)
            lab.update({k_: v for k_, v in upd.items() if k_ in lab})
            meta.update({k_: v for k_, v in upd.items() if k_.startswith("cond_")})
        x = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True)).permute(2, 0, 1).float() / 255
        if self.augment:  # photometric only: geometry stays tied to the labels
            x = self._augment(x)
        out = {
            "img": (x - MEAN) / STD,
            "global_orient": aa_to_rotmat(torch.tensor(L["global_orient"][i], dtype=torch.float32)),
            "body_pose": aa_to_rotmat(torch.tensor(L["body_pose"][i], dtype=torch.float32)),
            "betas": torch.tensor(L["betas"][i], dtype=torch.float32),
            "joints3d": torch.tensor(L["joints3d"][i], dtype=torch.float32),
            "joints2d": torch.tensor(lab["joints2d"] / IMG - 0.5, dtype=torch.float32),
            "vis": torch.tensor(lab["vis"], dtype=torch.bool),
            "occfrac": torch.tensor(float(lab["occfrac"])),
            "index": torch.tensor(i),
            "cond_quality": torch.tensor(float(meta["cond_quality"])),
            "cond_framing": torch.tensor(float(meta["cond_framing"])),
        }
        if "selfvis" in lab:
            out["selfvis"] = torch.tensor(lab["selfvis"], dtype=torch.bool)
        return out


# ---------------------------------------------------------------- model glue
def load_model():
    """Released HMR2.0 with the renderer/discriminator dropped (as in notebook 05)."""
    from hmr2.models import load_hmr2
    m, _ = load_hmr2()
    for attr in ("renderer", "mesh_renderer", "discriminator"):
        if hasattr(m, attr):
            setattr(m, attr, None)
    return m


def predict(model, img):
    """Backbone + head. img (B,3,256,256) normalised. Returns rotmats, betas, cam (s,tx,ty), cam_t."""
    feats = model.backbone(img[:, :, :, 32:-32])
    p, cam, _ = model.smpl_head(feats)
    B = img.shape[0]
    cam_t = torch.stack([cam[:, 1], cam[:, 2], 2 * FOCAL / (IMG * cam[:, 0] + 1e-9)], dim=-1)
    return {"global_orient": p["global_orient"].reshape(B, 3, 3), "body_pose": p["body_pose"].reshape(B, 23, 3, 3),
            "betas": p["betas"].reshape(B, 10), "cam": cam, "cam_t": cam_t}


def project(joints, cam):
    """Near-orthographic projection used for the 2D loss (see notebook 05 for the derivation)."""
    s, t = cam[:, :1, None], cam[:, None, 1:3]
    return s * (joints[..., :2] + t) / 2


def set_trainable(model, n_blocks):
    """Freeze everything, then unfreeze the SMPL head and the last n ViT blocks (+ final norm)."""
    model.requires_grad_(False)
    model.smpl_head.requires_grad_(True)
    if n_blocks > 0:
        for blk in model.backbone.blocks[-n_blocks:]:
            blk.requires_grad_(True)
        model.backbone.last_norm.requires_grad_(True)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ft] trainable parameters: {n / 1e6:.1f}M")
    return [p for p in model.parameters() if p.requires_grad]


def trainable_state(model):
    return {n: p.detach().half().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    _, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected, unexpected
    print(f"[ft] loaded {len(sd)} tensors from {os.path.basename(path)}")


# ---------------------------------------------------------------- losses
W = {"kp3d": 0.05, "kp2d": 0.01, "orient": 0.001, "pose": 0.001, "betas": 0.0005}


def compute_loss(pred, batch, fk, reduce=True):
    j_pred = fk(pred["global_orient"].float(), pred["body_pose"].float(), pred["betas"].float())
    j_gt = batch["joints3d"]
    rel_pred = j_pred - j_pred[:, :1]
    rel_gt = j_gt - j_gt[:, :1]
    l3d = (rel_pred - rel_gt).abs().sum((1, 2))
    l2d = (project(j_pred, pred["cam"].float()) - batch["joints2d"]).abs().sum((1, 2))
    lor = ((pred["global_orient"].float() - batch["global_orient"]) ** 2).sum((1, 2))
    lpo = ((pred["body_pose"].float() - batch["body_pose"]) ** 2).sum((1, 2, 3))
    lbe = ((pred["betas"].float() - batch["betas"]) ** 2).sum(1)
    parts = {"kp3d": l3d, "kp2d": l2d, "orient": lor, "pose": lpo, "betas": lbe}
    per_sample = sum(W[k] * v for k, v in parts.items())
    if not reduce:
        return per_sample
    return per_sample.mean(), {k: float(v.detach().mean()) for k, v in parts.items()}


# ---------------------------------------------------------------- metrics
def procrustes(X, Y):
    mx, my = X.mean(1, keepdim=True), Y.mean(1, keepdim=True)
    X0, Y0 = X - mx, Y - my
    U, S, Vt = torch.linalg.svd(X0.transpose(1, 2) @ Y0)
    d = torch.sign(torch.det(U @ Vt))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    Rm = U @ D @ Vt
    scale = (S * torch.diagonal(D, dim1=1, dim2=2)).sum(1) / (X0 ** 2).sum((1, 2)).clamp(min=1e-9)
    return scale[:, None, None] * X0 @ Rm + my


def joint_errors(j_pred, j_gt):
    P, G = j_pred[:, EVAL_JOINTS], j_gt[:, EVAL_JOINTS]
    P = P - P[:, :1]
    G = G - G[:, :1]
    e = (P - G).norm(dim=-1) * 1000
    pa = (procrustes(P, G) - G).norm(dim=-1) * 1000
    return e, pa


# ---------------------------------------------------------------- training
def _loader(ds, batch_size, shuffle, workers, device, drop_last=False, sampler=None):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None,
                                       sampler=sampler, num_workers=workers, drop_last=drop_last,
                                       pin_memory=device == "cuda")


def split_train_val(train_dir, seed=0, val_frac=0.05):
    """Notebook 05's split: 5% of the training folder held out for model selection."""
    full = SynthDataset(train_dir)
    perm = np.random.default_rng(seed).permutation(full.idx)
    n_val = max(1, int(len(perm) * val_frac))
    return perm[n_val:], perm[:n_val]


def train(model, train_dir, val_dir, fk, out_path, epochs=4, batch_size=16, lr=1e-5, n_blocks=4,
          val_frac=0.05, workers=2, device="cuda", seed=0, log_every=50, amp=True,
          train_ds=None, val_ds=None, trainable=None, save_fn=None, max_steps=None):
    """Default call == notebook 05. For experts pass train_ds/val_ds (any datasets with the same keys),
    trainable=fn(model)->params (e.g. LoRA adapter params) and save_fn=fn(model, path)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if train_ds is None:
        tr_idx, val_idx = split_train_val(train_dir, seed, val_frac)
        train_ds = SynthDataset(train_dir, indices=tr_idx, augment=True)
        val_ds = SynthDataset(val_dir, indices=val_idx)
    dl = _loader(train_ds, batch_size, True, workers, device, drop_last=True)
    params = set_trainable(model, n_blocks) if trainable is None else trainable(model)
    model.to(device).eval()      # eval mode on purpose: ViT-H drop-path stays off, frozen features fixed
    fk = fk.to(device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = max_steps or epochs * len(dl)
    epochs_run = math.ceil(steps / len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(enabled=amp and device == "cuda")
    save_fn = save_fn or (lambda m, p: torch.save(trainable_state(m), p))
    best, history, step, t0 = float("inf"), [], 0, time.time()
    print(f"[ft] train {len(train_ds)} | val {len(val_ds)} | {len(dl)} steps/epoch | {steps} steps")
    for ep in range(epochs_run):
        for batch in dl:
            if step >= steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
                pred = predict(model, batch["img"])
            loss, parts = compute_loss(pred, batch, fk)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % log_every == 0:
                el = time.time() - t0
                print(f"[ft] ep {ep + 1} step {step}/{steps} loss {loss.item():.4f} kp3d {parts['kp3d']:.3f} | "
                      f"{el / step:.2f}s/step ~{el / step * (steps - step) / 60:.0f} min left", flush=True)
        r = evaluate(model, val_ds, fk, device=device, batch_size=batch_size * 2, quiet=True, workers=workers)
        v = float(r["mpjpe"].mean())
        history.append({"epoch": ep + 1, "step": step, "val_mpjpe": v, "val_pa": float(r["pa_mpjpe"].mean())})
        print(f"[ft] epoch {ep + 1}: val MPJPE {v:.1f} mm | PA {history[-1]['val_pa']:.1f} mm", flush=True)
        if v < best:
            best = v
            save_fn(model, out_path)
            print(f"[ft] saved best -> {out_path}")
    return history


@torch.no_grad()
def evaluate(model, dataset, fk, device="cuda", batch_size=32, workers=2, quiet=False, amp=True,
             return_preds=False, predict_fn=None, predict_fn_batch=None):
    """Per-sample metrics (mm). return_preds adds rotmats/betas/cam for fusion experiments.
    predict_fn(img) can replace predict(model, img) (e.g. a DGS model); predict_fn_batch(batch) gets the
    whole batch (e.g. oracle routing that reads the true condition labels)."""
    if model is not None:
        model.to(device).eval()
    fk = fk.to(device)
    pf = predict_fn or (lambda img: predict(model, img))
    dl = _loader(dataset, batch_size, False, workers, device)
    keys = ["index", "mpjpe", "pa_mpjpe", "err_joint", "pa_joint", "vis", "occfrac"]
    out = {k: [] for k in keys}
    extra = {k: [] for k in ("selfvis", "global_orient", "body_pose", "betas", "cam")}
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            pred = predict_fn_batch(batch) if predict_fn_batch else pf(batch["img"])
        jp = fk(pred["global_orient"].float(), pred["body_pose"].float(), pred["betas"].float())
        e, pa = joint_errors(jp, batch["joints3d"])
        out["index"].append(batch["index"].cpu()); out["mpjpe"].append(e.mean(1).cpu())
        out["pa_mpjpe"].append(pa.mean(1).cpu()); out["err_joint"].append(e.cpu()); out["pa_joint"].append(pa.cpu())
        out["vis"].append(batch["vis"][:, EVAL_JOINTS].cpu()); out["occfrac"].append(batch["occfrac"].cpu())
        if "selfvis" in batch:
            extra["selfvis"].append(batch["selfvis"][:, EVAL_JOINTS].cpu())
        if return_preds:
            for k in ("global_orient", "body_pose", "betas", "cam"):
                extra[k].append(pred[k].float().cpu())
    r = {k: torch.cat(v).numpy() for k, v in out.items()}
    r.update({k: torch.cat(v).numpy() for k, v in extra.items() if v})
    if not quiet:
        print(f"[eval] {len(r['mpjpe'])} samples | MPJPE {r['mpjpe'].mean():.1f} | PA-MPJPE {r['pa_mpjpe'].mean():.1f} mm")
    return r
