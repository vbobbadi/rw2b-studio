"""The 'Specialize' step: build each expert's data, train adapters, and run the full
detector -> experts -> fusion model on a batch of crops.

Expert   Rows      Training data (all from notebook 04's folders, same number of images each)
E0       1, 16     B_clean
E1       7,8,10,11 D_occ3d (3D-rendered occluders; row 6 is contact, row 9 a tracking problem)
E2       20-22     B_clean + degrade.quality()   (blur, motion blur, low light, low resolution)
E3       12-15     B_clean + degrade.framing()   (cut-off, far, close)
generalist         the union of the four sets above, rank = 4 x expert rank (same trainable params),
                   trained for 4x the steps (same total compute)

Every expert uses the same image count, steps, LR and LoRA rank, so differences come from the data.
"""

import os

import numpy as np
import torch

from . import degrade, fusion, lora
from .detector import predict as det_predict
from .detector import router_weights
from .hmr2_ft import SynthDataset, predict, split_train_val

EXPERT_COND = {"E0": "B_clean", "E1": "D_occ3d", "E2": "B_clean", "E3": "B_clean"}
EXPERT_TF = {"E0": None, "E1": None, "E2": "quality", "E3": "framing"}


def _tf(name):
    if name == "quality":
        return degrade.quality()
    if name == "framing":
        return degrade.framing()
    return None


def expert_datasets(data_root, expert, seed=0, val_frac=0.05, n_train=None):
    """(train_ds, val_ds) for one expert. Train transforms are random per epoch; val is fixed."""
    cond = EXPERT_COND[expert]
    d = os.path.join(data_root, "train", cond)
    tr_idx, va_idx = split_train_val(d, seed, val_frac)
    if n_train is not None:
        tr_idx = np.random.default_rng(seed).choice(tr_idx, size=min(n_train, len(tr_idx)), replace=False)
    tf = _tf(EXPERT_TF[expert])
    tr = SynthDataset(d, indices=tr_idx, augment=True, transform=tf, deterministic=False)
    va = SynthDataset(d, indices=va_idx, transform=tf, transform_seed=10_000 + seed, deterministic=True)
    return tr, va


def test_sets(data_root):
    """The cross-evaluation conditions (fixed, deterministic): one per expert plus the two existing
    occlusion variants. Degraded test sets are generated on the fly from test/B_clean with fixed seeds."""
    t = os.path.join(data_root, "test")
    return {
        "clean": SynthDataset(f"{t}/B_clean"),
        "occ3d": SynthDataset(f"{t}/D_occ3d"),
        "paste2d": SynthDataset(f"{t}/C_paste2d"),
        "quality": SynthDataset(f"{t}/B_clean", transform=degrade.quality(), transform_seed=777),
        "framing": SynthDataset(f"{t}/B_clean", transform=degrade.framing(), transform_seed=778),
        "occ3d+quality": SynthDataset(f"{t}/D_occ3d", transform=degrade.quality(), transform_seed=779),
    }


class ConcatSame(torch.utils.data.ConcatDataset):
    """ConcatDataset that drops keys not shared by all parts (e.g. selfvis) so batches collate."""

    def __getitem__(self, i):
        b = super().__getitem__(i)
        b.pop("selfvis", None)
        return b


class DGS:
    """Diagnose -> Specialize -> fuse, on a batch of 256x256 crops.

    model: HMR2.0 with LoRA adapters E0..E3 injected and loaded.
    det:   trained Detector, or None to use `weights` given per call (e.g. oracle labels).
    mode:  'weighted' | 'per_joint' | 'learned' | 'top1'
    """

    def __init__(self, model, det=None, mode="weighted", experts=("E0", "E1", "E2", "E3"), learned=None,
                 gain=2.0):
        self.model, self.det, self.mode, self.experts = model, det, mode, list(experts)
        self.learned, self.gain = learned, gain
        self.last = {}

    _cache = {}            # shared by every DGS instance: the 3 fusion variants reuse the 4 expert passes

    @staticmethod
    def _key(img):
        return (tuple(img.shape), round(float(img[:, :, ::17, ::17].float().sum()), 3),
                round(float(img[:, :, 5::23, 3::29].float().sum()), 3))

    @torch.no_grad()
    def expert_preds(self, img):
        k = (self._key(img), tuple(self.experts))
        hit = DGS._cache.get(k)
        if hit is not None:
            return [{kk: v.to(img.device) for kk, v in p.items()} for p in hit]
        out = self._expert_preds(img)
        if len(DGS._cache) > 4000:
            DGS._cache.clear()
        DGS._cache[k] = [{kk: v.cpu() for kk, v in p.items()} for p in out]
        return out

    @torch.no_grad()
    def _expert_preds(self, img):
        out = []
        for e in self.experts:
            lora.set_active(self.model, e)
            out.append({k: v.float() for k, v in predict(self.model, img).items()})
        lora.set_active(self.model, None)
        return out

    def oracle(self, batch):
        """Route with the TRUE condition labels (upper bound for the detector): use with
        hmr2_ft.evaluate(..., predict_fn_batch=dgs.oracle)."""
        from .detector import oracle_weights
        w = oracle_weights(batch)
        vis = batch["vis"].float()
        saved, self.det = self.det, None
        try:
            if self.mode == "per_joint":
                preds = self.expert_preds(batch["img"])
                return fusion.fuse_per_joint(preds, w, vis, occ_index=self.experts.index("E1"), gain=self.gain)
            return self(batch["img"], weights=w)
        finally:
            self.det = saved

    @torch.no_grad()
    def __call__(self, img, weights=None):
        preds = self.expert_preds(img)
        if self.det is not None:
            d = det_predict(self.det, img.float())
            w = router_weights(d["cond"])
            vis, cond, occ = d["vis"], d["cond"], d["occfrac"]
        else:
            w = weights
            vis = torch.ones(img.shape[0], 24, device=img.device)
            cond = torch.zeros(img.shape[0], 3, device=img.device)
            occ = torch.zeros(img.shape[0], device=img.device)
        self.last = {"weights": w, "vis": vis, "cond": cond, "occfrac": occ}
        if self.mode == "weighted":
            return fusion.fuse_weighted(preds, w)
        if self.mode == "top1":
            return fusion.fuse_top1(preds, w)
        if self.mode == "per_joint":
            return fusion.fuse_per_joint(preds, w, vis, occ_index=self.experts.index("E1"), gain=self.gain)
        if self.mode == "learned":
            wj = self.learned.to(img.device)(cond, vis, occ)
            return fusion.fuse_joint_weights(preds, wj, project="gs")
        raise ValueError(self.mode)


@torch.no_grad()
def cache_for_fusion(model, det, dataset, experts=("E0", "E1", "E2", "E3"), device="cuda", batch_size=32,
                     workers=2):
    """Precompute every expert's prediction + detector outputs on a dataset (for learned fusion)."""
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    dgs = DGS(model, det, experts=experts)
    acc = {k: [] for k in ("R", "betas", "cam", "cond", "vis", "occfrac", "joints3d")}
    for batch in dl:
        img = batch["img"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            preds = dgs.expert_preds(img)
        R, b, c = fusion._stack(preds)
        d = det_predict(det, img.float())
        acc["R"].append(R.cpu()); acc["betas"].append(b.cpu()); acc["cam"].append(c.cpu())
        acc["cond"].append(d["cond"].cpu()); acc["vis"].append(d["vis"].cpu())
        acc["occfrac"].append(d["occfrac"].cpu()); acc["joints3d"].append(batch["joints3d"])
    return {k: torch.cat(v) for k, v in acc.items()}
