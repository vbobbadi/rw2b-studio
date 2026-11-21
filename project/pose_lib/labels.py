"""Add the per-joint self-occlusion label to the EXISTING test set without re-rendering.

blender/synth_gen.py --labels-only --selfvis-joints re-runs notebook 04's test generation with the same
seeds, skipping the renders. The random stream is unchanged, so it reproduces the same samples; we check
that sample by sample (joints3d must match) and then write `selfvis_joint` / `selfvis_frac` into each
condition's labels.npz. Images and every existing label stay as they are.
"""

import os
import shutil

import numpy as np


def merge_selfvis(gen_dirs, cond_dirs, atol=1e-4, backup=True):
    """gen_dirs: generator output folders (each with labels.npz, start=...), in shard order.
    cond_dirs: e.g. [.../test/B_clean, .../test/C_paste2d, .../test/D_occ3d]."""
    parts = []
    for g in gen_dirs:
        L = dict(np.load(os.path.join(g, "labels.npz")))
        assert "selfvis_joint" in L, f"{g} was generated without --selfvis-joints"
        parts.append(L)
    gen = {k: np.concatenate([p[k] for p in parts]) for k in ("joints3d", "selfvis_joint", "selfvis_frac")}
    report = {}
    for c in cond_dirs:
        path = os.path.join(c, "labels.npz")
        old = dict(np.load(path))
        n = len(old["names"])
        assert len(gen["joints3d"]) >= n, f"generated {len(gen['joints3d'])} samples, test set has {n}"
        diff = np.abs(gen["joints3d"][:n] - old["joints3d"]).max()
        if diff > atol:
            raise ValueError(f"{c}: regenerated samples differ from the test set (max |dj| = {diff:.4f} m). "
                             "Use the same --n/--start/--seed/--poses as notebook 04.")
        if backup and not os.path.exists(path + ".bak"):
            shutil.copy(path, path + ".bak")
        old["selfvis_joint"] = gen["selfvis_joint"][:n]
        old["selfvis_frac"] = gen["selfvis_frac"][:n]
        np.savez(path, **old)
        sv = old["selfvis_joint"][:, :22]
        report[os.path.basename(c)] = {"samples": n, "joints hidden by the body": float(1 - sv.mean())}
    return report
