"""
Sample diverse body poses from AMASS (SMPL+H G format) for the generator.

    python amass_sample.py --amass /content/amass --out /content/drive/MyDrive/CS747_project/data \
        --stride-sec 0.25 --min-diff-deg 10 --test-frac 0.1

- reads every *.npz under --amass (e.g. CMU/, ACCAD/ extracted from the .tar.bz2 downloads)
- takes one frame every --stride-sec seconds, and keeps it only if some joint differs by
  more than --min-diff-deg from the last kept frame of that sequence (drops standing still)
- splits by SEQUENCE (not frame), so test poses come from motions never seen in training
- writes poses_train.npz / poses_test.npz with 'poses' (P,72): root + 23 SMPL joints, axis-angle
  (SMPL+H hand joints are dropped; SMPL's two hand joints are set to zero)
"""
import argparse
import glob
import hashlib
import os

import numpy as np


def rotvec_angle_diff(a, b):
    """Rough per-joint difference between two axis-angle sets (radians), via rotation matrices."""
    def mats(r):
        th = np.linalg.norm(r, axis=-1, keepdims=True)
        k = np.divide(r, th, out=np.zeros_like(r), where=th > 1e-8)
        K = np.zeros(r.shape[:-1] + (3, 3))
        K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
        K = K - np.swapaxes(K, -1, -2)
        th = th[..., None]
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    Ra, Rb = mats(a), mats(b)
    c = (np.einsum("...ij,...ij->...", Ra, Rb) - 1) / 2
    return np.arccos(np.clip(c, -1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride-sec", type=float, default=0.25)
    ap.add_argument("--min-diff-deg", type=float, default=10.0)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--max-test", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.amass, "**", "*.npz"), recursive=True)
                   if not f.endswith("shape.npz"))
    print(f"{len(files)} sequence files")
    out = {"train": ([], []), "test": ([], [])}
    min_diff = np.radians(a.min_diff_deg)
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "poses" not in d:
            continue
        fps = float(d["mocap_framerate"] if "mocap_framerate" in d else d.get("mocap_frame_rate", 120))
        poses = d["poses"][:, :66]                       # root + 21 body joints
        step = max(1, int(round(fps * a.stride_sec)))
        rel = os.path.relpath(f, a.amass)
        split = "test" if int(hashlib.md5(rel.encode()).hexdigest(), 16) % 1000 < a.test_frac * 1000 else "train"
        last = None
        for p in poses[::step]:
            body = p[3:].reshape(21, 3)
            if last is not None and rotvec_angle_diff(body, last).max() < min_diff:
                continue
            last = body
            full = np.zeros(72)
            full[:66] = p
            out[split][0].append(full)
            out[split][1].append(rel)

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for split, cap in (("train", a.max_train), ("test", a.max_test)):
        P, src = np.array(out[split][0]), np.array(out[split][1])
        if len(P) > cap:
            keep = rng.choice(len(P), cap, replace=False)
            P, src = P[keep], src[keep]
        np.savez(os.path.join(a.out, f"poses_{split}.npz"), poses=P.astype(np.float32), src=src)
        print(f"{split}: {len(P)} poses from {len(set(src.tolist()))} sequences")


if __name__ == "__main__":
    main()
