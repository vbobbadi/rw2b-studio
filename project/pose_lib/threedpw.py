"""3DPW (von Marcard et al., ECCV 2018): real outdoor video with IMU-based SMPL ground truth.

Used for RQ1 with real errors in mm instead of proxy metrics. Protocol here:
  * test split (24 sequences), every person, frames with campose_valid, every STRIDE-th frame;
  * crop from the GT joints projected into the image (box enlarged 1.2x), fed to HMR2.0's own
    ViTDetDataset -> identical crops for every model;
  * GT joints from the neutral SMPL with the GT betas (only the neutral model is available to us), in the
    camera frame; errors on the 22 SMPL body joints, root-relative (MPJPE) and Procrustes-aligned (PA).
Numbers are therefore comparable between our models, not to papers that use gendered models and the
H36M-14 joint regressor.

Data license: non-commercial research only, cite the ECCV'18 paper, do not redistribute the raw data.
"""

import glob
import os
import pickle

import numpy as np

from .rotations import aa_to_rotmat
from .smpl import fk

URL = "https://virtualhumans.mpi-inf.mpg.de/3DPW"
FILES = ["readme_and_demo.zip", "sequenceFiles.zip", "imageFiles.zip"]


def load_seq(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def gt_joints(seq, p, fknp, frames):
    """Camera-frame GT joints (T,24,3), root-relative is up to the metric; also returns the joints in
    camera coordinates *with* translation (for projection)."""
    poses = np.asarray(seq["poses"][p], float)[frames]            # (T,72)
    trans = np.asarray(seq["trans"][p], float)[frames]            # (T,3)
    betas = np.asarray(seq["betas"][p], float)[:10]
    cam = np.asarray(seq["cam_poses"], float)[frames]             # (T,4,4) world -> camera
    Rc, tc = cam[:, :3, :3], cam[:, :3, 3]
    R = aa_to_rotmat(poses.reshape(len(frames), 24, 3))           # (T,24,3,3)
    R_cam = R.copy()
    R_cam[:, 0] = Rc @ R[:, 0]
    J = np.repeat(fknp.rest_joints(betas), len(frames), 0)
    j_world = fk(R, J) + trans[:, None]                           # SMPL: posed joints + trans
    j_cam_full = np.einsum("tij,tkj->tki", Rc, j_world) + tc[:, None]
    j_cam_rot = fk(R_cam, J)                                      # same pose, camera orientation, no trans
    return j_cam_rot, j_cam_full, R_cam


def project(j_cam, K):
    x = j_cam @ np.asarray(K, float).T
    return x[..., :2] / np.clip(x[..., 2:3], 1e-6, None)


def bbox_from_2d(uv, img_w, img_h, enlarge=1.2):
    """(T,J,2) -> (T,4) x0,y0,x1,y1, enlarged around the centre, clipped to the image (person boxes)."""
    lo, hi = uv.min(1), uv.max(1)
    c, half = (lo + hi) / 2, (hi - lo) / 2 * enlarge
    half[:, 1] += (hi - lo)[:, 1] * 0.08                          # joints stop at the head/ankle centres
    b = np.concatenate([c - half, c + half], 1)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, img_w - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, img_h - 1)
    return b


def samples(root, split="test", stride=2, fknp=None, img_wh=None):
    """Every (sequence, person, frame) to evaluate, with GT joints and crop boxes.
    root = folder containing sequenceFiles/ and imageFiles/. img_wh(seq) -> (W,H) of that sequence."""
    out = []
    for f in sorted(glob.glob(f"{root}/sequenceFiles/{split}/*.pkl")):
        seq = load_seq(f)
        name = seq["sequence"]
        W, H = img_wh(name)
        K = seq["cam_intrinsics"]
        for p in range(len(seq["poses"])):
            valid = np.asarray(seq["campose_valid"][p]).astype(bool)
            frames = np.flatnonzero(valid)[::stride]
            if len(frames) == 0:
                continue
            j_rot, j_full, _ = gt_joints(seq, p, fknp, frames)
            boxes = bbox_from_2d(project(j_full, K), W, H)
            ok = (boxes[:, 2] - boxes[:, 0] > 16) & (boxes[:, 3] - boxes[:, 1] > 16) & (j_full[:, 0, 2] > 0.1)
            for i in np.flatnonzero(ok):
                out.append({"seq": name, "person": p, "frame": int(frames[i]), "box": boxes[i].astype(np.float32),
                            "gt": j_rot[i].astype(np.float32),
                            "img": f"{root}/imageFiles/{name}/image_{frames[i]:05d}.jpg"})
    return out


def cluster_bootstrap(a, b, groups, n=5000, seed=0):
    """Paired difference mean(a-b) with a bootstrap over groups (sequences), since frames are correlated.
    Returns mean, 95% CI, two-sided p."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    g = np.asarray(groups)
    keys = np.unique(g)
    sums = np.array([d[g == k].sum() for k in keys])
    cnts = np.array([(g == k).sum() for k in keys])
    idx = rng.integers(0, len(keys), (n, len(keys)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    m = d.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return float(m), float(lo), float(hi), float(max(p, 1 / n))
