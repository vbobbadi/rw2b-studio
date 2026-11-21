"""Metrics.

Synthetic data (exact ground truth): MPJPE, PA-MPJPE per joint, split by visible / hidden-by-object /
hidden-by-self joints and by hidden-body fraction.

Real video (no ground truth): proxy metrics. Low jitter means smooth, not accurate, so these are reported
next to a visual score from two raters (see `rating_template`) and cross-model agreement.
"""

import numpy as np

from .smpl import BONES, EVAL_JOINTS, FOOT_JOINTS, LEFT_RIGHT


# ================================================================ synthetic
def procrustes(X, Y):
    """Similarity-align X to Y. (B,J,3) each -> aligned X."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    mx, my = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    X0, Y0 = X - mx, Y - my
    U, S, Vt = np.linalg.svd(np.swapaxes(X0, 1, 2) @ Y0)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones((len(X), 3))
    D[:, 2] = d
    R = (U * D[:, None]) @ Vt
    scale = (S * D).sum(1) / np.maximum((X0 ** 2).sum((1, 2)), 1e-12)
    return scale[:, None, None] * X0 @ R + my


def joint_errors(pred, gt, joints=EVAL_JOINTS):
    """Per-joint errors in mm (root-relative): returns (mpjpe (B,J), pa_mpjpe (B,J))."""
    P = np.asarray(pred, np.float64)[:, joints]
    G = np.asarray(gt, np.float64)[:, joints]
    P = P - P[:, :1]
    G = G - G[:, :1]
    e = np.linalg.norm(P - G, axis=-1) * 1000
    pa = np.linalg.norm(procrustes(P, G) - G, axis=-1) * 1000
    return e, pa


OCC_BINS = [(0.0, 0.0), (1e-4, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]


def summarize(r, min_n=10):
    """One (model, test set) result dict -> summary numbers. Same keys as notebook 05's summary.csv,
    plus a self-occlusion split when the test set has `selfvis_joint` labels."""
    s = {"n": len(r["mpjpe"]), "MPJPE": float(np.mean(r["mpjpe"])), "PA-MPJPE": float(np.mean(r["pa_mpjpe"]))}
    vis = np.asarray(r["vis"], bool)
    pa_j = np.asarray(r["pa_joint"])
    if (~vis).any():
        s["PA visible joints"] = float(pa_j[vis].mean())
        s["PA hidden joints"] = float(pa_j[~vis].mean())
    if "selfvis" in r:                       # joints visible to the camera ignoring occluders
        sv = np.asarray(r["selfvis"], bool)
        groups = {"PA joints: visible": vis & sv, "PA joints: hidden by object": ~vis & sv,
                  "PA joints: hidden by body": sv == 0}
        for k, m in groups.items():
            if m.sum() >= min_n:
                s[k] = float(pa_j[m].mean())
    occ = np.asarray(r["occfrac"])
    for lo, hi in OCC_BINS:
        m = (occ == 0) if hi == 0 else (occ >= lo) & (occ <= hi)
        if m.sum() >= min_n:
            key = "PA occ 0%" if hi == 0 else f"PA occ {lo:.0%}-{min(hi, 1):.0%}"
            s[key] = float(np.asarray(r["pa_mpjpe"])[m].mean())
    return s


# ================================================================ real video (proxy)
def _rel(j):
    j = np.asarray(j, np.float64)
    return j - j[:, :1]


def _contiguous_runs(frame_idx):
    """Split indices into runs of consecutive frames (finite differences only inside a run)."""
    fi = np.asarray(frame_idx)
    cuts = np.nonzero(np.diff(fi) != 1)[0] + 1
    return np.split(np.arange(len(fi)), cuts)


def joint_jitter(seq, joints=EVAL_JOINTS, use_conf=None):
    """Mean |acceleration| of root-relative joints, mm/frame^2, computed only across consecutive frames.
    use_conf: None (all joints), 'high' (conf >= 0.5) or 'low' (conf < 0.5)."""
    j = _rel(seq.joints3d)[:, joints] * 1000
    c = seq.joint_conf[:, joints]
    vals = []
    for run in _contiguous_runs(seq.frame_idx):
        if len(run) < 3:
            continue
        a = np.linalg.norm(j[run][2:] - 2 * j[run][1:-1] + j[run][:-2], axis=-1)
        cc = c[run][1:-1]
        if use_conf == "high":
            a = a[cc >= 0.5]
        elif use_conf == "low":
            a = a[cc < 0.5]
        vals.append(a[np.isfinite(a)].ravel())
    v = np.concatenate(vals) if vals else np.array([])
    return float(np.mean(v)) if len(v) else np.nan


def rotation_jitter(seq):
    """Mean rotation acceleration of local joint rotations, deg/frame^2 (SMPL models only).
    Same quantity as the 'Jitter' columns in the progress report."""
    if not seq.is_smpl:
        return np.nan
    from .rotations import aa_to_rotmat, geodesic_deg
    R = aa_to_rotmat(np.concatenate([seq.global_orient[:, None], seq.body_pose[:, :21]], 1))
    vals = []
    for run in _contiguous_runs(seq.frame_idx):
        if len(run) < 3:
            continue
        r = R[run]
        v1 = geodesic_deg(r[1:-1], r[:-2])
        v2 = geodesic_deg(r[2:], r[1:-1])
        vals.append(np.abs(v2 - v1).ravel())
    v = np.concatenate(vals) if vals else np.array([])
    return float(np.mean(v)) if len(v) else np.nan


def bone_length_cv(seq):
    """Mean coefficient of variation of bone lengths over time (0 = rigid skeleton). Catches
    MediaPipe's changing bone lengths; SMPL models are ~0 by construction."""
    j = np.asarray(seq.joints3d, np.float64)
    L = np.stack([np.linalg.norm(j[:, c] - j[:, p], axis=-1) for c, p in BONES], 1)
    m = np.nanmean(L, 0)
    return float(np.nanmean(np.nanstd(L, 0) / np.maximum(m, 1e-6)))


def foot_skate(seq, contact_speed=0.01):
    """Mean horizontal foot speed (mm/frame) while the foot should be planted: the lower foot, in frames
    where its vertical speed is below `contact_speed` m/frame. Needs absolute positions, so NaN for
    root-relative outputs. Camera frame: y is down, so 'lower' = larger y."""
    if seq.joints3d_rel:
        return np.nan
    j = np.asarray(seq.joints3d, np.float64)
    vals = []
    for run in _contiguous_runs(seq.frame_idx):
        if len(run) < 2:
            continue
        f = j[run][:, [7, 8]]                                 # ankles
        low = np.argmax(f[..., 1], axis=1)                     # index of the lower ankle per frame
        fl = f[np.arange(len(f)), low]
        d = np.diff(fl, axis=0)
        same = low[1:] == low[:-1]
        contact = same & (np.abs(d[:, 1]) < contact_speed)
        vals.append(np.linalg.norm(d[contact][:, [0, 2]], axis=-1) * 1000)
    v = np.concatenate(vals) if vals else np.array([])
    return float(np.mean(v)) if len(v) else np.nan


def left_right_swaps(seq, pairs=((4, 5), (7, 8), (18, 19), (20, 21)), margin=0.05):
    """Number of frame transitions where swapping a left/right pair would fit the previous frame
    better by more than `margin` metres (a likely left/right flip)."""
    j = _rel(seq.joints3d)
    n = 0
    for run in _contiguous_runs(seq.frame_idx):
        for a, b in pairs:
            A, B = j[run][:, a], j[run][:, b]
            keep = np.linalg.norm(A[1:] - A[:-1], axis=-1) + np.linalg.norm(B[1:] - B[:-1], axis=-1)
            swap = np.linalg.norm(A[1:] - B[:-1], axis=-1) + np.linalg.norm(B[1:] - A[:-1], axis=-1)
            n += int(np.sum(swap + margin < keep))
    return n


def pose_jumps(seq, thresh_m=0.25):
    """Fraction of frame transitions where some root-relative joint moves more than `thresh_m` metres
    in one frame (limb flips, lost limbs)."""
    j = _rel(seq.joints3d)[:, EVAL_JOINTS]
    jumps, total = 0, 0
    for run in _contiguous_runs(seq.frame_idx):
        if len(run) < 2:
            continue
        d = np.nanmax(np.linalg.norm(np.diff(j[run], axis=0), axis=-1), axis=1)
        jumps += int(np.sum(d > thresh_m))
        total += len(d)
    return jumps / max(total, 1)


def gaps(seq):
    """(number of gaps, longest gap in frames) in the detections."""
    d = np.diff(np.concatenate([[-1], seq.frame_idx, [seq.n_frames]])) - 1
    d = d[d > 0]
    return int(len(d)), int(d.max()) if len(d) else 0


def agreement(seq_a, seq_b, joints=EVAL_JOINTS):
    """PA-MPJPE (mm) between two models' outputs on the frames both found the person.
    With no ground truth, disagreement with the consensus of other models flags likely failures."""
    common, ia, ib = np.intersect1d(seq_a.frame_idx, seq_b.frame_idx, return_indices=True)
    if len(common) == 0:
        return np.nan
    A, B = seq_a.joints3d[ia][:, joints], seq_b.joints3d[ib][:, joints]
    ok = np.isfinite(A).all((1, 2)) & np.isfinite(B).all((1, 2))
    if not ok.any():
        return np.nan
    _, pa = joint_errors(A[ok], B[ok], joints=list(range(len(joints))))
    return float(pa.mean())


def proxy_report(seq):
    """All proxy metrics for one PoseSeq, as a flat dict (one row of the RQ1 failure table)."""
    g, gmax = gaps(seq)
    return {
        "model": seq.model, "clip": seq.clip, "frames": int(seq.n_frames),
        "detection_rate": round(seq.detection_rate, 3), "gaps": g, "longest_gap": gmax,
        "jitter_mm": joint_jitter(seq), "jitter_mm_low_conf": joint_jitter(seq, use_conf="low"),
        "rot_jitter_deg": rotation_jitter(seq), "bone_cv": bone_length_cv(seq), "foot_skate_mm": foot_skate(seq),
        "lr_swaps": left_right_swaps(seq), "pose_jumps": pose_jumps(seq),
        "low_conf_joints": float(np.mean(seq.joint_conf[:, EVAL_JOINTS] < 0.5)),
    }


def rating_template(clips, models, path):
    """CSV for the two-rater visual score (1 = unusable ... 5 = looks right), one row per clip x model."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "model", "rater1_score_1to5", "rater2_score_1to5", "main_failure", "notes"])
        for c in clips:
            for m in models:
                w.writerow([c, m, "", "", "", ""])
    return path


FOOT = FOOT_JOINTS
LR = LEFT_RIGHT
