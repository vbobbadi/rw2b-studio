"""Temporal smoothing. Same settings as the baseline studies (Savitzky-Golay, order 3), applied only
inside runs of consecutive frames so a detection gap never gets smeared across."""

import numpy as np
from scipy.signal import savgol_filter

from .format import PoseSeq
from .metrics import _contiguous_runs
from .rotations import aa_to_quat, quat_fix_signs, quat_to_aa


def _sg(x, window, order):
    if len(x) < 3:
        return x
    w = min(window, len(x) if len(x) % 2 else len(x) - 1)
    if w <= order:
        return x
    return savgol_filter(x, w, order, axis=0, mode="interp")


def smooth_positions(x, frame_idx, window=9, order=3):
    out = np.array(x, dtype=np.float64, copy=True)
    for run in _contiguous_runs(frame_idx):
        out[run] = _sg(out[run], window, order)
    return out


def smooth_rotations(aa, frame_idx, window=15, order=3):
    """aa (T,J,3) axis-angle. Quaternions with consistent signs -> SG filter -> renormalise."""
    q = aa_to_quat(np.asarray(aa, np.float64))
    out = np.empty_like(q)
    for run in _contiguous_runs(frame_idx):
        qr = quat_fix_signs(q[run])
        qr = _sg(qr, window, order)
        out[run] = qr / np.linalg.norm(qr, axis=-1, keepdims=True)
    return quat_to_aa(out)


def smooth_seq(seq: PoseSeq, pos_window=9, rot_window=15, order=3, fk=None):
    """Smoothed copy. SMPL models: smooth rotations + camera, then recompute joints by FK (if `fk` given),
    so bone lengths stay exact. Landmark models: smooth joint positions."""
    s = seq.subset(np.ones(seq.T, bool))
    s.model = seq.model + "+sg"
    if seq.is_smpl:
        rot = np.concatenate([seq.global_orient[:, None], seq.body_pose], 1)
        rot = smooth_rotations(rot, seq.frame_idx, rot_window, order)
        s.global_orient, s.body_pose = rot[:, 0].astype(np.float32), rot[:, 1:].astype(np.float32)
        s.camera = smooth_positions(seq.camera, seq.frame_idx, pos_window, order).astype(np.float32)
        if fk is not None:
            j = fk(s.global_orient, s.body_pose, s.betas)
            s.joints3d = (j + (0 if seq.joints3d_rel else s.camera[:, None])).astype(np.float32)
        else:
            s.joints3d = smooth_positions(seq.joints3d, seq.frame_idx, pos_window, order).astype(np.float32)
    else:
        s.joints3d = smooth_positions(seq.joints3d, seq.frame_idx, pos_window, order).astype(np.float32)
    return s
