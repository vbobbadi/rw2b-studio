"""Convert every model's raw output into the standard PoseSeq file (pose_lib.format).

SMPL-type models need forward kinematics to get joints3d, so pass `fk = SMPLJointsNP(smpl_neutral.npz)`.

    from pose_lib.smpl import SMPLJointsNP
    from pose_lib import convert
    fk = SMPLJointsNP(".../data/smpl_neutral.npz")
    convert.from_smpl_npz(".../outputs/video_smpl.npz", fk, model="hmr2", clip="video").save(...)
    convert.from_mediapipe(".../outputs/clipA_full.npz", fps=24, model="mediapipe_full", clip="clipA").save(...)
"""

import numpy as np

from .format import PoseSeq

# ---------------------------------------------------------------- SMPL-param files
_ALIASES = {
    "global_orient": ("global_orient", "root_orient", "root_orientation", "go"),
    "body_pose": ("body_pose", "pose_body", "body"),
    "betas": ("betas", "shape", "body_shape"),
    "camera": ("camera", "transl", "trans", "root_position", "root_pos", "cam_t"),
    "frame_idx": ("frame_idx", "frames", "frame_numbers", "frame_ids", "frame"),
    "score": ("score", "scores", "confidence", "det_score"),
    "fps": ("fps", "mocap_framerate"),
}


def _pick(d, name, required=True):
    for k in _ALIASES[name]:
        if k in d:
            return np.asarray(d[k])
    if required:
        raise KeyError(f"none of {_ALIASES[name]} found; keys are {sorted(d.keys())}")
    return None


def from_params(global_orient, body_pose, fk, *, model, clip, fps, frame_idx=None, n_frames=None, betas=None,
                camera=None, score=None, bbox=None, joint_conf=None, camera_is_reliable=True, extra=None,
                img_size=None, track_id=-1):
    """Core SMPL converter. global_orient (T,3), body_pose (T,21|23,3) axis-angle, camera (T,3) metres."""
    go = np.asarray(global_orient, np.float64).reshape(-1, 3)
    T = len(go)
    bp = np.asarray(body_pose, np.float64).reshape(T, -1, 3)
    if bp.shape[1] == 21:                 # SMPL-X body (21) -> SMPL (23, hands zero)
        bp = np.concatenate([bp, np.zeros((T, 2, 3))], 1)
    bp = bp[:, :23]
    b = np.zeros(10) if betas is None else np.asarray(betas, np.float64)
    if b.ndim == 2 and b.shape[0] == T:
        b = b[:, :10]
    else:
        b = b.reshape(-1)[:10]
    j = fk(go, bp, b)                                     # (T,24,3), root at rest position
    if camera is not None:
        camera = np.asarray(camera, np.float64).reshape(T, 3)
        if np.nanmedian(camera[:, 2]) > 20:               # e.g. PHALP's focal-5000 camera: depth is not metric
            camera_is_reliable = False
        if camera_is_reliable:
            j = j + camera[:, None]
    fi = np.arange(T) if frame_idx is None else np.asarray(frame_idx).reshape(-1)
    return PoseSeq(model=model, clip=clip, fps=float(fps), frame_idx=fi,
                   n_frames=int(n_frames if n_frames is not None else fi.max() + 1), joints3d=j,
                   joints3d_rel=(camera is None or not camera_is_reliable), joint_conf=joint_conf,
                   global_orient=go, body_pose=bp, betas=b, camera=camera, score=score, bbox=bbox,
                   extra=extra or {}, img_size=img_size, track_id=track_id)


def from_smpl_npz(path, fk, *, model="hmr2", clip=None, n_frames=None, fps=None):
    """Notebook 01 (HMR2.0 + PHALP) and Madeeha's Multi-HMR video_smpl.npz, or anything with the same keys.
    Multi-HMR's SMPL-X body pose (21 joints, hands zeroed) maps 1:1 onto SMPL joints 1..21."""
    import os
    d = dict(np.load(path, allow_pickle=True))
    f = _pick(d, "fps", required=False)
    fps = float(fps if fps is not None else (f if f is not None else 30))
    clip = clip or os.path.basename(path).replace("_smpl.npz", "").replace(".npz", "")
    extra = {k: d[k] for k in d if k not in sum(_ALIASES.values(), ()) and np.asarray(d[k]).dtype != object}
    return from_params(_pick(d, "global_orient"), _pick(d, "body_pose"), fk, model=model, clip=clip, fps=fps,
                       frame_idx=_pick(d, "frame_idx", False), n_frames=n_frames, betas=_pick(d, "betas", False),
                       camera=_pick(d, "camera", False), score=_pick(d, "score", False), extra=extra)


# ---------------------------------------------------------------- MediaPipe (33 landmarks)
MP = dict(nose=0, l_ear=7, r_ear=8, l_sh=11, r_sh=12, l_el=13, r_el=14, l_wr=15, r_wr=16, l_pinky=17,
          r_pinky=18, l_index=19, r_index=20, l_hip=23, r_hip=24, l_kn=25, r_kn=26, l_an=27, r_an=28,
          l_heel=29, r_heel=30, l_toe=31, r_toe=32)


def mediapipe33_to_smpl24(x33, vis33=None):
    """(T,33,3) MediaPipe world landmarks -> (T,24,3) SMPL-order joints, plus (T,24) confidence.
    Directly observed: hips, knees, ankles, feet, shoulders, elbows, wrists. Derived (pelvis, spine,
    neck, collars, head, hands) get the minimum visibility of the landmarks they're built from."""
    x = np.asarray(x33, np.float64)
    T = len(x)
    v = np.ones((T, 33)) if vis33 is None else np.asarray(vis33, np.float64)
    J = np.full((T, 24, 3), np.nan)
    C = np.zeros((T, 24))

    def put(j, idx, w=None):
        idx = list(idx)
        w = np.ones(len(idx)) / len(idx) if w is None else np.asarray(w)
        J[:, j] = np.einsum("k,tkc->tc", w, x[:, idx])
        C[:, j] = v[:, idx].min(1)

    m = MP
    put(1, [m["l_hip"]]); put(2, [m["r_hip"]]); put(4, [m["l_kn"]]); put(5, [m["r_kn"]])
    put(7, [m["l_an"]]); put(8, [m["r_an"]]); put(10, [m["l_toe"]]); put(11, [m["r_toe"]])
    put(16, [m["l_sh"]]); put(17, [m["r_sh"]]); put(18, [m["l_el"]]); put(19, [m["r_el"]])
    put(20, [m["l_wr"]]); put(21, [m["r_wr"]])
    put(22, [m["l_index"], m["l_pinky"]]); put(23, [m["r_index"], m["r_pinky"]])
    put(0, [m["l_hip"], m["r_hip"]])
    put(12, [m["l_sh"], m["r_sh"]])
    put(15, [m["l_ear"], m["r_ear"]])
    pel, neck = J[:, 0], J[:, 12]
    cp = np.minimum(C[:, 0], C[:, 12])
    for j, a in ((3, 0.2), (6, 0.45), (9, 0.6)):
        J[:, j] = pel + a * (neck - pel)
        C[:, j] = cp
    J[:, 13] = neck + 0.35 * (J[:, 16] - neck); C[:, 13] = np.minimum(C[:, 12], C[:, 16])
    J[:, 14] = neck + 0.35 * (J[:, 17] - neck); C[:, 14] = np.minimum(C[:, 12], C[:, 17])
    return J, C


def from_mediapipe(path, *, fps, model="mediapipe", clip=None, use="raw", n_frames=None):
    """Nishitha's clipX_<size>.npz: world/raw (N,33,3) hip-centred metres, visibility (N,33), found (N,).
    MediaPipe world axes: x right, y down, z away from camera (same as OpenCV), origin at the hips,
    so joints3d is root-relative. Frames with found == 0 are dropped (the file fills them by interpolation)."""
    import os
    d = np.load(path)
    x = d[use]
    found = d["found"].astype(bool) if "found" in d.files else np.ones(len(x), bool)
    vis = d["visibility"] if "visibility" in d.files else None
    J, C = mediapipe33_to_smpl24(x, vis)
    keep = found & np.isfinite(J[:, 0, 0])
    clip = clip or os.path.basename(path).split("_")[0]
    extra = {"landmarks33": x[keep].astype(np.float32)}
    if vis is not None:
        extra["visibility33"] = vis[keep].astype(np.float32)
    return PoseSeq(model=model, clip=clip, fps=float(fps), frame_idx=np.nonzero(keep)[0],
                   n_frames=int(n_frames or len(x)), joints3d=J[keep], joints3d_rel=True, joint_conf=C[keep],
                   extra=extra)


# ---------------------------------------------------------------- PARE / WHAM / GVHMR
def _load_any(path):
    if str(path).endswith((".pt", ".pth")):
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    import joblib
    return joblib.load(path)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _pick_track(res, track_id):
    if track_id is None:
        track_id = max(res, key=lambda k: len(_np(res[k].get("frame_ids", res[k].get("frames", [])))))
    return track_id, res[track_id]


def from_pare(path_or_dict, fk, *, fps, clip, model="pare", track_id=None, n_frames=None, focal=5000.0, crop=224):
    """PARE demo output (pare_output.pkl): {track_id: {pose (T,72), betas (T,10), pred_cam (T,3), frame_ids}}.
    pred_cam is weak-perspective (s, tx, ty) in the crop, so depth is only approximate -> joints3d_rel."""
    res = _load_any(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
    tid, r = _pick_track(res, track_id)
    pose = _np(r["pose"])
    if pose.ndim == 4:                                   # (T,24,3,3) rotation matrices
        from .rotations import rotmat_to_aa
        pose = rotmat_to_aa(pose)
    pose = pose.reshape(len(pose), 24, 3)
    cam = _np(r["pred_cam"]) if "pred_cam" in r else None
    camera = None
    if cam is not None:
        camera = np.stack([cam[:, 1], cam[:, 2], 2 * focal / (crop * cam[:, 0] + 1e-9)], -1)
    return from_params(pose[:, 0], pose[:, 1:], fk, model=model, clip=clip, fps=fps,
                       frame_idx=_np(r.get("frame_ids", np.arange(len(pose)))), n_frames=n_frames,
                       betas=_np(r["betas"]) if "betas" in r else None, camera=camera, camera_is_reliable=False,
                       bbox=_np(r["bboxes"]) if "bboxes" in r else None, track_id=int(tid))


def from_wham(path_or_dict, fk, *, fps, clip, model="wham", track_id=None, n_frames=None):
    """WHAM demo output (wham_output.pkl): {id: {pose (T,72), trans (T,3), betas (T,10), pose_world,
    trans_world, frame_ids}}. Camera-frame result goes in the main keys, world-frame in extra_world_*."""
    res = _load_any(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
    tid, r = _pick_track(res, track_id)
    pose = _np(r["pose"]).reshape(-1, 24, 3)
    extra = {}
    if "pose_world" in r:
        pw = _np(r["pose_world"]).reshape(-1, 24, 3)
        extra = {"world_global_orient": pw[:, 0], "world_transl": _np(r["trans_world"])}
    return from_params(pose[:, 0], pose[:, 1:], fk, model=model, clip=clip, fps=fps,
                       frame_idx=_np(r.get("frame_ids", np.arange(len(pose)))), n_frames=n_frames,
                       betas=_np(r["betas"]), camera=_np(r["trans"]), extra=extra, track_id=int(tid))


def from_gvhmr(path_or_dict, fk, *, fps, clip, model="gvhmr", n_frames=None):
    """GVHMR demo output (hmr4d_results.pt): smpl_params_incam / smpl_params_global with body_pose (T,63)
    SMPL-X body, global_orient (T,3), transl (T,3), betas (T,10). One person, every frame."""
    res = _load_any(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
    p = {k: _np(v) for k, v in res["smpl_params_incam"].items()}
    T = len(p["global_orient"])
    extra = {}
    if "smpl_params_global" in res:
        g = {k: _np(v) for k, v in res["smpl_params_global"].items()}
        extra = {"world_global_orient": g["global_orient"], "world_transl": g["transl"]}
    return from_params(p["global_orient"], p["body_pose"].reshape(T, -1, 3), fk, model=model, clip=clip,
                       fps=fps, frame_idx=np.arange(T), n_frames=n_frames or T, betas=p.get("betas"),
                       camera=p.get("transl"), extra=extra)
