"""The standard per-clip result file. Every model's output is converted to this, so metrics,
smoothing, fusion and Blender all read one format.

File name convention:  <clip_stem>__<model>.npz   e.g.  row07_desk_01__hmr2.npz

Keys (T = frames where the person was found; arrays are float32 unless noted)
------------------------------------------------------------------------------
format_version  ()            int, = 1
model           ()            str   e.g. "hmr2", "multihmr", "mediapipe_full", "pare", "gvhmr", "E1_occ"
clip            ()            str   clip stem, e.g. "row07_desk_01"
fps             ()            float frames per second of the processed clip
frame_idx       (T,)          int   frame number in the processed clip
n_frames        ()            int   frames in the processed clip (so detection rate = T / n_frames)
joints3d        (T,24,3)      SMPL-24 joint order, metres, camera frame (x right, y down, z forward).
                              NaN where a model has no estimate for that joint (e.g. MediaPipe hands).
joints3d_rel    ()            bool  True if joints3d is root-relative (no reliable depth)
joint_conf      (T,24)        0..1 per-joint confidence; 1 if the model gives none

SMPL models also have (absent for landmark-only models such as MediaPipe):
global_orient   (T,3)         axis-angle, camera frame
body_pose       (T,23,3)      axis-angle, SMPL order (the same layout notebook 02's smplx_mocap.py reads)
betas           (10,)         or (T,10)
camera          (T,3)         translation of the body in the camera frame, metres

Optional:
bbox            (T,4)         x0,y0,x1,y1 pixels
score           (T,)          detector score
track_id        ()            int
img_size        (2,)          width, height of the processed clip
extra_*         anything model-specific (e.g. extra_landmarks33 for MediaPipe)

Backward compatible: files written here load in notebooks 02/03 because global_orient, body_pose,
camera, frame_idx and fps keep the same names and shapes as notebook 01's video_smpl.npz.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FORMAT_VERSION = 1
SMPL_KEYS = ("global_orient", "body_pose", "betas", "camera")


@dataclass
class PoseSeq:
    model: str
    clip: str
    fps: float
    frame_idx: np.ndarray
    n_frames: int
    joints3d: np.ndarray
    joint_conf: Optional[np.ndarray] = None
    joints3d_rel: bool = False
    global_orient: Optional[np.ndarray] = None
    body_pose: Optional[np.ndarray] = None
    betas: Optional[np.ndarray] = None
    camera: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    score: Optional[np.ndarray] = None
    track_id: int = -1
    img_size: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    # ---------- checks ----------
    def __post_init__(self):
        self.frame_idx = np.asarray(self.frame_idx, dtype=np.int64).reshape(-1)
        T = len(self.frame_idx)
        self.joints3d = np.asarray(self.joints3d, dtype=np.float32).reshape(T, 24, 3)
        self.joint_conf = (np.ones((T, 24), np.float32) if self.joint_conf is None
                           else np.asarray(self.joint_conf, dtype=np.float32).reshape(T, 24))
        if self.global_orient is not None:
            self.global_orient = np.asarray(self.global_orient, dtype=np.float32).reshape(T, 3)
            bp = np.asarray(self.body_pose, dtype=np.float32).reshape(T, -1, 3)
            if bp.shape[1] == 21:
                bp = np.concatenate([bp, np.zeros((T, 2, 3), np.float32)], 1)
            assert bp.shape[1] == 23, f"body_pose must have 21 or 23 joints, got {bp.shape}"
            self.body_pose = bp
            self.betas = np.zeros(10, np.float32) if self.betas is None else np.asarray(self.betas, np.float32)
            self.camera = (np.zeros((T, 3), np.float32) if self.camera is None
                           else np.asarray(self.camera, dtype=np.float32).reshape(T, 3))
        assert np.all(np.diff(self.frame_idx) > 0), "frame_idx must be strictly increasing"

    @property
    def T(self):
        return len(self.frame_idx)

    @property
    def is_smpl(self):
        return self.global_orient is not None

    @property
    def detection_rate(self):
        return self.T / max(int(self.n_frames), 1)

    # ---------- io ----------
    def save(self, path):
        d = dict(format_version=FORMAT_VERSION, model=self.model, clip=self.clip, fps=float(self.fps),
                 frame_idx=self.frame_idx, n_frames=int(self.n_frames), joints3d=self.joints3d,
                 joints3d_rel=bool(self.joints3d_rel), joint_conf=self.joint_conf, track_id=int(self.track_id))
        for k in ("global_orient", "body_pose", "betas", "camera", "bbox", "score", "img_size"):
            v = getattr(self, k)
            if v is not None:
                d[k] = np.asarray(v)
        for k, v in self.extra.items():
            d[k if k.startswith("extra_") else "extra_" + k] = np.asarray(v)
        np.savez_compressed(path, **d)
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        if "format_version" not in z.files:
            raise ValueError(f"{path} is not a pose_lib file; convert it with pose_lib.convert first")
        g = lambda k: z[k] if k in z.files else None  # noqa: E731
        return cls(model=str(z["model"]), clip=str(z["clip"]), fps=float(z["fps"]), frame_idx=z["frame_idx"],
                   n_frames=int(z["n_frames"]), joints3d=z["joints3d"], joint_conf=g("joint_conf"),
                   joints3d_rel=bool(z["joints3d_rel"]), global_orient=g("global_orient"),
                   body_pose=g("body_pose"), betas=g("betas"), camera=g("camera"), bbox=g("bbox"),
                   score=g("score"), track_id=int(z["track_id"]), img_size=g("img_size"),
                   extra={k: z[k] for k in z.files if k.startswith("extra_")})

    # ---------- helpers ----------
    def root_relative(self):
        j = self.joints3d.copy()
        return j - j[:, :1]

    def dense(self):
        """joints3d on every frame 0..n_frames-1, NaN where the person was not found."""
        out = np.full((int(self.n_frames), 24, 3), np.nan, np.float32)
        out[self.frame_idx] = self.joints3d
        return out

    def subset(self, mask):
        mask = np.asarray(mask)
        kw = dict(model=self.model, clip=self.clip, fps=self.fps, frame_idx=self.frame_idx[mask],
                  n_frames=self.n_frames, joints3d=self.joints3d[mask], joint_conf=self.joint_conf[mask],
                  joints3d_rel=self.joints3d_rel, track_id=self.track_id, img_size=self.img_size, extra=self.extra)
        if self.is_smpl:
            b = self.betas[mask] if self.betas.ndim == 2 else self.betas
            kw.update(global_orient=self.global_orient[mask], body_pose=self.body_pose[mask], betas=b,
                      camera=self.camera[mask])
        for k in ("bbox", "score"):
            v = getattr(self, k)
            if v is not None:
                kw[k] = v[mask]
        return PoseSeq(**kw)


def result_path(out_dir, clip, model):
    import os
    return os.path.join(out_dir, f"{clip}__{model}.npz")


def parse_name(path):
    """'.../row07_desk_01__hmr2.npz' -> ('row07_desk_01', 'hmr2', 7)."""
    import os
    import re
    stem = os.path.basename(path)[:-4]
    clip, _, model = stem.rpartition("__")
    m = re.match(r"row(\d+)", clip)
    return clip, model, (int(m.group(1)) if m else -1)
