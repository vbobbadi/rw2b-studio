"""SMPL without chumpy: .pkl -> .npz, a chumpy-free .pkl for HMR2.0, and forward kinematics.

This is the fix for the 'environment currently broken' risk in the progress report: HMR2.0 and the
data generator only need the arrays inside the SMPL .pkl, so we convert once and never import chumpy.

    python -m pose_lib.smpl basicModel_neutral_lbs_10_207_0_v1.0.0.pkl smpl_neutral.npz
"""

import os
import pickle
import sys

import numpy as np

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_JOINTS = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle",
               "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder",
               "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"]
EVAL_JOINTS = list(range(22))          # standard: drop the two hand joints
FOOT_JOINTS = [7, 8, 10, 11]           # ankles + feet, for foot-skating
# bones used for bone-length consistency (child, parent) on the 22 body joints
BONES = [(j, p) for j, p in enumerate(SMPL_PARENTS[:22]) if p >= 0]
LEFT_RIGHT = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)]


# ---------------- .pkl -> .npz ----------------
class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def _to_np(x):
    if isinstance(x, _Stub):
        if "x" in x.__dict__:
            return np.asarray(x.__dict__["x"])
        raise ValueError(f"chumpy object without stored value: {list(x.__dict__)}")
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


KEYS = ("v_template", "f", "J_regressor", "weights", "kintree_table", "shapedirs", "posedirs")


def pkl_to_npz(src, dst):
    with open(src, "rb") as fh:
        d = _Unpickler(fh, encoding="latin1").load()
    out = {k: _to_np(d[k]) for k in KEYS if k in d}
    out["f"] = out["f"].astype(np.int64)
    out["kintree_table"] = out["kintree_table"].astype(np.int64)
    out["kintree_table"][0, 0] = -1
    np.savez(dst, **out)
    return {k: v.shape for k, v in out.items()}


def write_plain_pkl(npz_path, pkl_path):
    """A chumpy-free SMPL_NEUTRAL.pkl (plain numpy arrays) for HMR2.0 / smplx (notebook 05 trick)."""
    m = dict(np.load(npz_path))
    os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(m, fh)
    return pkl_path


def setup_hmr2_smpl(smpl_npz, cache_dir=None):
    """Put the plain SMPL .pkl where HMR2.0 (4D-Humans) looks for it."""
    if cache_dir is None:
        from hmr2.configs import CACHE_DIR_4DHUMANS as cache_dir  # noqa: N811
    return write_plain_pkl(smpl_npz, os.path.join(cache_dir, "data", "smpl", "SMPL_NEUTRAL.pkl"))


# ---------------- forward kinematics (numpy) ----------------
class SMPLJointsNP:
    """24 SMPL joints from axis-angle params. Root at its rest position (zero translation)."""

    def __init__(self, smpl_npz):
        m = np.load(smpl_npz)
        Jr = m["J_regressor"]
        Jr = Jr.toarray() if hasattr(Jr, "toarray") else np.asarray(Jr, dtype=np.float64)
        v = m["v_template"].astype(np.float64)
        sd = m["shapedirs"].astype(np.float64)[..., :10]
        self.J_template = Jr @ v
        self.J_dirs = np.einsum("jv,vck->jck", Jr, sd)

    @classmethod
    def from_arrays(cls, J_template, J_dirs):
        o = cls.__new__(cls)
        o.J_template, o.J_dirs = np.asarray(J_template, float), np.asarray(J_dirs, float)
        return o

    def rest_joints(self, betas):
        betas = np.atleast_2d(np.asarray(betas, dtype=np.float64))[..., :10]
        return self.J_template[None] + np.einsum("jck,bk->bjc", self.J_dirs, betas)

    def __call__(self, global_orient, body_pose, betas=None):
        """global_orient (T,3) aa, body_pose (T,23,3) aa, betas (10,) or (T,10) -> (T,24,3)."""
        from .rotations import aa_to_rotmat
        go = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
        T = len(go)
        bp = np.asarray(body_pose, dtype=np.float64).reshape(T, -1, 3)
        if bp.shape[1] == 21:
            bp = np.concatenate([bp, np.zeros((T, 2, 3))], 1)
        b = np.zeros(10) if betas is None else np.asarray(betas, dtype=np.float64)
        J = self.rest_joints(b)
        if len(J) == 1:
            J = np.repeat(J, T, 0)
        R = aa_to_rotmat(np.concatenate([go[:, None], bp], 1))
        return fk(R, J)


def fk(R, J, parents=SMPL_PARENTS):
    """R (T,24,3,3) local rotations, J (T,24,3) rest joints -> posed joints (T,24,3)."""
    G = [None] * len(parents)
    P = [None] * len(parents)
    for j, p in enumerate(parents):
        if p < 0:
            G[j], P[j] = R[:, 0], J[:, 0]
        else:
            G[j] = G[p] @ R[:, j]
            P[j] = P[p] + (G[p] @ (J[:, j] - J[:, p])[..., None])[..., 0]
    return np.stack(P, 1)


# ---------------- forward kinematics (torch) ----------------
def smpl_joints_torch(smpl_npz):
    """Factory so importing this module doesn't require torch."""
    import torch
    import torch.nn as nn

    class SMPLJoints(nn.Module):
        """(global_orient (B,3,3), body_pose (B,23,3,3), betas (B,10)) -> joints (B,24,3)."""

        def __init__(self, path):
            super().__init__()
            npj = SMPLJointsNP(path) if isinstance(path, str) else path
            self.register_buffer("J_template", torch.tensor(npj.J_template, dtype=torch.float32))
            self.register_buffer("J_dirs", torch.tensor(npj.J_dirs, dtype=torch.float32))

        def forward(self, global_orient, body_pose, betas):
            J = self.J_template[None] + torch.einsum("jck,bk->bjc", self.J_dirs, betas)
            R = torch.cat([global_orient[:, None], body_pose], dim=1)
            G, P = [None] * 24, [None] * 24
            for j, p in enumerate(SMPL_PARENTS):
                if p < 0:
                    G[j], P[j] = R[:, 0], J[:, 0]
                else:
                    G[j] = G[p] @ R[:, j]
                    P[j] = P[p] + (G[p] @ (J[:, j] - J[:, p])[..., None])[..., 0]
            return torch.stack(P, dim=1)

    return SMPLJoints(smpl_npz)


def fake_smpl_npz(path, seed=0):
    """A small fake 'SMPL' with a plausible rest skeleton, for tests only (no licensed data)."""
    rng = np.random.default_rng(seed)
    rest = np.array([
        [0, 0, 0], [.08, -.08, 0], [-.08, -.08, 0], [0, .1, 0], [.1, -.45, 0], [-.1, -.45, 0], [0, .23, 0],
        [.1, -.85, 0], [-.1, -.85, 0], [0, .28, 0], [.1, -.9, .1], [-.1, -.9, .1], [0, .5, 0], [.07, .42, 0],
        [-.07, .42, 0], [0, .62, 0], [.18, .45, 0], [-.18, .45, 0], [.45, .45, 0], [-.45, .45, 0],
        [.7, .45, 0], [-.7, .45, 0], [.78, .45, 0], [-.78, .45, 0]], dtype=np.float64)
    V = 24 * 4
    v = np.repeat(rest, 4, 0) + rng.normal(0, 0.01, (V, 3))
    Jr = np.zeros((24, V))
    for j in range(24):
        Jr[j, 4 * j:4 * j + 4] = 0.25
    W = np.zeros((V, 24))
    W[np.arange(V), np.arange(V) // 4] = 1
    kt = np.stack([np.array(SMPL_PARENTS), np.arange(24)])
    np.savez(path, v_template=v, f=np.zeros((1, 3), np.int64), J_regressor=Jr, weights=W,
             kintree_table=kt, shapedirs=rng.normal(0, 0.01, (V, 3, 10)), posedirs=np.zeros((V, 3, 207)))
    return path


if __name__ == "__main__":
    print(pkl_to_npz(sys.argv[1], sys.argv[2]))
