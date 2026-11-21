"""
Convert the SMPL .pkl (basicModel_neutral_lbs_10_207_0_v1.0.0.pkl) to a plain .npz
WITHOUT chumpy (chumpy breaks on Python 3.11+). Chumpy arrays are unpickled into a stub
and their stored value ('x') is pulled out.

    python smpl_pkl_to_npz.py basicModel_neutral_lbs_10_207_0_v1.0.0.pkl smpl_neutral.npz
"""
import pickle
import sys

import numpy as np


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


def to_np(x):
    if isinstance(x, _Stub):
        if "x" in x.__dict__:
            return np.asarray(x.__dict__["x"])
        raise ValueError(f"chumpy object without stored value: {list(x.__dict__)}")
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def convert(src, dst):
    with open(src, "rb") as fh:
        d = _Unpickler(fh, encoding="latin1").load()
    out = {}
    for k in ("v_template", "f", "J_regressor", "weights", "kintree_table", "shapedirs", "posedirs"):
        if k in d:
            out[k] = to_np(d[k])
    out["f"] = out["f"].astype(np.int64)
    out["kintree_table"] = out["kintree_table"].astype(np.int64)
    out["kintree_table"][0, 0] = -1
    np.savez(dst, **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
