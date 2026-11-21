"""Execute notebooks 06-10 end to end on CPU with the toy model and toy data.
Setup/install cells are replaced; '!'/'%' lines are skipped; 'cuda' -> 'cpu'.
    python tests/dryrun_notebooks.py /tmp/dry
"""

import functools
import json
import os
import re
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def toy_env(tmp):
    import numpy as np
    import torch
    import toy
    from pose_lib import detector as dt, experts as ex, hmr2_ft as ft
    smpl, data = toy.setup(tmp)
    os.makedirs(f"{tmp}/data", exist_ok=True)
    shutil.copy(smpl, f"{tmp}/data/smpl_neutral.npz")
    torch.set_num_threads(2)

    def cpu(fn):
        @functools.wraps(fn)
        def w(*a, **k):
            import inspect
            params = inspect.signature(fn).parameters
            for key, val in (("device", "cpu"), ("workers", 0), ("amp", False)):
                if key in params:
                    k[key] = val
            return fn(*a, **k)
        return w
    for mod, names in ((ft, ["train", "evaluate"]), (dt, ["train", "evaluate"]), (ex, ["cache_for_fusion"])):
        for n in names:
            setattr(mod, n, cpu(getattr(mod, n)))
    ft.load_model = lambda: toy.ToyHMR()
    _Det = dt.Detector
    dt.Detector = lambda backbone="resnet18", pretrained=True: _Det("tiny")
    return {"PROJECT_DIR": tmp, "CODE_DIR": REPO, "DATA": data, "SMPL_NPZ": f"{tmp}/data/smpl_neutral.npz"}


SETUP_SUB = '''
import os, sys, json, glob, time, gc, shutil, subprocess
import numpy as np
import torch
from pose_lib import hmr2_ft as ft, lora, metrics, stats
fk = ft.SMPLJoints(SMPL_NPZ)
HAS_SELFOCC = "selfvis_joint" in np.load(f"{DATA}/test/D_occ3d/labels.npz").files
def display(x): print(x if not hasattr(x, "to_string") else x.to_string()[:2000])
'''


def run_notebook(path, env, skip_markers=()):
    nb = json.load(open(path))
    g = dict(env)
    g["__name__"] = "__main__"
    exec(SETUP_SUB, g)
    import matplotlib
    matplotlib.use("Agg")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith(("PROJECT_DIR =", "# HMR2.0 without PHALP", "# synthetic AMASS dataset")):
            continue                                             # replaced by the toy env
        if any(m in src for m in skip_markers):
            print(f"  [skip cell {i}]")
            continue
        lines = []
        for l in src.split("\n"):
            s = l.lstrip()
            lines.append(l[:len(l) - len(s)] + "pass" if s.startswith(("!", "%")) else l)
        src = "\n".join(lines).replace('"cuda"', '"cpu"').replace(".cuda()", ".cpu()")
        src = src.replace('map_location="cpu"', 'map_location="cpu", weights_only=False')
        try:
            exec(compile(src, f"{os.path.basename(path)}:cell{i}", "exec"), g)
        except Exception:
            print(f"  FAILED in cell {i}:\n" + "\n".join(src.split("\n")[:6]))
            traceback.print_exc()
            return False, g
    return True, g


if __name__ == "__main__":
    import numpy as np
    root = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    env = toy_env(root)
    NB = os.path.join(REPO, "notebooks")
    results = {}
    # 06: shrink to 2 seeds and fewer epochs; the selfocc (Blender) cell is skipped
    src06 = json.load(open(f"{NB}/06_seeds_rq2.ipynb"))
    for c in src06["cells"]:
        s = "".join(c["source"])
        if s.startswith("SEEDS"):
            c["source"] = ["SEEDS=[0,1]\nCONDITIONS=['B_clean','C_paste2d','D_occ3d']\nEPOCHS,BATCH,LR,N_BLOCKS=1,8,1e-3,2\nADD_SELFOCC=False\n"]
    p06 = f"{root}/06.ipynb"; json.dump(src06, open(p06, "w"))
    results["06"] = run_notebook(p06, env)[0]

    def shrink(name, repl):
        nbj = json.load(open(f"{NB}/{name}.ipynb"))
        for c in nbj["cells"]:
            s = "".join(c["source"])
            for k, v in repl.items():
                if s.lstrip().startswith(k):
                    c["source"] = [v]
        p = f"{root}/{name}.ipynb"; json.dump(nbj, open(p, "w")); return p

    p07 = shrink("07_experts_rq4", {"EXPERTS = ": 'EXPERTS=["E0","E1","E2","E3"]\nR_LORA,LORA_BLOCKS=4,2\nEPOCHS,BATCH,LR,SEED=1,8,1e-3,0\nRUN=f"experts_r{R_LORA}_b{LORA_BLOCKS}_s{SEED}"\n'})
    results["07"] = run_notebook(p07, env)[0]
    p08 = shrink("08_detector_fusion_rq5_rq6", {"EXPERTS = ": 'EXPERTS=["E0","E1","E2","E3"]\nR_LORA,LORA_BLOCKS,SEED=4,2,0\nRUN=f"experts_r{R_LORA}_b{LORA_BLOCKS}_s{SEED}"\nDET_EPOCHS,DET_PER_SOURCE=1,16\n'})
    # notebook 08 builds det_val with 400/source: fine (MixtureDataset caps at source size)
    results["08"] = run_notebook(p08, env)[0]
    p10 = shrink("10_selector_rq3", {"STRATEGIES": 'STRATEGIES,SEEDS=["random","hem","bandit"],[0,1]\nROUNDS,ROUND_SIZE,STEPS_PER_ROUND=2,8,2\nR_LORA,LORA_BLOCKS,LR,BATCH=4,2,1e-3,4\n'})
    results["10"] = run_notebook(p10, env)[0]

    # 09: sections 2-5 with real sample files (video section and Blender export skipped)
    from pose_lib import convert
    from pose_lib.smpl import SMPLJointsNP
    fknp = SMPLJointsNP(env["SMPL_NPZ"])
    real = f"{root}/results/real"; os.makedirs(real, exist_ok=True)
    S = "/home/claude/samples"
    convert.from_smpl_npz(f"{S}/video_smpl.npz", fknp, model="hmr2", clip="row01_walk_01").save(f"{real}/row01_walk_01__hmr2.npz")
    raw = f"{root}/results/real_raw"
    os.makedirs(f"{raw}/mediapipe_full", exist_ok=True); os.makedirs(f"{raw}/multihmr", exist_ok=True)
    shutil.copy(f"{S}/clipA_full.npz", f"{raw}/mediapipe_full/clipA_full.npz")
    shutil.copy(f"{S}/video_smpl.npz", f"{raw}/multihmr/row01_walk_01_smpl.npz")
    p09 = shrink("09_real_video_rq1_rq6", {"TARGET_H": 'TARGET_H, MAX_SECONDS = 720, 60\nEXPERTS=["E0","E1","E2","E3"]\nRUN="none"\nLEGACY={"clipA": 9, "clipB": 7}\n',
                                            "!pip install -q ultralytics": 'from pose_lib import video, convert, experts as ex, detector as dt, fusion, smoothing\nfrom pose_lib.format import PoseSeq, parse_name\nfrom pose_lib.smpl import SMPLJointsNP\nfknp = SMPLJointsNP(SMPL_NPZ)\nREAL = f"{PROJECT_DIR}/results/real"\n'})
    ok, g = run_notebook(p09, env, skip_markers=("video.run(", "smplx_mocap.py"))
    results["09"] = ok
    if ok:
        print(g["P"][["clip", "model", "row", "group", "detection_rate", "jitter_mm", "bone_cv", "disagreement_mm", "flags"]].to_string())
    print("\nRESULT:", results)
    sys.exit(0 if all(results.values()) else 1)
