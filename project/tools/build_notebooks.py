"""Builds notebooks/06..10 as Colab .ipynb files.  python tools/build_notebooks.py"""

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks")


def nb(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        c = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def md(s):
    return ("markdown", s)


def code(s):
    return ("code", s)


# ----------------------------------------------------------------------------- shared cells
SETUP = code('''
PROJECT_DIR = "/content/drive/MyDrive/CS747_project"
CODE_DIR    = f"{PROJECT_DIR}/code/rw2b"          # the rw2b folder (pose_lib/, blender/, notebooks/)

from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, glob, time, gc, shutil, subprocess
import numpy as np
if not os.path.exists(f"{CODE_DIR}/pose_lib"):              # first run: unpack the code next to the zip
    !cd "{PROJECT_DIR}/code" && unzip -oq rw2b.zip
sys.path.insert(0, CODE_DIR)
!nvidia-smi --query-gpu=name,memory.total --format=csv
''')

INSTALL_HMR2 = code('''
# HMR2.0 without PHALP/detectron2 (the install that worked in notebook 05 on Python 3.13)
%cd /content
![ -d 4D-Humans ] || git clone -q https://github.com/shubham-goel/4D-Humans.git
%cd /content/4D-Humans
!pip install -q numpy yacs smplx pytorch_lightning timm einops omegaconf trimesh webdataset pyrender
!pip install -q --no-build-isolation git+https://github.com/mattloper/chumpy
!pip install -q --no-deps .
!sed -i 's/HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)/HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, weights_only=False)/' hmr2/models/__init__.py
%cd /content
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
sys.path.insert(0, "/content/4D-Humans")

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models
download_models(CACHE_DIR_4DHUMANS)                           # ~2.5 GB, cached for the session

# SMPL: plain-numpy files only, chumpy never imported (the fix for the broken environment)
from pose_lib.smpl import pkl_to_npz, setup_hmr2_smpl
SMPL_NPZ = f"{PROJECT_DIR}/data/smpl_neutral.npz"
if not os.path.exists(SMPL_NPZ):
    print(pkl_to_npz(f"{PROJECT_DIR}/body_models/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl", SMPL_NPZ))
setup_hmr2_smpl(SMPL_NPZ)

import torch
from pose_lib import hmr2_ft as ft, lora, metrics, stats
fk = ft.SMPLJoints(SMPL_NPZ)
print("HMR2.0 + SMPL ready")
''')

DATA = code('''
# synthetic AMASS dataset from notebook 04 (B_clean / C_paste2d / D_occ3d, train + test)
DATA = "/content/dataset"
if not os.path.exists(f"{DATA}/test/D_occ3d/labels.npz"):
    !tar -xf "{PROJECT_DIR}/data/synth_amass.tar" -C /content
# restore self-occlusion labels if notebook 06 added them in an earlier session
for c in ["B_clean", "C_paste2d", "D_occ3d"]:
    saved = f"{PROJECT_DIR}/data/test_labels_selfocc_{c}.npz"
    if os.path.exists(saved):
        shutil.copy(saved, f"{DATA}/test/{c}/labels.npz")
HAS_SELFOCC = "selfvis_joint" in np.load(f"{DATA}/test/D_occ3d/labels.npz").files
print(os.listdir(f"{DATA}/train"), "| self-occlusion labels:", HAS_SELFOCC)
''')

FRESH = code('''
def fresh_model():
    m = ft.load_model()
    gc.collect(); torch.cuda.empty_cache()
    return m
''')

# ============================================================================= 06 seeds (RQ2)
nb06 = [
    md('''
# 06 · Three-seed runs for 2D vs 3D occlusion (RQ2)

Repeats notebook 05's experiment with seeds 1 and 2 (seed 0 is already in `results/ep4_b4_s0`), with the
**same code path** (`pose_lib.hmr2_ft` is notebook 05's `finetune_lib`, unchanged by default).

Answers: *Does 3D-rendered occlusion beat 2D-pasted occlusion for fine-tuning?* A difference counts as
**robust** only if it has the same sign in all three seeds **and** the two-level bootstrap CI (resampling
seeds and test samples) excludes zero. Within-seed CIs alone (notebook 05) ignore training noise.

Optional: adds the **per-joint self-occlusion label** to the existing test set (no re-rendering, ~5 min),
so errors split into *visible / hidden by an object / hidden by the body*.

Resumable: every checkpoint and result is on Drive and skipped if present. Runtime on a T4: ~35 min per
seed for training + ~5 min evaluation.
'''),
    code('''
SEEDS      = [0, 1, 2]
CONDITIONS = ["B_clean", "C_paste2d", "D_occ3d"]
EPOCHS, BATCH, LR, N_BLOCKS = 4, 16, 1e-5, 4      # identical to notebook 05
ADD_SELFOCC = True                                 # add the self-occlusion label to the test set (needs Blender)
'''),
    SETUP, INSTALL_HMR2, DATA, FRESH,
    md("## Optional: per-joint self-occlusion labels for the test set (labels only, no rendering)"),
    code('''
if ADD_SELFOCC and not HAS_SELFOCC:
    !apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 libgl1 libegl1 > /dev/null
    ![ -d /content/blender ] || (wget -q https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz -O /content/blender.tar.xz && tar -xf /content/blender.tar.xz -C /content && mv /content/blender-4.2.3-linux-x64 /content/blender && rm /content/blender.tar.xz)
    BLENDER = "/content/blender/blender"
    N_TEST, N_PROCS, SEED0 = 1000, 3, 100          # must match notebook 04's test generation
    per = (N_TEST + N_PROCS - 1) // N_PROCS
    procs, gens = [], []
    os.makedirs("/content/selfocc", exist_ok=True)
    for p in range(N_PROCS):
        k = min(per, N_TEST - p * per); out = f"/content/selfocc/test_{p}"
        cmd = [BLENDER, "--background", "--python", f"{CODE_DIR}/blender/synth_gen.py", "--", "--model", SMPL_NPZ,
               "--poses", f"{PROJECT_DIR}/data/poses_test.npz", "--out", out, "--n", str(k), "--start", str(p * per),
               "--seed", str(SEED0 + p), "--samples", "1", "--labels-only", "--selfvis-joints"]
        procs.append(subprocess.Popen(cmd, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT)); gens.append(out)
    while any(pr.poll() is None for pr in procs):
        time.sleep(30); print(time.strftime("%H:%M"), [open(g + ".log").read().count("[synth]") for g in gens])
    from pose_lib.labels import merge_selfvis
    print(merge_selfvis(gens, [f"{DATA}/test/{c}" for c in CONDITIONS]))   # checks the samples match first
    for c in CONDITIONS:
        shutil.copy(f"{DATA}/test/{c}/labels.npz", f"{PROJECT_DIR}/data/test_labels_selfocc_{c}.npz")
    HAS_SELFOCC = True
'''),
    md("## Train (skips anything already on Drive)"),
    code('''
for seed in SEEDS:
    RUN = f"ep{EPOCHS}_b{N_BLOCKS}_s{seed}"
    MODELS = f"{PROJECT_DIR}/models/{RUN}"; os.makedirs(MODELS, exist_ok=True)
    for cond in CONDITIONS:
        ckpt = f"{MODELS}/{cond}.pt"
        if os.path.exists(ckpt):
            print(f"[skip] seed {seed} {cond}"); continue
        print(f"\\n===== seed {seed}: training on {cond} =====")
        model = fresh_model()
        hist = ft.train(model, f"{DATA}/train/{cond}", f"{DATA}/train/{cond}", fk, ckpt,
                        epochs=EPOCHS, batch_size=BATCH, lr=LR, n_blocks=N_BLOCKS, seed=seed)
        json.dump(hist, open(f"{MODELS}/{cond}_history.json", "w"), indent=1)
        del model
'''),
    md("## Evaluate every model on every test set"),
    code('''
test = {c: ft.SynthDataset(f"{DATA}/test/{c}") for c in CONDITIONS}
print({c: len(d) for c, d in test.items()}, "test samples (occ_ok only)")
names = ["pretrained"] + CONDITIONS
for seed in SEEDS:
    RUN = f"ep{EPOCHS}_b{N_BLOCKS}_s{seed}"
    MODELS, RESULTS = f"{PROJECT_DIR}/models/{RUN}", f"{PROJECT_DIR}/results/{RUN}"
    os.makedirs(RESULTS, exist_ok=True)
    for name in names:
        if name == "pretrained" and seed != SEEDS[0]:
            continue                                   # the released model doesn't depend on the seed
        todo = [t for t in CONDITIONS if not os.path.exists(f"{RESULTS}/{name}__{t}.npz")
                or (HAS_SELFOCC and "selfvis" not in np.load(f"{RESULTS}/{name}__{t}.npz").files)]
        if not todo:
            continue
        model = fresh_model()
        if name != "pretrained":
            ft.load_trainable(model, f"{MODELS}/{name}.pt")
        for t in todo:
            print(f"seed {seed} {name:>10} on {t}:", end=" ")
            np.savez(f"{RESULTS}/{name}__{t}.npz", **ft.evaluate(model, test[t], fk))
        del model
'''),
    md("## Results: mean ± std over seeds, robust comparisons"),
    code('''
import pandas as pd
R = {}
for seed in SEEDS:
    RES = f"{PROJECT_DIR}/results/ep{EPOCHS}_b{N_BLOCKS}_s{seed}"
    for n in names:
        src = RES if n != "pretrained" else f"{PROJECT_DIR}/results/ep{EPOCHS}_b{N_BLOCKS}_s{SEEDS[0]}"
        for t in CONDITIONS:
            R[(seed, n, t)] = dict(np.load(f"{src}/{n}__{t}.npz"))

rows = []
for t in CONDITIONS:
    for n in names:
        per = [metrics.summarize(R[(s, n, t)]) for s in SEEDS]
        keys = [k for k in per[0] if k != "n" and all(k in p for p in per)]
        row = {"test": t, "model": n}
        row.update(stats.mean_std_table({k: [p[k] for p in per] for k in keys}))
        rows.append(row)
df = pd.DataFrame(rows).set_index(["test", "model"])
OUTDIR = f"{PROJECT_DIR}/results/seeds_ep{EPOCHS}_b{N_BLOCKS}"; os.makedirs(OUTDIR, exist_ok=True)
df.to_csv(f"{OUTDIR}/summary_mean_std.csv"); pd.set_option("display.width", 250); df
'''),
    code('''
def cmp(a, b, t, metric="pa_mpjpe"):
    r = stats.seed_comparison([R[(s, a, t)][metric] for s in SEEDS], [R[(s, b, t)][metric] for s in SEEDS])
    verdict = "ROBUST" if r["robust"] else ("same sign, CI includes 0" if r["same_sign_all_seeds"] else "sign flips across seeds")
    return (f"{a} - {b} on {t}: {r['diff_mean']:+.1f} mm (seed sd {r['diff_std_across_seeds']:.1f}; "
            f"per seed {r['per_seed']}) [95% CI {r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] -> {verdict}")

lines = ([cmp("D_occ3d", "C_paste2d", t) for t in CONDITIONS] +
         [cmp("D_occ3d", "B_clean", t) for t in ("C_paste2d", "D_occ3d")] +
         [cmp("C_paste2d", "B_clean", t) for t in ("C_paste2d", "D_occ3d")] +
         [cmp("D_occ3d", "pretrained", t) for t in CONDITIONS])
print("\\n".join(lines)); open(f"{OUTDIR}/comparisons_seeds.txt", "w").write("\\n".join(lines))
'''),
    md('''
**How to read RQ2.** The key test is *D beats C on C's own test set* (each model has a home advantage).
The cost on clean images (D − B on B_clean) is the trade-off that motivates specialists.
'''),
    code('''
import matplotlib.pyplot as plt
bins = [(0, .2), (.2, .4), (.4, .6), (.6, 1.01)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, t in zip(axes, ["C_paste2d", "D_occ3d"]):
    for n, style in zip(names, ["k--", "C0-o", "C1-s", "C2-^"]):
        ys = np.array([[R[(s, n, t)]["pa_mpjpe"][(R[(s, n, t)]["occfrac"] >= lo) & (R[(s, n, t)]["occfrac"] < hi)].mean()
                        for lo, hi in bins] for s in SEEDS])
        ax.errorbar(range(len(bins)), ys.mean(0), yerr=ys.std(0, ddof=1) if len(SEEDS) > 1 else None,
                    fmt=style, capsize=3, label=n)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels([f"{lo:.0%}-{min(hi,1):.0%}" for lo, hi in bins])
    ax.set_title(f"test set: {t}"); ax.set_xlabel("fraction of body hidden"); ax.grid(alpha=.3)
axes[0].set_ylabel("PA-MPJPE (mm), mean ± sd over seeds"); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUTDIR}/error_vs_occlusion_seeds.png", dpi=150); plt.show()
'''),
    code('''
# Self-occlusion split (if labels were added): visible vs hidden-by-object vs hidden-by-body, D_occ3d test
if HAS_SELFOCC:
    keys = ["PA joints: visible", "PA joints: hidden by object", "PA joints: hidden by body"]
    tab = {n: stats.mean_std_table({k: [metrics.summarize(R[(s, n, "D_occ3d")]).get(k, np.nan) for s in SEEDS]
                                    for k in keys}) for n in names}
    t = pd.DataFrame(tab).T; t.to_csv(f"{OUTDIR}/selfocc_split_D_occ3d.csv"); display(t)
'''),
]

# ============================================================================= 07 experts (RQ4)
nb07 = [
    md('''
# 07 · Specialists vs one generalist (RQ4)

**Specialize.** Four LoRA experts on one frozen HMR2.0 backbone (adapters in the last 4 ViT blocks + the
SMPL head), each trained on one kind of problem, with **the same images, steps, LR and rank**:

| Expert | Rows | Data (from notebook 04's folders, no new rendering) |
|---|---|---|
| E0 Clean | 1, 16 | `B_clean` |
| E1 Occlusion | 7, 8, 10, 11 | `D_occ3d` (3D-rendered occluders) |
| E2 Quality | 20–22 | `B_clean` + blur / motion blur / low light / low resolution |
| E3 Framing | 13–15 | `B_clean` + cut-off / far / close |

**Generalist baseline** (the key comparison): one adapter with rank 4×r, so exactly the same trainable
parameters as the four experts together, trained on the union of their data for the same total steps.

**Cross-evaluation matrix:** every model on every test condition (clean, 3D occlusion, 2D paste,
quality, framing, occlusion+quality). Routing here uses the *true* condition labels (oracle); notebook 08
replaces it with the learned detector. Resumable; ~1 h on a T4.
'''),
    code('''
EXPERTS = ["E0", "E1", "E2", "E3"]
R_LORA, LORA_BLOCKS = 16, 4
EPOCHS, BATCH, LR, SEED = 4, 16, 1e-4, 0          # LoRA uses a higher LR than full fine-tuning (1e-5)
RUN = f"experts_r{R_LORA}_b{LORA_BLOCKS}_s{SEED}"
'''),
    SETUP, INSTALL_HMR2, DATA, FRESH,
    code('''
from pose_lib import experts as ex
MODELS = f"{PROJECT_DIR}/models/{RUN}"; RESULTS = f"{PROJECT_DIR}/results/{RUN}"
os.makedirs(MODELS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

def model_with_adapters():
    m = fresh_model()
    lora.inject(m, lora.hmr2_targets(n_blocks=LORA_BLOCKS), EXPERTS, r=R_LORA)
    lora.add_adapter(m, "GEN", r=lora.generalist_rank(R_LORA, len(EXPERTS)))
    print("params per expert:", lora.adapter_numel(m, "E0"), "| generalist:", lora.adapter_numel(m, "GEN"),
          "| 4 experts:", sum(lora.adapter_numel(m, e) for e in EXPERTS))
    return m

model = model_with_adapters()
'''),
    md("## Train the four experts"),
    code('''
for e in EXPERTS:
    path = f"{MODELS}/{e}.pt"
    if os.path.exists(path):
        print(f"[skip] {e}"); continue
    tr, va = ex.expert_datasets(DATA, e, seed=SEED)
    print(f"\\n===== {e}: {len(tr)} train / {len(va)} val =====")
    hist = ft.train(model, None, None, fk, path, epochs=EPOCHS, batch_size=BATCH, lr=LR, seed=SEED,
                    train_ds=tr, val_ds=va, trainable=lora.trainable_fn(e),
                    save_fn=lambda m, p, e=e: lora.save_adapter(m, e, p))
    json.dump(hist, open(f"{MODELS}/{e}_history.json", "w"), indent=1)
'''),
    md("## Train the generalist (same parameters, same total steps, union of the data)"),
    code('''
path = f"{MODELS}/GEN.pt"
if not os.path.exists(path):
    parts = [ex.expert_datasets(DATA, e, seed=SEED) for e in EXPERTS]
    tr, va = ex.ConcatSame([p[0] for p in parts]), ex.ConcatSame([p[1] for p in parts])
    hist = ft.train(model, None, None, fk, path, epochs=EPOCHS, batch_size=BATCH, lr=LR, seed=SEED,
                    train_ds=tr, val_ds=va, trainable=lora.trainable_fn("GEN"),
                    save_fn=lambda m, p: lora.save_adapter(m, "GEN", p))
    json.dump(hist, open(f"{MODELS}/GEN_history.json", "w"), indent=1)
for e in EXPERTS + ["GEN"]:
    lora.load_adapter(model, e, f"{MODELS}/{e}.pt")
lora.set_active(model, None)
'''),
    md("## Cross-evaluation matrix: every model on every condition"),
    code('''
tests = ex.test_sets(DATA)
print({k: len(v) for k, v in tests.items()})

def run_eval(name, ds, **kw):
    f = f"{RESULTS}/{name}__{tname}.npz"
    if os.path.exists(f):
        return dict(np.load(f))
    print(f"{name:>16} on {tname}:", end=" ")
    r = ft.evaluate(model, ds, fk, **kw); np.savez(f, **r); return r

dgs_w, dgs_j = ex.DGS(model, None, mode="weighted"), ex.DGS(model, None, mode="per_joint")
R = {}
for tname, ds in tests.items():
    lora.set_active(model, None);  R[("pretrained", tname)] = run_eval("pretrained", ds)
    for e in EXPERTS + ["GEN"]:
        lora.set_active(model, e); R[(e, tname)] = run_eval(e, ds)
    lora.set_active(model, None)
    R[("DGS_oracle_weighted", tname)] = run_eval("DGS_oracle_weighted", ds, predict_fn_batch=dgs_w.oracle)
    R[("DGS_oracle_perjoint", tname)] = run_eval("DGS_oracle_perjoint", ds, predict_fn_batch=dgs_j.oracle)
'''),
    code('''
import pandas as pd, matplotlib.pyplot as plt
MODELS_ROWS = ["pretrained"] + EXPERTS + ["GEN", "DGS_oracle_weighted", "DGS_oracle_perjoint"]
mat = pd.DataFrame({t: {m: R[(m, t)]["pa_mpjpe"].mean() for m in MODELS_ROWS} for t in tests}).round(1)
mat.to_csv(f"{RESULTS}/cross_eval_pa_mpjpe.csv")
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(mat.values, cmap="viridis_r"); plt.colorbar(im, label="PA-MPJPE (mm)")
ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=30, ha="right")
ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]; best = v == mat.values[:, j].min()
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="w" if v > mat.values.mean() else "k",
                fontweight="bold" if best else None)
ax.set_title("Cross-evaluation (bold = best per condition)"); plt.tight_layout()
plt.savefig(f"{RESULTS}/cross_eval.png", dpi=150); plt.show(); mat
'''),
    code('''
# RQ4: detector + specialists vs the generalist, per condition (paired bootstrap, Holm-corrected)
rows = []
for cand in ["DGS_oracle_weighted", "DGS_oracle_perjoint"]:
    for t in tests:
        m, lo, hi = stats.paired_bootstrap(R[(cand, t)]["pa_mpjpe"], R[("GEN", t)]["pa_mpjpe"])
        p = stats.bootstrap_p(R[(cand, t)]["pa_mpjpe"], R[("GEN", t)]["pa_mpjpe"])
        rows.append({"model": cand, "test": t, "diff_vs_GEN_mm": round(m, 2), "ci": f"[{lo:+.1f}, {hi:+.1f}]", "p": p})
cmp = pd.DataFrame(rows); cmp["p_holm"] = stats.holm(cmp["p"].values).round(4)
cmp["verdict"] = np.where((cmp.p_holm < 0.05) & (cmp.diff_vs_GEN_mm < 0), "specialists better",
                 np.where((cmp.p_holm < 0.05), "generalist better", "no significant difference"))
cmp.to_csv(f"{RESULTS}/rq4_vs_generalist.csv", index=False); cmp
'''),
    md('''
**Reading RQ4.** Specialists help if (a) each expert wins its home condition and (b) routed specialists
beat the generalist without losing on clean images. Oracle routing is the upper bound; notebook 08 shows how
much of it the learned detector keeps. Run with `SEED = 1, 2` to get seed spread for the final report.
'''),
]

# ============================================================================= 08 detector + fusion (RQ5, RQ6 synthetic)
nb08 = [
    md('''
# 08 · Problem detector and fusion (RQ5, RQ6 on synthetic data)

**Diagnose.** A ResNet-18 trained on Blender images predicts, from the same 256×256 crop the pose model sees:
occlusion / quality / framing probabilities, per-joint visibility, and the hidden fraction of the body.
Labels are free: they come from notebook 04's generator and from `pose_lib.degrade`.

**Fuse.** Router weights come from the detector. Fusion methods compared (RQ5):
`weighted` (one weight per expert), `per_joint` (the occlusion expert gets more say on joints predicted
hidden), `learned` (small MLP trained on a held-out split), and `top1` (hard routing), plus oracle routing.

Needs notebook 07's adapters. ~30 min on a T4.
'''),
    code('''
EXPERTS = ["E0", "E1", "E2", "E3"]
R_LORA, LORA_BLOCKS, SEED = 16, 4, 0
RUN = f"experts_r{R_LORA}_b{LORA_BLOCKS}_s{SEED}"           # notebook 07's run
DET_EPOCHS, DET_PER_SOURCE = 3, 3000
'''),
    SETUP, INSTALL_HMR2, DATA, FRESH,
    code('''
!pip install -q torchvision
from pose_lib import experts as ex, detector as dt, degrade, fusion
MODELS = f"{PROJECT_DIR}/models/{RUN}"; RESULTS = f"{PROJECT_DIR}/results/{RUN}"
model = fresh_model()
lora.inject(model, lora.hmr2_targets(n_blocks=LORA_BLOCKS), EXPERTS, r=R_LORA)
for e in EXPERTS:
    lora.load_adapter(model, e, f"{MODELS}/{e}.pt")
model.cuda().eval(); lora.set_active(model, None)
'''),
    md("## Train the detector"),
    code('''
def split(cond):
    return ft.split_train_val(f"{DATA}/train/{cond}", seed=SEED)
trB, vaB = split("B_clean"); trD, vaD = split("D_occ3d"); trC, vaC = split("C_paste2d")
Q, Fr = degrade.quality(), degrade.framing()
def sources(tr_or_va, deterministic):
    iB, iD, iC = (trB, trD, trC) if tr_or_va == "train" else (vaB, vaD, vaC)
    kw = dict(deterministic=deterministic, augment=not deterministic)
    return [ft.SynthDataset(f"{DATA}/train/B_clean", indices=iB, **kw),
            ft.SynthDataset(f"{DATA}/train/D_occ3d", indices=iD, **kw),
            ft.SynthDataset(f"{DATA}/train/C_paste2d", indices=iC, **kw),
            ft.SynthDataset(f"{DATA}/train/B_clean", indices=iB, transform=Q, transform_seed=1, **kw),
            ft.SynthDataset(f"{DATA}/train/B_clean", indices=iB, transform=Fr, transform_seed=2, **kw),
            ft.SynthDataset(f"{DATA}/train/D_occ3d", indices=iD, transform=Q, transform_seed=3, **kw)]
det_train = dt.MixtureDataset(sources("train", False), [DET_PER_SOURCE] * 6, seed=SEED)
det_val = dt.MixtureDataset(sources("val", True), [400] * 6, seed=SEED)
DET_PATH = f"{MODELS}/detector_resnet18.pt"
det = dt.Detector("resnet18").cuda()
if os.path.exists(DET_PATH):
    det.load_state_dict(torch.load(DET_PATH, map_location="cuda"))
else:
    hist = dt.train(det, det_train, det_val, DET_PATH, epochs=DET_EPOCHS, device="cuda")
    json.dump(hist, open(f"{MODELS}/detector_history.json", "w"), indent=1)
    det.load_state_dict(torch.load(DET_PATH, map_location="cuda"))
det.eval()
'''),
    code('''
# RQ6 (synthetic half): how well does the detector diagnose held-out TEST conditions?
tests = ex.test_sets(DATA)
det_test = dt.MixtureDataset(list(tests.values()), [len(v) for v in tests.values()], seed=0)
m = dt.evaluate(det, det_test, device="cuda")
json.dump(m, open(f"{RESULTS}/detector_test_metrics.json", "w"), indent=1); m
'''),
    md("## Learned fusion: fit on held-out validation crops (never the test set)"),
    code('''
FUS_PATH = f"{MODELS}/learned_fusion.pt"
if os.path.exists(FUS_PATH):
    learned = fusion.LearnedFusion(k=len(EXPERTS)); learned.load_state_dict(torch.load(FUS_PATH))
else:
    cache = ex.cache_for_fusion(model, det, det_val, experts=EXPERTS, device="cuda")
    learned, best = fusion.train_learned_fusion(cache, fk.cpu(), epochs=150)
    torch.save(learned.state_dict(), FUS_PATH)
learned.eval()
'''),
    md("## RQ5: compare fusion methods on every test condition"),
    code('''
def run_eval(name, ds, tname, **kw):
    f = f"{RESULTS}/{name}__{tname}.npz"
    if os.path.exists(f):
        return dict(np.load(f))
    print(f"{name:>22} on {tname}:", end=" ")
    r = ft.evaluate(model, ds, fk, **kw); np.savez(f, **r); return r

variants = {"DGS_det_weighted": ex.DGS(model, det, "weighted"), "DGS_det_perjoint": ex.DGS(model, det, "per_joint"),
            "DGS_det_learned": ex.DGS(model, det, "learned", learned=learned), "DGS_det_top1": ex.DGS(model, det, "top1")}
R = {}
for tname, ds in tests.items():
    for name, dgs in variants.items():
        R[(name, tname)] = run_eval(name, ds, tname, predict_fn=dgs)
    for ref in ["pretrained", "GEN", "DGS_oracle_weighted", "DGS_oracle_perjoint"] + EXPERTS:
        f = f"{RESULTS}/{ref}__{tname}.npz"
        if os.path.exists(f):
            R[(ref, tname)] = dict(np.load(f))
'''),
    code('''
import pandas as pd
rows_ = ["pretrained", "GEN", "E0", "E1", "E2", "E3", "DGS_oracle_weighted", "DGS_oracle_perjoint"] + list(variants)
tab = pd.DataFrame({t: {m: R[(m, t)]["pa_mpjpe"].mean() for m in rows_ if (m, t) in R} for t in tests}).round(1)
tab["mean"] = tab.mean(1).round(1)
tab.to_csv(f"{RESULTS}/rq5_fusion_pa_mpjpe.csv"); tab
'''),
    code('''
# Is the best detector-routed variant better than the generalist? (RQ4 with a real detector)
best = tab.loc[list(variants), "mean"].idxmin(); print("best fusion:", best)
rows = []
for t in tests:
    mdiff, lo, hi = stats.paired_bootstrap(R[(best, t)]["pa_mpjpe"], R[("GEN", t)]["pa_mpjpe"])
    rows.append({"test": t, f"{best} - GEN (mm)": round(mdiff, 2), "95% CI": f"[{lo:+.1f}, {hi:+.1f}]",
                 "p": stats.bootstrap_p(R[(best, t)]["pa_mpjpe"], R[("GEN", t)]["pa_mpjpe"])})
c = pd.DataFrame(rows); c["p_holm"] = stats.holm(c.p.values).round(4)
c.to_csv(f"{RESULTS}/rq4_detector_vs_generalist.csv", index=False); c
'''),
]

# ============================================================================= 09 real video (RQ1, RQ6 real, Blender)
nb09 = [
    md('''
# 09 · Real phone video: failure analysis (RQ1), detector on real video (RQ6), Blender export

1. **Video → SMPL without PHALP.** YOLO person detection + a simple IoU tracker + HMR2.0 on the official
   ViTDetDataset crops. This is the fix for the broken notebook 01 environment (Week 1 task).
2. Runs, on the **same crops**: the released HMR2.0 and (if notebooks 07/08 ran) the full
   Diagnose → Specialize → Fuse model. Also converts teammates' outputs (Multi-HMR, MediaPipe, PARE,
   WHAM/GVHMR) into the same standard files.
3. Builds the RQ1 failure table: proxy metrics per clip × model, grouped by the 29-row test plan.
4. RQ6: does the detector trained on Blender images flag the right rows on real video?

Clips: `CS747_project/videos/rows/rowNN_<desc>_<k>.mp4` (the test-plan naming). Results:
`results/real/<clip>__<model>.npz`. Everything is skipped if it already exists.
'''),
    code('''
TARGET_H, MAX_SECONDS = 720, 60
EXPERTS = ["E0", "E1", "E2", "E3"]
RUN = "experts_r16_b4_s0"                       # notebook 07/08 run (optional)
LEGACY = {"clipA": 9, "clipB": 7}               # clips recorded before the naming scheme -> test-plan row
'''),
    SETUP, INSTALL_HMR2,
    code('''
!pip install -q ultralytics torchvision
from pose_lib import video, convert, experts as ex, detector as dt, fusion, smoothing
from pose_lib.format import PoseSeq, parse_name
from pose_lib.smpl import SMPLJointsNP
fknp = SMPLJointsNP(SMPL_NPZ)
REAL = f"{PROJECT_DIR}/results/real"; os.makedirs(REAL, exist_ok=True)

model, model_cfg = video.load_hmr2_with_cfg(); model.cuda().eval()
predictors = {"hmr2": video.base_predictor(model)}
MODELS = f"{PROJECT_DIR}/models/{RUN}"
if all(os.path.exists(f"{MODELS}/{e}.pt") for e in EXPERTS) and os.path.exists(f"{MODELS}/detector_resnet18.pt"):
    lora.inject(model, lora.hmr2_targets(4), EXPERTS, r=16)
    for e in EXPERTS: lora.load_adapter(model, e, f"{MODELS}/{e}.pt")
    det = dt.Detector("resnet18", pretrained=False).cuda().eval()
    det.load_state_dict(torch.load(f"{MODELS}/detector_resnet18.pt", map_location="cuda"))
    predictors["dgs_weighted"] = ex.DGS(model, det, "weighted")
    predictors["dgs_perjoint"] = ex.DGS(model, det, "per_joint")
    if os.path.exists(f"{MODELS}/learned_fusion.pt"):
        lf = fusion.LearnedFusion(4); lf.load_state_dict(torch.load(f"{MODELS}/learned_fusion.pt"))
        predictors["dgs_learned"] = ex.DGS(model, det, "learned", learned=lf.eval())
model.cuda()
print("predictors:", list(predictors))
yolo = video.load_detector_yolo("yolo11n.pt")
'''),
    md("## 1. Run every clip (Week 1 check: start with one clip)"),
    code('''
clips = sorted(glob.glob(f"{PROJECT_DIR}/videos/rows/row*.mp4"))
clips += [p for p in sorted(glob.glob(f"{PROJECT_DIR}/videos/*.mp4")) if os.path.splitext(os.path.basename(p))[0] in LEGACY]
print(len(clips), "clips")
TIMING = f"{REAL}/timing.json"; timing = json.load(open(TIMING)) if os.path.exists(TIMING) else {}
for p in clips:
    clip = os.path.splitext(os.path.basename(p))[0]
    todo = {k: v for k, v in predictors.items() if not os.path.exists(f"{REAL}/{clip}__{k}.npz")}
    if not todo:
        continue
    _, t = video.run(p, REAL, todo, model_cfg, fknp, yolo=yolo, target_h=TARGET_H, max_seconds=MAX_SECONDS)
    timing[clip] = t; json.dump(timing, open(TIMING, "w"), indent=1)
'''),
    md('''
## 2. Teammates' outputs → standard files
Put raw outputs here (any that exist are converted; the rest are skipped):
```
results/real_raw/multihmr/<clip>_smpl.npz           Madeeha (Multi-HMR, same keys as notebook 01)
results/real_raw/mediapipe_<lite|full|heavy>/<clip>_<size>.npz   Nishitha (world, raw, visibility, found)
results/real_raw/pare/<clip>/pare_output.pkl        Gnanitha (PARE demo output)
results/real_raw/gvhmr/<clip>/hmr4d_results.pt      Salma (GVHMR demo output)
results/real_raw/wham/<clip>/wham_output.pkl        Salma (WHAM demo output)
```
'''),
    code('''
RAW = f"{PROJECT_DIR}/results/real_raw"
import cv2
def clip_info(clip):
    vids = glob.glob(f"{PROJECT_DIR}/videos/rows/{clip}.mp4") + glob.glob(f"{PROJECT_DIR}/videos/{clip}.mp4")
    if not vids: return 30.0, None
    cap = cv2.VideoCapture(vids[0]); fps = cap.get(cv2.CAP_PROP_FPS) or 30; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    return fps, n
done = []
for f in glob.glob(f"{RAW}/multihmr/*_smpl.npz"):
    clip = os.path.basename(f)[:-9]; out = f"{REAL}/{clip}__multihmr.npz"
    if not os.path.exists(out):
        convert.from_smpl_npz(f, fknp, model="multihmr", clip=clip).save(out); done.append(out)
for f in glob.glob(f"{RAW}/mediapipe_*/*.npz"):
    size = os.path.basename(os.path.dirname(f)).split("_", 1)[1]
    clip = os.path.basename(f)[:-4].rsplit("_", 1)[0]; out = f"{REAL}/{clip}__mediapipe_{size}.npz"
    if not os.path.exists(out) and "world" in np.load(f).files:
        fps, n = clip_info(clip)
        convert.from_mediapipe(f, fps=fps, model=f"mediapipe_{size}", clip=clip, n_frames=n).save(out); done.append(out)
for name, pattern, fn in [("pare", "pare/*/pare_output.pkl", convert.from_pare),
                          ("gvhmr", "gvhmr/*/hmr4d_results.pt", convert.from_gvhmr),
                          ("wham", "wham/*/wham_output.pkl", convert.from_wham)]:
    for f in glob.glob(f"{RAW}/{pattern}"):
        clip = os.path.basename(os.path.dirname(f)); out = f"{REAL}/{clip}__{name}.npz"
        if not os.path.exists(out):
            fps, n = clip_info(clip)
            fn(f, fknp, fps=fps, clip=clip, n_frames=n).save(out); done.append(out)
print(len(done), "converted")
'''),
    md("## 3. Temporal baseline: Savitzky–Golay smoothing of every output (`+sg`)"),
    code('''
for f in glob.glob(f"{REAL}/*__*.npz"):
    clip, m, _ = parse_name(f)
    if m.endswith("+sg"): continue
    out = f"{REAL}/{clip}__{m}+sg.npz"
    if not os.path.exists(out):
        s = PoseSeq.load(f); smoothing.smooth_seq(s, fk=fknp if s.is_smpl else None).save(out)
'''),
    md("## 4. RQ1 failure table"),
    code('''
import pandas as pd, re
GROUP = {**{r: "People" for r in range(1, 7)}, **{r: "Occlusion" for r in range(7, 12)},
         **{r: "Body position" for r in range(12, 16)}, **{r: "Camera" for r in range(16, 20)},
         **{r: "Video quality" for r in range(20, 25)}}
seqs = {}
for f in sorted(glob.glob(f"{REAL}/*__*.npz")):
    clip, m, row = parse_name(f)
    if re.search(r"_t\\d+$", clip):              # secondary tracks (other people) are kept but not tabulated
        continue
    row = row if row > 0 else LEGACY.get(clip, -1)
    seqs[(clip, m)] = (PoseSeq.load(f), row)
rows = []
for (clip, m), (s, row) in seqs.items():
    r = metrics.proxy_report(s); r.update(row=row, group=GROUP.get(row, "?"))
    others = [o for (c2, m2), (o, _) in seqs.items() if c2 == clip and m2 != m and not m2.endswith("+sg")]
    r["disagreement_mm"] = np.nanmean([metrics.agreement(s, o) for o in others]) if others else np.nan
    rows.append(r)
P = pd.DataFrame(rows); P.to_csv(f"{REAL}/proxy_metrics_all.csv", index=False)
cols = ["detection_rate", "jitter_mm", "rot_jitter_deg", "pose_jumps", "lr_swaps", "bone_cv", "foot_skate_mm",
        "low_conf_joints", "disagreement_mm"]
by_group = P.groupby(["group", "model"])[cols].mean().round(3); by_group.to_csv(f"{REAL}/rq1_by_group.csv"); by_group
'''),
    code('''
# Flags: which failure each model shows on which row (thresholds are deliberately simple; tune once on rows 1/16)
TH = {"detection_rate": ("<", 0.9), "pose_jumps": (">", 0.05), "lr_swaps": (">", 0), "bone_cv": (">", 0.05),
      "jitter_mm": (">", 2 * P[P.row.isin([1, 16])]["jitter_mm"].median() if P.row.isin([1, 16]).any() else 30),
      "disagreement_mm": (">", 80)}
def flags(r):
    out = []
    for k, (op, v) in TH.items():
        x = r[k]
        if np.isfinite(x) and ((op == "<" and x < v) or (op == ">" and x > v)):
            out.append(k)
    return ", ".join(out)
P["flags"] = P.apply(flags, axis=1)
fail = P[~P.model.str.endswith("+sg")].pivot_table(index=["row", "clip"], columns="model", values="flags", aggfunc="first")
fail.to_csv(f"{REAL}/rq1_failure_table.csv"); fail
'''),
    code('''
# Two-rater visual score sheet (fill in, then merge with the proxy table for the report)
metrics.rating_template(sorted({c for c, _ in seqs}), sorted({m for _, m in seqs if not m.endswith("+sg")}),
                        f"{REAL}/visual_ratings_template.csv")
'''),
    md("## 5. RQ6: does the Blender-trained detector flag the right rows on real video?"),
    code('''
def frame_scores(model_name, key):
    xs, ys = [], []
    for (clip, m), (s, row) in seqs.items():
        if m != model_name or f"extra_{key}" not in s.extra: continue
        xs.append(s.extra[f"extra_{key}"]); ys.append(np.full(len(s.extra[f"extra_{key}"]), row))
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (None, None)

POS = {"occlusion": range(7, 12), "quality": range(20, 23), "framing": range(12, 16)}
NEG = [1, 16]
res = {}
src = "dgs_weighted" if any(m == "dgs_weighted" for _, m in seqs) else None
if src:
    cond, row = frame_scores(src, "dgs_cond")          # (frames, 3): occlusion, quality, framing
    for i, name in enumerate(dt.COND_NAMES):
        mask = np.isin(row, list(POS[name]) + NEG)
        res[f"detector AUC {name}"] = dt.auc(np.isin(row[mask], list(POS[name])), cond[mask, i])
for name, key in [("quality", "chk_cond_quality"), ("framing", "chk_cond_framing")]:
    x, row = frame_scores("hmr2", key)
    if x is not None:
        mask = np.isin(row, list(POS[name]) + NEG)
        res[f"cheap-check AUC {name}"] = dt.auc(np.isin(row[mask], list(POS[name])), x[mask])
json.dump(res, open(f"{REAL}/rq6_real_detector_auc.json", "w"), indent=1); res
'''),
    md('''
## 6. Blender: animate a character with any result file
The standard files keep notebook 01's keys, so notebooks 02/03 and `blender/smplx_mocap.py` read them as is.
'''),
    code('''
CLIP, MODEL = "clipB", "dgs_perjoint+sg"          # pick from results/real
motion = f"{REAL}/{CLIP}__{MODEL}.npz"
!apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 libgl1 libegl1 > /dev/null
![ -d /content/blender ] || (wget -q https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz -O /content/blender.tar.xz && tar -xf /content/blender.tar.xz -C /content && mv /content/blender-4.2.3-linux-x64 /content/blender && rm /content/blender.tar.xz)
OUT = f"{PROJECT_DIR}/outputs/{CLIP}__{MODEL}"
!/content/blender/blender --background --python "{CODE_DIR}/blender/smplx_mocap.py" -- --model "{PROJECT_DIR}/body_models/smplx/SMPLX_NEUTRAL.npz" --motion "{motion}" --blend "{OUT}.blend" --video "{OUT}.mp4" --max-frames 150 2>&1 | grep "smplx_mocap\\|Error"
'''),
]

# ============================================================================= 10 RQ3
nb10 = [
    md('''
# 10 · Choosing hard data adaptively (RQ3)

*Does adaptively selected hard data beat random selection and hard-example mining at the same data budget?*

The selector is a multi-armed bandit (multiplicative weights; one-step RL with no state). Arms are
condition bins of the 3D-occlusion pool (hidden-body fraction 5–20 %, 20–40 %, 40–60 %, 60–100 %). Each
round it probes every arm with the current model and shifts the data budget toward the bins with the
highest loss. Baselines: uniform random, and hard-example mining (score 4× candidates, keep the hardest).

Same for every strategy: images per round, rounds, optimiser steps, LoRA rank, starting point (the
released HMR2.0 with a fresh adapter). Probing and mining cost forward passes only, never training images.
~1 h on a T4 for 3 strategies × 3 seeds.

Live Blender generation would only change where a bin's images come from (`ArmPool.sample`); with the
existing pool the whole comparison fits in one session.
'''),
    code('''
STRATEGIES, SEEDS = ["random", "hem", "bandit"], [0, 1, 2]
ROUNDS, ROUND_SIZE, STEPS_PER_ROUND = 8, 250, 60      # 2,000 training images, 480 steps per run
R_LORA, LORA_BLOCKS, LR, BATCH = 16, 4, 1e-4, 16
'''),
    SETUP, INSTALL_HMR2, DATA, FRESH,
    code('''
from pose_lib import selector
OUT = f"{PROJECT_DIR}/results/rq3_r{R_LORA}_rounds{ROUNDS}x{ROUND_SIZE}"; os.makedirs(OUT, exist_ok=True)
tr_idx, _ = ft.split_train_val(f"{DATA}/train/D_occ3d", seed=0)
pool_ds = ft.SynthDataset(f"{DATA}/train/D_occ3d", indices=tr_idx, augment=True)
score_ds = ft.SynthDataset(f"{DATA}/train/D_occ3d", indices=tr_idx, augment=False)
arms, names = selector.occfrac_arms(pool_ds.L["occfrac"])
pool = selector.ArmPool([np.intersect1d(a, pool_ds.idx) for a in arms], names)
print({n: len(a) for n, a in zip(pool.names, pool.arms)})
tests = {c: ft.SynthDataset(f"{DATA}/test/{c}") for c in ["D_occ3d", "C_paste2d", "B_clean"]}
'''),
    code('''
for seed in SEEDS:
    for strat in STRATEGIES:
        tag = f"{strat}_s{seed}"
        if all(os.path.exists(f"{OUT}/{tag}__{t}.npz") for t in tests):
            print("[skip]", tag); continue
        model = fresh_model()
        lora.inject(model, lora.hmr2_targets(LORA_BLOCKS), [tag], r=R_LORA)
        torch.manual_seed(seed)
        trainer = selector.AdapterTrainer(model, tag, pool_ds, fk, lr=LR, batch_size=BATCH,
                                          steps_per_round=STEPS_PER_ROUND, device="cuda", seed=seed, score_ds=score_ds)
        log = selector.run(strat, pool, rounds=ROUNDS, round_size=ROUND_SIZE, train_on=trainer.train_on,
                           loss_of=trainer.loss_of, seed=seed)
        occ = pool_ds.L["occfrac"]
        json.dump({"bandit_probs": log["bandit_probs"], "arm_names": log["arm_names"],
                   "mean_occfrac_per_round": [float(occ[c].mean()) for c in log["chosen"]]},
                  open(f"{OUT}/{tag}_log.json", "w"), indent=1)
        lora.set_active(model, tag)
        for t, ds in tests.items():
            print(f"{tag} on {t}:", end=" "); np.savez(f"{OUT}/{tag}__{t}.npz", **ft.evaluate(model, ds, fk))
        del model, trainer
'''),
    code('''
import pandas as pd, matplotlib.pyplot as plt
R = {(s, sd, t): dict(np.load(f"{OUT}/{s}_s{sd}__{t}.npz")) for s in STRATEGIES for sd in SEEDS for t in tests}
rows = []
for t in tests:
    for s in STRATEGIES:
        per = [metrics.summarize(R[(s, sd, t)]) for sd in SEEDS]
        keys = [k for k in per[0] if k != "n" and all(k in p for p in per)]
        rows.append({"test": t, "strategy": s, **stats.mean_std_table({k: [p[k] for p in per] for k in keys})})
tab = pd.DataFrame(rows).set_index(["test", "strategy"]); tab.to_csv(f"{OUT}/rq3_summary.csv"); tab
'''),
    code('''
lines = []
for t in tests:
    for other in ["random", "hem"]:
        r = stats.seed_comparison([R[("bandit", sd, t)]["pa_mpjpe"] for sd in SEEDS],
                                  [R[(other, sd, t)]["pa_mpjpe"] for sd in SEEDS])
        lines.append(f"bandit - {other} on {t}: {r['diff_mean']:+.1f} mm per seed {r['per_seed']} "
                     f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] robust={r['robust']}")
print("\\n".join(lines)); open(f"{OUT}/rq3_comparisons.txt", "w").write("\\n".join(lines))
'''),
    code('''
# What did the bandit choose? (arm probabilities per round, seed 0)
lg = json.load(open(f"{OUT}/bandit_s0_log.json"))
p = np.array(lg["bandit_probs"])
plt.figure(figsize=(6, 3.5)); plt.stackplot(range(1, len(p) + 1), p.T, labels=lg["arm_names"])
plt.xlabel("round"); plt.ylabel("share of new data"); plt.legend(loc="upper left", fontsize=8); plt.title("Bandit allocation")
plt.tight_layout(); plt.savefig(f"{OUT}/bandit_allocation.png", dpi=150); plt.show()
'''),
]

# ============================================================================= 11 stock clips for the 29-row plan
nb11 = [
    md('''
# 11 · Stock clips for the test plan (rows 1–24) + fit check

Downloads free Pexels clips (Pexels license: free to use, no attribution required; sources are listed in
`videos/rows/_sources.csv` anyway), makes two edited clips (row 13: legs out of frame via a virtual camera;
row 22: 480p re-encode), then checks every clip: people per frame, how much of the frame the person fills,
whether the box touches the border, brightness. Contact sheets go to `videos/rows/_preview/`.

Rows 7 and 9 are the team's own clips (`clipB`, `clipA`); rows 15 and 19 have no stock clip yet; rows 25–29
are Blender checks. After this notebook, re-run notebook 09: it only processes clips it hasn't seen.
'''),
    SETUP,
    code('''
!pip install -q ultralytics
import pandas as pd
from pose_lib import clips as C
from ultralytics import YOLO
yolo = YOLO("yolo11n.pt")
ROWS = f"{PROJECT_DIR}/videos/rows"; PREV = f"{ROWS}/_preview"
os.makedirs(PREV, exist_ok=True)
'''),
    md("## 1. Download"),
    code('''
src, failed = [], []
jobs = [(r, pid, d, 1) for r, (pid, d) in C.PEXELS.items()] + \\
       [(r, pid, d, 2 + i) for r, ex in C.EXTRA.items() for i, (pid, d) in enumerate(ex)]
for row, pid, desc, k in jobs:
    dst = f"{ROWS}/{C.name(row, desc, k)}"
    try:
        C.download(pid, dst); src.append({"file": os.path.basename(dst), "row": row, "pexels_id": pid,
                                          "url": f"https://www.pexels.com/video/{pid}/"})
        print("ok  ", os.path.basename(dst), f"{os.path.getsize(dst)/1e6:.1f} MB")
    except Exception as e:
        failed.append((row, pid, str(e)[:120])); print("FAIL", row, pid, e)
pd.DataFrame(src).to_csv(f"{ROWS}/_sources.csv", index=False)
print(len(src), "downloaded,", len(failed), "failed")
'''),
    md("## 2. Edited clips (rows 13 and 22)"),
    code('''
r16 = f"{ROWS}/{C.name(16, 'tripod')}"; r1 = f"{ROWS}/{C.name(1, 'slowwalk')}"
e13 = f"{ROWS}/{C.name(13, 'upperbody', edit=True)}"; e22 = f"{ROWS}/{C.name(22, 'lowres', edit=True)}"
if os.path.exists(r16) and not os.path.exists(e13): C.upper_body_cam(r16, e13, yolo)
if os.path.exists(r1) and not os.path.exists(e22): C.to_480p(r1, e22)
'''),
    md("## 3. Fit check + contact sheets"),
    code('''
rows = []
for p in sorted(glob.glob(f"{ROWS}/row*.mp4")):
    n = os.path.basename(p)[:-4]
    r = C.check(p, yolo, sheet=f"{PREV}/{n}.jpg"); r["clip"] = n; rows.append(r)
chk = pd.DataFrame(rows).set_index("clip"); chk.to_csv(f"{ROWS}/_check.csv"); chk
'''),
]

# ============================================================================= 12 3DPW (real video with ground truth)
nb12 = [
    md('''
# 12 · 3DPW: real outdoor video with 3D ground truth (RQ1 in mm, RQ2/RQ4/RQ5 on real data)

Every model from notebooks 06–08 on the 3DPW **test** split, on identical crops:
released HMR2.0 · B/C/D full fine-tunes (seed 0, RQ2) · experts E0–E3 · generalist GEN (RQ4) ·
DGS weighted / per-joint / learned / top-1 (RQ5). Errors on the 22 SMPL body joints, MPJPE and PA-MPJPE in
mm; comparisons use a bootstrap over **sequences** (frames within a sequence are correlated).

Protocol notes (say these in the report): crops come from the GT joints projected into the image; GT joints
use the neutral SMPL with the GT shape (no gendered models), so absolute numbers are not comparable to
papers (HMR2.0 reports 54.3 PA-MPJPE with the H36M-14 regressor). Comparisons between our models are fair.

3DPW license: non-commercial research only, cite von Marcard et al., ECCV 2018, do not redistribute the raw
data — it stays on the Colab disk; only our results go to Drive.
'''),
    code('''
STRIDE, BATCH, WORKERS = 2, 48, 8
RUN = "experts_r16_b4_s0"; RQ2_RUN = "ep4_b4_s0"
EXPERTS = ["E0", "E1", "E2", "E3"]
'''),
    SETUP, INSTALL_HMR2,
    md("## 1. Download 3DPW to the Colab disk (license accepted on the website)"),
    code('''
from pose_lib import threedpw as tp, experts as ex, detector as dt, fusion
ROOT = "/content/3dpw"; os.makedirs(ROOT, exist_ok=True)
for f in tp.FILES:
    if not os.path.exists(f"{ROOT}/{f}"):
        !wget -q -c "{tp.URL}/{f}" -O "{ROOT}/{f}"
    print(f, f"{os.path.getsize(f'{ROOT}/{f}')/1e9:.2f} GB")
if not os.path.exists(f"{ROOT}/sequenceFiles/test"):
    !cd {ROOT} && unzip -oq sequenceFiles.zip
test_seqs = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{ROOT}/sequenceFiles/test/*.pkl"))
missing = [s for s in test_seqs if not os.path.exists(f"{ROOT}/imageFiles/{s}")]
if missing:
    pats = " ".join(f'"imageFiles/{s}/*"' for s in missing)
    !cd {ROOT} && unzip -oq imageFiles.zip {pats}
print(len(test_seqs), "test sequences")
'''),
    md("## 2. Samples (GT joints + crop boxes)"),
    code('''
import cv2
from pose_lib.smpl import SMPLJointsNP
fknp = SMPLJointsNP(SMPL_NPZ)
_wh = {}
def img_wh(seq):
    if seq not in _wh:
        h, w = cv2.imread(f"{ROOT}/imageFiles/{seq}/image_00000.jpg").shape[:2]; _wh[seq] = (w, h)
    return _wh[seq]
S = tp.samples(ROOT, "test", stride=STRIDE, fknp=fknp, img_wh=img_wh)
print(len(S), "samples from", len({s['seq'] for s in S}), "sequences")
'''),
    md("## 3. Models"),
    code('''
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from pose_lib import video
model, model_cfg = video.load_hmr2_with_cfg()
MODELS = f"{PROJECT_DIR}/models/{RUN}"
lora.inject(model, lora.hmr2_targets(4), EXPERTS, r=16)
lora.add_adapter(model, "GEN", r=lora.generalist_rank(16, len(EXPERTS)))
for e in EXPERTS + ["GEN"]: lora.load_adapter(model, e, f"{MODELS}/{e}.pt")
model.cuda().eval()
det = dt.Detector("resnet18", pretrained=False).cuda().eval()
det.load_state_dict(torch.load(f"{MODELS}/detector_resnet18.pt", map_location="cuda"))
lf = fusion.LearnedFusion(4); lf.load_state_dict(torch.load(f"{MODELS}/learned_fusion.pt")); lf = lf.cuda().eval()
rq2 = {}
for c in ["B_clean", "C_paste2d", "D_occ3d"]:
    m = ft.load_model(); ft.load_trainable(m, f"{PROJECT_DIR}/models/{RQ2_RUN}/{c}.pt"); rq2[c] = m.cuda().eval()
print("models ready;", torch.cuda.memory_allocated() / 1e9, "GB")
'''),
    md("## 4. Inference (resumable: saved to Drive at the end)"),
    code('''
OUT = f"{PROJECT_DIR}/results/3dpw"; os.makedirs(OUT, exist_ok=True)
PRED = f"{OUT}/pred_joints_stride{STRIDE}.npz"

class Crops(torch.utils.data.Dataset):
    def __len__(self): return len(S)
    def __getitem__(self, i):
        s = S[i]; img = cv2.imread(s["img"])
        return torch.as_tensor(ViTDetDataset(model_cfg, img, s["box"][None])[0]["img"]), i

def joints(p):
    return fk(p["global_orient"].float(), p["body_pose"].float(), p["betas"].float()).cpu().numpy()

if not os.path.exists(PRED):
    acc, aux = {}, {k: [] for k in ("cond", "occfrac", "router")}
    dl = torch.utils.data.DataLoader(Crops(), batch_size=BATCH, num_workers=WORKERS)
    fk = fk.cuda(); t0 = time.time()
    for bi, (img, idx) in enumerate(dl):
        img = img.cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = {}
            lora.set_active(model, None); out["hmr2"] = ft.predict(model, img)
            preds = []
            for e in EXPERTS:
                lora.set_active(model, e); p = {k: v.float() for k, v in ft.predict(model, img).items()}
                preds.append(p); out[e] = p
            lora.set_active(model, "GEN"); out["GEN"] = ft.predict(model, img); lora.set_active(model, None)
            d = dt.predict(det, img.float()); w = dt.router_weights(d["cond"])
            out["dgs_weighted"] = fusion.fuse_weighted(preds, w)
            out["dgs_perjoint"] = fusion.fuse_per_joint(preds, w, d["vis"], occ_index=1)
            out["dgs_learned"] = fusion.fuse_joint_weights(preds, lf(d["cond"].float(), d["vis"].float(), d["occfrac"].float()), project="gs")
            out["dgs_top1"] = fusion.fuse_top1(preds, w)
            for c, m in rq2.items(): out[c] = ft.predict(m, img)
        for k, p in out.items(): acc.setdefault(k, []).append(joints(p))
        aux["cond"].append(d["cond"].float().cpu().numpy()); aux["occfrac"].append(d["occfrac"].float().cpu().numpy())
        aux["router"].append(w.float().cpu().numpy())
        if bi % 50 == 0: print(f"{bi * BATCH}/{len(S)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(PRED, **{k: np.concatenate(v).astype(np.float32) for k, v in acc.items()},
             **{"aux_" + k: np.concatenate(v) for k, v in aux.items()},
             gt=np.stack([s["gt"] for s in S]), seq=np.array([s["seq"] for s in S]),
             person=np.array([s["person"] for s in S]), frame=np.array([s["frame"] for s in S]))
    print("saved", PRED, f"{time.time() - t0:.0f}s")
'''),
    md("## 5. Results"),
    code('''
import pandas as pd
Z = np.load(PRED); gt, seq = Z["gt"], Z["seq"]
NAMES = ["hmr2", "B_clean", "C_paste2d", "D_occ3d"] + EXPERTS + ["GEN", "dgs_weighted", "dgs_perjoint", "dgs_learned", "dgs_top1"]
E = {}
for n in NAMES:
    e, pa = metrics.joint_errors(Z[n], gt); E[n] = (e.mean(1), pa.mean(1))
tab = pd.DataFrame({n: {"MPJPE": E[n][0].mean(), "PA-MPJPE": E[n][1].mean()} for n in NAMES}).T.round(1)
tab["n"] = len(gt); tab.to_csv(f"{OUT}/summary.csv"); tab
'''),
    code('''
# Paired comparisons, bootstrap over sequences, Holm-corrected
pairs = [("D_occ3d", "C_paste2d"), ("D_occ3d", "B_clean"), ("C_paste2d", "B_clean"), ("B_clean", "hmr2"),
         ("GEN", "hmr2"), ("dgs_learned", "hmr2"), ("dgs_weighted", "hmr2"), ("dgs_perjoint", "hmr2"),
         ("dgs_learned", "GEN"), ("dgs_weighted", "GEN"), ("E1", "E0"), ("dgs_learned", "dgs_weighted")]
rows = []
for a, b in pairs:
    for k, metric in [(1, "PA-MPJPE"), (0, "MPJPE")]:
        m, lo, hi, p = tp.cluster_bootstrap(E[a][k], E[b][k], seq)
        rows.append({"comparison": f"{a} - {b}", "metric": metric, "diff_mm": round(m, 2), "ci": f"[{lo:+.1f}, {hi:+.1f}]", "p": p})
cmp = pd.DataFrame(rows)
for metric in ["PA-MPJPE", "MPJPE"]:
    mk = cmp.metric == metric; cmp.loc[mk, "p_holm"] = stats.holm(cmp.loc[mk, "p"].values).round(4)
cmp.to_csv(f"{OUT}/comparisons.csv", index=False); cmp
'''),
    code('''
# Per sequence (PA-MPJPE) and: does the occlusion expert help where the detector sees occlusion? (RQ6 on real)
per_seq = pd.DataFrame({n: pd.Series(E[n][1]).groupby(seq).mean() for n in ["hmr2", "D_occ3d", "GEN", "dgs_learned", "E1"]}).round(1)
per_seq["frames"] = pd.Series(seq).value_counts(); per_seq.to_csv(f"{OUT}/per_sequence_pa.csv"); display(per_seq)
occ = Z["aux_occfrac"]; bins = [0, 0.05, 0.15, 0.3, 1.01]
b = np.digitize(occ, bins) - 1
rows = []
for i in range(len(bins) - 1):
    m = b == i
    if m.sum() < 50: continue
    rows.append({"detector occ fraction": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "frames": int(m.sum()),
                 **{n: round(E[n][1][m].mean(), 1) for n in ["hmr2", "E0", "E1", "GEN", "dgs_learned"]}})
occ_tab = pd.DataFrame(rows); occ_tab.to_csv(f"{OUT}/by_detector_occlusion.csv", index=False); occ_tab
'''),
]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, cells in [("06_seeds_rq2", nb06), ("07_experts_rq4", nb07), ("08_detector_fusion_rq5_rq6", nb08),
                        ("09_real_video_rq1_rq6", nb09), ("10_selector_rq3", nb10),
                        ("11_stock_clips", nb11), ("12_3dpw_eval", nb12)]:
        with open(os.path.join(OUT, name + ".ipynb"), "w") as fh:
            json.dump(nb(cells), fh, indent=1)
        print("wrote", name)
