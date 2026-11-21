# Real World to Blender — code for the final report (CS 747, Fall 2026)

Builds on the team's notebooks 01–05 in `CS747_project/` (same Drive layout, same data, same results
folders). Notebook 05's fine-tuning code is ported unchanged into `pose_lib/hmr2_ft.py`, so the new seeds
compare directly with the existing seed-0 results in `results/ep4_b4_s0`.

## Setup (once)

1. Unzip this folder to `MyDrive/CS747_project/code/rw2b/` (so `.../code/rw2b/pose_lib/` exists).
   In Colab: upload `rw2b.zip` to `CS747_project/code/`, then `!cd "/content/drive/MyDrive/CS747_project/code" && unzip -o rw2b.zip`.
2. Open the notebooks from `code/rw2b/notebooks/` in Colab, with a GPU runtime. Each notebook installs what it needs
   and skips any step whose output is already on Drive, so a disconnect only costs the step that was running.

## Run order and what each notebook answers

| Notebook | Plan task (progress report §11) | Answers | T4 time |
|---|---|---|---|
| `09_real_video_rq1_rq6` §1 | Fix the SMPL environment; HMR2.0 end to end on one clip | Week 1 | 5 min |
| `06_seeds_rq2` | Three-seed runs for 2D vs 3D occlusion (+ self-occlusion split) | **RQ2** | ~80 min |
| `07_experts_rq4` | Build E0–E3 + generalist; cross-evaluation matrix | **RQ4** (oracle routing) | ~1 h |
| `08_detector_fusion_rq5_rq6` | Problem detector; fusion comparison | **RQ5**, RQ6 (synthetic), RQ4 with the real detector | ~30 min |
| `09_real_video_rq1_rq6` §2–6 | All models on the row clips; failure table; Blender export | **RQ1**, RQ6 (real) | ~2 min/clip |
| `10_selector_rq3` | Adaptive hard-data selection vs random vs hard-example mining | **RQ3** | ~1 h |

`06` and `07` are independent (run them in parallel on two accounts). `08` needs `07`. `09` works with the
released HMR2.0 alone and adds the full model once `07`/`08` exist. Run `07`/`08` with `SEED = 1, 2` too
if there is time, to report seed spread for RQ4.

## What is where

```
pose_lib/            shared library (all of it CPU-tested, see tests/)
  format.py          standard per-clip result file <clip>__<model>.npz (keeps notebook 01's keys, so 02/03 read it)
  convert.py         HMR2.0 / Multi-HMR / MediaPipe / PARE / WHAM / GVHMR outputs -> standard file
  smpl.py            chumpy-free SMPL (.pkl -> .npz, plain .pkl for HMR2.0), forward kinematics
  metrics.py         MPJPE / PA-MPJPE with visible / hidden-by-object / hidden-by-body split;
                     real-video proxies: jitter, rotation jitter, bone-length CV, foot skate, L/R swaps,
                     pose jumps, gaps, cross-model disagreement; two-rater score sheet
  stats.py           paired bootstrap, two-level (seed x sample) bootstrap, Holm correction
  smoothing.py       Savitzky-Golay on positions and on (sign-fixed) quaternions, gap-aware
  hmr2_ft.py         notebook 05's fine-tune/eval + transforms, LoRA training, oracle routing
  lora.py            several named LoRA adapters on one frozen backbone
  degrade.py         E2 quality (blur, motion blur, low light, low-res, JPEG) and E3 framing
                     (cut-off, far, close) made from clean renders, with exact label updates
  detector.py        learned problem detector (occlusion / quality / framing + per-joint visibility)
  fusion.py          weighted, per-joint and learned fusion (rotation averaging on SO(3)), top-1
  experts.py         expert datasets, test grid, the full Diagnose -> Specialize -> Fuse model (DGS)
  conditions.py      cheap per-frame checks on real video: blur, brightness, cut-off, scale, camera motion
  selector.py        RQ3: bandit (multiplicative weights) vs random vs hard-example mining
  video.py           phone video -> standard files: YOLO + IoU tracker + HMR2.0 (no PHALP/detectron2)
  labels.py          adds the self-occlusion label to the existing test set without re-rendering
blender/             notebook 04's scripts (from the copy that built synth_amass.tar) + [selfocc] additions
notebooks/           06-10 (built by tools/build_notebooks.py)
tests/               pytest suite + dryrun_notebooks.py (runs 06-10 end to end on CPU with a toy model)
```

## Clip naming for the 29-row test plan

`videos/rows/rowNN_<desc>_<k>.mp4`, e.g. `row07_desk_01.mp4`. Results land in `results/real/` as
`row07_desk_01__hmr2.npz`, `row07_desk_01__dgs_perjoint+sg.npz`, … (`+sg` = smoothed). Other people in
the clip are saved as `<clip>_t<id>__<model>.npz` (rows 3–5). Teammates' raw outputs go in
`results/real_raw/<model>/…` (paths listed in notebook 09 §2) and are converted automatically.

## Design notes worth stating in the final report

- **Same budget everywhere.** Experts share images, steps, LR (1e-4) and rank (16). The generalist has
  rank 64 (= 4 x 16, identical trainable parameters) and 4x the steps over the union of the data
  (identical compute). Detector and fusion parameters are extra on the specialist side; say so.
- **Oracle vs detector routing.** 07 routes with the true condition labels (upper bound); 08 uses the
  detector. The gap between the two is how much diagnosis costs.
- **Robustness of small gaps (RQ2).** A difference is called robust only if it has the same sign in every
  seed and the two-level bootstrap CI excludes zero. Notebook 05's single-seed CIs were tight because
  they only resample test samples; they say nothing about training noise.
- **Self-occlusion.** A joint is "hidden by the body" if fewer than 25 % of 16 rays to its skin surface
  reach the camera with no occluders present. Labels are added to the existing test set by re-running the
  generator with `--labels-only` (same seeds; placement uses ray casts, not renders); `labels.merge_selfvis`
  refuses to write if the regenerated samples don't match.
- **RQ3 is a bandit, not full RL.** Multiplicative weights over condition bins (stateless, one-step RL),
  reward = current loss on a probe from each bin. It draws from the existing 3D-occlusion pool; live
  Blender generation would only change `ArmPool.sample`. Probing and mining cost forward passes, never
  training images.
- **Real video has no ground truth.** Report proxy metrics together with the two-rater visual score
  (`visual_ratings_template.csv`) and cross-model disagreement; low jitter means smooth, not correct.
- **E2/E3 data is image-space.** Blur, low light, low resolution and reframing are applied to the clean
  Blender renders; global orientation is kept unchanged under reframing (as in standard crop augmentation).

## Tested here vs. not

Tested on CPU: every `pose_lib` module (`python -m pytest -q tests`, 16 tests) and notebooks 06-10
executed cell by cell with a toy model that has HMR2.0's structure (`python tests/dryrun_notebooks.py`).
The converters were checked on the real `video_smpl.npz` and `clipA_full.npz` from Drive.

Not run here (no GPU, no licensed SMPL/AMASS files): the real HMR2.0 forward/backward pass, Blender, YOLO.
The HMR2.0 glue is notebook 05's code, which already ran on Colab; `video.py` uses HMR2.0's own
`ViTDetDataset` for crops. The first real run of each notebook is the check: compare the pretrained
numbers in 07 with notebook 05's (PA-MPJPE 46.8 / 61.5 / 73.1 mm on B / C / D).

Known limits: the IoU tracker can swap identities when people cross (row 4); camera depth in notebook 01's
old PHALP output is not metric (the converter detects this and marks it root-relative); MediaPipe joints are
mapped to SMPL order with derived spine/neck/collar points, so compare MediaPipe mainly on proxy metrics.
