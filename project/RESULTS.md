# CS 747 — results (Colab runs, Nov 19–20 2026)

Synthetic numbers are PA-MPJPE in mm on notebook 04's AMASS test set (942 samples, occ_ok only). Real-data sections say which data they use.
Result tables are in `results/` next to this file. Code: `pose_lib/` and `notebooks/` (06–12).

## Sanity check
New code reproduces notebook 05 exactly: released HMR2.0 = 46.8 / 61.5 / 73.1 (B / C / D test); seed-0 B_clean model = 31.7 on B.

## RQ2 — 2D-pasted vs 3D-rendered occlusion (notebook 06, 3 seeds, full fine-tune, last 4 blocks)
Seed spread is tiny (±0.1–0.3 mm), so every gap below is robust (same sign all seeds, two-level bootstrap CI excludes 0):
- D (3D) beats C (2D) on D's test: −2.1 mm [−2.6, −1.6]
- C beats D on C's test: −2.3 mm; on clean images D costs +3.1 mm vs C
- D beats clean-only B on D's test: −5.4 mm; D ≈ B on C's test (sign flips)
- Every fine-tuned model beats the released model by 8.5–11.6 mm

Self-occlusion split (D test, per-joint label; ~17% of joints are hidden by the body):

| model | visible | hidden by object | hidden by body |
|---|---|---|---|
| pretrained | 51.6 | 96.9 | 75.1 |
| B_clean | 44.4 | 92.4 | 67.8 |
| C_paste2d | 42.1 | 87.6 | 65.1 |
| D_occ3d | 40.2 | 85.1 | 63.4 |

On real data (3DPW, below) D is 2.1 mm *worse* than C on PA-MPJPE but 1.7 mm better on MPJPE — 3DPW has little occlusion, consistent with D's cost on clean images.

## RQ4 — specialists vs one generalist (notebooks 07, 08; LoRA r=16, last 4 blocks + head)

| | clean | occ3d | paste2d | quality | framing | occ3d+quality |
|---|---|---|---|---|---|---|
| pretrained | 46.8 | 73.1 | 61.5 | 52.2 | 63.6 | 80.3 |
| E0 clean | 32.2 | 66.9 | 53.0 | 39.9 | 53.8 | 75.7 |
| E1 occlusion | 38.1 | 62.1 | 53.5 | 44.8 | 53.4 | 71.0 |
| E2 quality | 33.8 | 68.8 | 53.8 | 39.7 | 54.8 | 77.3 |
| E3 framing | 35.4 | 66.4 | 53.4 | 42.2 | 46.7 | 74.5 |
| **GEN (r=64, union data, 4× steps)** | **28.4** | **58.6** | **48.3** | **35.6** | **41.7** | **68.1** |
| DGS oracle routing | 32.2 | 62.1 | 53.5 | 39.7 | 46.7 | 72.3 |
| DGS detector + learned fusion | 32.3 | 62.1 | 51.6 | 39.6 | 46.4 | 71.4 |

The generalist wins every condition, including each expert's home condition, by 3.3–4.8 mm (Holm-corrected p < 0.001). Holds on 3DPW too (GEN − DGS learned = −1.4 mm PA).
Caveat: GEN matches the experts' *combined* parameters and compute, but each expert alone saw ¼ of GEN's steps and data. A compute-per-expert-matched run is the open follow-up.

## RQ5 — fusion (notebook 08)
Learned fusion is best (mean 50.6) vs weighted 51.0, per-joint 51.1, top-1 51.1, oracle 51.1. On 3DPW learned = weighted (+0.05 mm, n.s.). Fusion cannot close the gap to GEN.

## RQ6 — problem detector (ResNet-18)
- Synthetic test AUC: occlusion 0.9999, quality 0.9998, framing 0.9998; per-joint visibility accuracy 93.3%; hidden-fraction MAE 0.033.
- **Real stock clips (rows 1–24): the detector does not transfer.** AUC occlusion 0.50, quality 0.33, framing 0.54 (chance = 0.5). The cheap image checks do better (quality 0.56, framing 0.68).
- 3DPW: HMR2.0 error rises with the detector's occlusion score (47.3 → 49.6 → 50.1 mm over bins 0–0.05 / 0.05–0.15 / 0.15–0.3), so the score carries some signal, but E1 is never better than E0 in any bin.

## RQ3 — adaptive hard-data selection (notebook 10, 3 seeds, 2,000 images, 480 steps)

| strategy | D test | C test | B test |
|---|---|---|---|
| random | **63.7 ± 0.4** | **55.1 ± 0.2** | **39.9 ± 0.5** |
| bandit (Hedge) | 64.8 ± 0.3 | 55.8 ± 0.3 | 41.2 ± 0.5 |
| hard-example mining | 66.8 ± 0.3 | 58.0 ± 0.5 | 43.0 ± 0.5 |

The bandit moved 92% of the budget into the 60–100%-hidden bin and still only ties random there (81.3). Robust ordering: random > bandit > HEM.

## RQ1a — 3DPW, real video with ground truth (notebook 12)
Test split, 24 sequences, every 2nd valid frame, 17,764 person-frames; crops from projected GT joints; GT joints from neutral SMPL + GT betas; 22 SMPL body joints. Comparable between our models, not to published numbers (gendered models + H36M-14 regressor). Bootstrap over sequences, Holm-corrected. Dataset licence: non-commercial research, cite von Marcard et al., ECCV 2018. The raw data is not redistributed here.

| model | PA-MPJPE | MPJPE |
|---|---|---|
| released HMR2.0 | 47.9 | **73.2** |
| B clean / C 2D-occ / D 3D-occ | 43.8 / 43.7 / 45.9 | 77.4 / 77.2 / 75.5 |
| E0 / E1 / E2 / E3 | 44.1 / 45.3 / 45.1 / 43.9 | 77.5 / 73.6 / 82.8 / 77.9 |
| **GEN** | **42.9** | 83.7 |
| DGS weighted / per-joint / learned / top-1 | 44.3 / 44.3 / 44.3 / 44.8 | 78.1 / 79.0 / 78.7 / 79.8 |

- Synthetic fine-tuning transfers for articulated pose: GEN −5.0 mm PA vs released [−6.9, −3.0]; B −4.1; DGS −3.6. All p_holm ≤ 0.003.
- Global orientation gets worse: MPJPE (no rotation alignment) +10.5 mm for GEN [+4.6, +15.9], +4–6 mm for B/DGS. Likely the camera/viewpoint distribution of the Blender data.
- E1 vs E0: +1.2 mm PA, −3.9 mm MPJPE (occlusion data helps orientation, hurts articulation on mostly unoccluded frames).
- Per sequence: GEN is best on 16/24 and within 0.5 mm on 18; biggest gains on walking sequences (e.g. warmWelcome 56.8 → 39.9); released HMR2.0 is best on flat_guitar_01 (37.3 vs 41.4) and downtown_stairs_00.

## RQ1b — 29-row test plan on real clips (notebooks 11 + 09; proxy metrics, no ground truth)
22 Pexels stock clips for rows 1–24 (sources in `results/clips/sources.csv`). Row 13 (legs out of frame) and row 22 (480p) are edits of rows 16 / 1. Rows 7 and 9 are team clips. Rows 15 and 19 have no clip yet.

Raw jitter (mm, main track) — DGS weighted vs released HMR2.0:
- DGS lower in only 9/22 clips; mean 14.5 vs 14.0 mm. Clear DGS wins: row 10 self-occlusion stretch (14.1 → 10.6), row 22 low-res (10.8 → 8.5), row 6 hug (22.2 → 20.1). Clear losses: row 20 low light (16.4 → 24.9), row 23 long crowd clip (8.1 → 10.4), row 14 far (16.5 → 18.7).
- After Savitzky–Golay smoothing every model is ~2 mm.
- Detection rate (YOLO + IoU tracker, main track) is the main real-world failure: far 0.34, long crowd 0.34, low light 0.41, hug 0.59, crowd 0.61; 1.0 on 13 clips. Left/right swaps appear only in crowd (4), far (6), blur (1).
- Runtime (L4, 12,282 frames): detect + track 598 s, HMR2.0 331 s, DGS 1,408 s (4.3×).

## Not done yet
- Rows 15 and 19 clips; better replacements for rows 1, 3, 21.
- A re-identifying tracker (ByteTrack-style) — first fix for real video.
- Compute-matched RQ4 rerun (experts at 4× steps).
- Check notebook 04's camera ranges against 3DPW (explains the MPJPE increase).
