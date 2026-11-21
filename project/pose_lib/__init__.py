"""pose_lib: shared code for the CS 747 'Real World to Blender' project.

Modules
-------
format      standard per-clip .npz (PoseSeq) used by every model and every teammate
convert     HMR2.0 / Multi-HMR / MediaPipe / PARE / WHAM / GVHMR outputs -> PoseSeq
rotations   numpy + torch rotation helpers (axis-angle, matrices, quaternions)
smpl        chumpy-free SMPL loading, .pkl -> .npz, forward kinematics (numpy and torch)
metrics     MPJPE / PA-MPJPE (synthetic) and proxy metrics for real video (jitter, foot skate, ...)
stats       paired bootstrap, seed aggregation
smoothing   Savitzky-Golay on positions, quaternion smoothing on rotations
conditions  cheap per-frame checks: blur, brightness, cut-off, scale, camera motion
degrade     synthetic quality (E2) and framing (E3) conditions made from clean renders
hmr2_ft     HMR2.0 fine-tuning / evaluation (ported from notebook 05, plus LoRA mode)
lora        LoRA adapters: many named experts on one frozen backbone
detector    learned problem detector (occlusion / quality / framing + per-joint visibility)
fusion      weighted-average, per-joint-confidence and learned fusion of expert outputs
selector    RQ3: bandit vs random vs hard-example mining at a fixed data budget
video       phone video -> PoseSeq with YOLO + simple tracker + HMR2.0 (no PHALP / detectron2)
"""

__version__ = "0.1.0"
