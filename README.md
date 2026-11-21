# Real World to Blender Studio

Project site for CS 747 (Deep Learning, GMU, Fall 2026): turning everyday video of people into animated 3D characters.

**Live site:** https://vbobbadi.github.io/rw2b-studio/

- **Pose Studio**: upload a video or photo with up to 5 people, pick a character for each person, record the result or export the motion for Blender.
- **Live Camera**: the same with your webcam, one character per person.
- **Character Library**: built-in characters plus your own Mixamo FBX/GLB rigs.
- **Architecture**: animated diagram of the pipeline (person detection → problem identifier → router → LoRA experts on HMR2.0 → fusion → smoothing → character).
- **Research questions**: one page per RQ with the results from 3DPW and our 24 real clips.

Everything runs in the browser (MediaPipe Pose Landmarker, three.js). Videos never leave your device.

## Project code

The research code is in [`project/`](project/): `pose_lib/` (training, experts, fusion, problem detector, evaluation), `notebooks/` 06–12 (Colab), `blender/` (synthetic data and export), `tests/`, and the result tables in `project/results/`. A summary of every result is in [`project/RESULTS.md`](project/RESULTS.md). Model weights, 3DPW and the rendered datasets are not included (size and licence).

Run the site locally: `python -m http.server 8000` in this folder, then open http://localhost:8000.
