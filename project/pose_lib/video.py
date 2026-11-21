"""Phone video -> PoseSeq files, without PHALP or detectron2.

Notebook 01 used 4D-Humans' track.py, which needs PHALP (and detectron2); that is what broke on
Colab's Python 3.13. This replaces the front end with:
    YOLO person detection (ultralytics, plain pip)  ->  greedy IoU tracker  ->  HMR2.0 on 256x256 crops
The crops come from HMR2.0's own ViTDetDataset, so preprocessing is identical to the official demo.
One pass over the video can run several predictors on the same crops (the released model, each expert,
the full DGS model), and also records the cheap condition checks per frame.

    from pose_lib import video
    model, cfg = video.load_hmr2_with_cfg()
    video.run(clip_path, out_dir, {"hmr2": video.base_predictor(model)}, cfg, fk)
"""

import os
import time

import numpy as np
import torch

from . import conditions
from .convert import from_params
from .rotations import rotmat_to_aa


# ---------------------------------------------------------------- models
def load_hmr2_with_cfg():
    from hmr2.models import load_hmr2
    m, cfg = load_hmr2()
    for attr in ("renderer", "mesh_renderer", "discriminator"):
        if hasattr(m, attr):
            setattr(m, attr, None)
    return m, cfg


def base_predictor(model):
    from .hmr2_ft import predict
    from . import lora

    def f(img):
        if lora.lora_modules(model):
            lora.set_active(model, None)
        return predict(model, img)
    return f


def expert_predictor(model, name):
    from .hmr2_ft import predict
    from . import lora

    def f(img):
        lora.set_active(model, name)
        out = predict(model, img)
        lora.set_active(model, None)
        return out
    return f


def load_detector_yolo(weights="yolo11n.pt"):
    from ultralytics import YOLO
    return YOLO(weights)


# ---------------------------------------------------------------- tracking
def iou(a, b):
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


class Tracker:
    """Greedy IoU matching with a short memory. Good enough for rows 1-2 and 6-24; identity swaps when
    people cross (row 4) are a known limitation (that is the 'tracking front-end' part of the plan)."""

    def __init__(self, iou_thresh=0.3, max_age=15):
        self.iou_thresh, self.max_age = iou_thresh, max_age
        self.tracks = {}      # id -> {"box", "last"}
        self.next_id = 0

    def update(self, t, boxes):
        ids = [-1] * len(boxes)
        live = [k for k, v in self.tracks.items() if t - v["last"] <= self.max_age]
        pairs = sorted(((iou(self.tracks[k]["box"], b), k, i) for k in live for i, b in enumerate(boxes)),
                       reverse=True)
        used_k, used_i = set(), set()
        for s, k, i in pairs:
            if s < self.iou_thresh or k in used_k or i in used_i:
                continue
            ids[i] = k
            used_k.add(k); used_i.add(i)
        for i, b in enumerate(boxes):
            if ids[i] < 0:
                ids[i] = self.next_id
                self.next_id += 1
            self.tracks[ids[i]] = {"box": np.asarray(b, float), "last": t}
        return ids


# ---------------------------------------------------------------- camera
def cam_crop_to_full(cam, box_center, box_size, img_size, focal):
    """HMR2.0's crop camera (s, tx, ty) -> translation in the full image's camera (same formula as
    hmr2.utils.renderer.cam_crop_to_full)."""
    w2, h2 = img_size[:, 0] / 2, img_size[:, 1] / 2
    bs = box_size * cam[:, 0] + 1e-9
    tz = 2 * focal / bs
    tx = 2 * (box_center[:, 0] - w2) / bs + cam[:, 1]
    ty = 2 * (box_center[:, 1] - h2) / bs + cam[:, 2]
    return np.stack([tx, ty, tz], -1)


# ---------------------------------------------------------------- main
def read_frames(path, target_h=720, max_seconds=None, fps_out=None):
    """Decode a clip, downscaled to target_h. Returns frames (list of BGR), fps."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1 if not fps_out else max(1, int(round(fps / fps_out)))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if max_seconds and i >= max_seconds * fps:
            break
        if i % step == 0:
            if target_h and f.shape[0] != target_h:
                s = target_h / f.shape[0]
                f = cv2.resize(f, (int(round(f.shape[1] * s / 2) * 2), target_h), interpolation=cv2.INTER_AREA)
            frames.append(f)
        i += 1
    cap.release()
    return frames, fps / step


def run(clip_path, out_dir, predictors, model_cfg, fk, yolo=None, clip=None, target_h=720, max_seconds=None,
        det_conf=0.35, min_track_len=15, batch_size=32, device="cuda", all_tracks=True, extra_per_frame=None):
    """Process one clip with every predictor. Writes <clip>__<name>[_t<id>].npz into out_dir and returns
    {name: [paths]} plus timing. The main (longest) track is saved without a suffix."""
    import cv2
    from hmr2.datasets.vitdet_dataset import ViTDetDataset

    clip = clip or os.path.splitext(os.path.basename(clip_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    yolo = yolo or load_detector_yolo()
    frames, fps = read_frames(clip_path, target_h, max_seconds)
    H, W = frames[0].shape[:2]
    tracker = Tracker()

    # 1) detect + track + crop + cheap checks
    t0 = time.time()
    crops, meta = [], []           # meta: (frame, track id, box, center, size, score, checks)
    prev_small = None
    for t, f in enumerate(frames):
        r = yolo(f, classes=[0], conf=det_conf, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        small = cv2.cvtColor(cv2.resize(f, (320, int(320 * H / W))), cv2.COLOR_BGR2GRAY)
        if len(boxes) == 0:
            prev_small = small
            continue
        ids = tracker.update(t, boxes)
        ds = ViTDetDataset(model_cfg, f, boxes)
        for i in range(len(boxes)):
            item = ds[i]
            chk = conditions.frame_checks(f, boxes[i], prev_small, small)
            crops.append(torch.as_tensor(item["img"]))
            meta.append((t, ids[i], boxes[i], np.asarray(item["box_center"], float), float(item["box_size"]),
                         float(scores[i]), chk))
        prev_small = small
    t_detect = time.time() - t0
    if not meta:
        print(f"[video] {clip}: no person detected")
        return {}, {}

    # 2) run every predictor on the crops (same crops for all -> a fair comparison)
    imgs = torch.stack(crops)
    results, timing = {}, {"detect_track_s": t_detect, "frames": len(frames)}
    for name, pf in predictors.items():
        t1 = time.time()
        acc = {k: [] for k in ("global_orient", "body_pose", "betas", "cam")}
        extra_acc = {}
        for b in torch.split(imgs, batch_size):
            b = b.to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                p = pf(b)
            for k in acc:
                acc[k].append(p[k].float().cpu())
            last = getattr(pf, "last", None)
            if isinstance(last, dict):
                for k, v in last.items():
                    extra_acc.setdefault(k, []).append(v.float().cpu())
        P = {k: torch.cat(v).numpy() for k, v in acc.items()}
        X = {k: torch.cat(v).numpy() for k, v in extra_acc.items()}
        timing[f"{name}_s"] = time.time() - t1
        results[name] = (P, X)

    # 3) one PoseSeq per (predictor, track)
    tid = np.array([m[1] for m in meta])
    counts = {k: int((tid == k).sum()) for k in np.unique(tid)}
    main_tid = max(counts, key=counts.get)
    keep_tids = [k for k, c in counts.items() if c >= min_track_len and (all_tracks or k == main_tid)]
    focal = 5000.0 / 256 * max(W, H)
    paths = {}
    for name, (P, X) in results.items():
        for k in keep_tids:
            sel = np.nonzero(tid == k)[0]
            fr = np.array([meta[i][0] for i in sel])
            _, first = np.unique(fr, return_index=True)          # one detection per frame per track
            sel = sel[first]
            fr = fr[first]
            centers = np.stack([meta[i][3] for i in sel])
            sizes = np.array([meta[i][4] for i in sel])
            cam_full = cam_crop_to_full(P["cam"][sel], centers, sizes, np.tile([W, H], (len(sel), 1)), focal)
            checks = {}
            for key in meta[sel[0]][6]:
                checks["chk_" + key] = np.array([meta[i][6][key] for i in sel], np.float32)
            extra = {**checks, **{f"dgs_{kk}": vv[sel] for kk, vv in X.items()}}
            if extra_per_frame:
                extra.update(extra_per_frame)
            seq = from_params(rotmat_to_aa(P["global_orient"][sel]), rotmat_to_aa(P["body_pose"][sel]), fk,
                              model=name, clip=clip, fps=fps, frame_idx=fr, n_frames=len(frames),
                              betas=P["betas"][sel].mean(0), camera=cam_full,
                              score=np.array([meta[i][5] for i in sel]), bbox=np.stack([meta[i][2] for i in sel]),
                              extra=extra, img_size=np.array([W, H]), track_id=int(k))
            suffix = "" if k == main_tid else f"_t{k}"
            p = os.path.join(out_dir, f"{clip}{suffix}__{name}.npz")
            seq.save(p)
            paths.setdefault(name, []).append(p)
    timing["n_tracks_saved"] = len(keep_tids)
    print(f"[video] {clip}: {len(frames)} frames, {len(meta)} detections, tracks {counts} -> saved {keep_tids}; "
          + ", ".join(f"{k} {v:.0f}s" for k, v in timing.items() if k.endswith("_s")))
    return paths, timing
