"""Cheap per-frame checks for real video (the non-learned half of 'Diagnose').

Blur, brightness, cut-off, person scale and camera motion are simple image measurements; occlusion is
left to the learned detector. Each check returns a raw value and a 0..1 score (1 = problem present),
with thresholds set so a typical phone clip of rows 1/16 scores ~0.

    scores = frame_checks(frame_bgr, bbox, prev_gray)   # dict
"""

import numpy as np

# thresholds (raw value at which the score reaches 0 and 1); tune on rows 1/16 (clean) vs 20-22
BLUR_LAPVAR = (300.0, 30.0)        # variance of Laplacian on the person crop: high = sharp
DARK_LUMA = (90.0, 25.0)           # mean luma 0..255 on the person crop
SMALL_HEIGHT = (0.35, 0.12)        # bbox height / image height
CAM_MOTION_PX = (2.0, 15.0)        # median background shift between frames, pixels at 720p


def _ramp(x, good, bad):
    """0 at `good`, 1 at `bad`, linear in between (works for increasing or decreasing)."""
    t = (x - good) / (bad - good)
    return float(np.clip(t, 0, 1))


def crop(img, bbox, pad=0.1):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 - pad * bw)), int(min(w, x1 + pad * bw))
    y0, y1 = int(max(0, y0 - pad * bh)), int(min(h, y1 + pad * bh))
    return img[y0:y1, x0:x1]


def blur(img_bgr, bbox):
    import cv2
    c = crop(img_bgr, bbox)
    if c.size == 0:
        return np.nan, 1.0
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (128, int(128 * g.shape[0] / max(g.shape[1], 1)) or 1))
    v = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return v, _ramp(v, *BLUR_LAPVAR)


def brightness(img_bgr, bbox):
    import cv2
    c = crop(img_bgr, bbox)
    if c.size == 0:
        return np.nan, 1.0
    y = float(cv2.cvtColor(c, cv2.COLOR_BGR2YUV)[..., 0].mean())
    return y, _ramp(y, *DARK_LUMA)


def cutoff(bbox, img_w, img_h, margin=3):
    """Which image borders the box touches (person likely continues outside the frame)."""
    x0, y0, x1, y1 = bbox
    sides = [x0 <= margin, y0 <= margin, x1 >= img_w - margin, y1 >= img_h - margin]
    return int(sum(sides)), float(min(1.0, sum(sides)))


def scale(bbox, img_h):
    h = (bbox[3] - bbox[1]) / img_h
    return float(h), _ramp(h, *SMALL_HEIGHT)


def close_up(bbox, img_w, img_h):
    """Box covers most of the frame -> very close to the camera (row 15)."""
    a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img_w * img_h)
    return float(a), _ramp(a, 0.45, 0.85)


def camera_motion(prev_gray, gray, bbox=None):
    """Global shift between consecutive frames by phase correlation, with the person masked out
    (so their own motion doesn't count). Returns pixels and a score."""
    import cv2
    if prev_gray is None:
        return 0.0, 0.0
    a, b = prev_gray.astype(np.float32), gray.astype(np.float32)
    if bbox is not None:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        m = np.ones_like(a)
        m[max(0, y0):y1, max(0, x0):x1] = 0
        a, b = a * m + a.mean() * (1 - m), b * m + b.mean() * (1 - m)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    d = float(np.hypot(dx, dy))
    return d, _ramp(d, *CAM_MOTION_PX)


def frame_checks(img_bgr, bbox, prev_gray=None, small_gray=None):
    """All cheap checks for one frame. small_gray: downscaled grayscale frame for camera motion."""
    h, w = img_bgr.shape[:2]
    out = {}
    out["blur_raw"], out["blur"] = blur(img_bgr, bbox)
    out["dark_raw"], out["dark"] = brightness(img_bgr, bbox)
    out["cutoff_raw"], out["cutoff"] = cutoff(bbox, w, h)
    out["small_raw"], out["small"] = scale(bbox, h)
    out["close_raw"], out["close"] = close_up(bbox, w, h)
    if small_gray is not None:
        sb = None
        if bbox is not None:
            f = small_gray.shape[1] / w
            sb = [v * f for v in bbox]
        out["cam_motion_raw"], out["cam_motion"] = camera_motion(prev_gray, small_gray, sb)
    # condition scores matching the experts / sequence module
    out["cond_quality"] = max(out["blur"], out["dark"])
    out["cond_framing"] = max(out["cutoff"], out["small"], out["close"])
    return out
