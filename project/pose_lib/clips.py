"""Stock clips for the 29-row test plan (Pexels, free license) + edited variants + a quick fit check.

Rows 13 and 22 are *edits* of another clip (named `..._edit`): a virtual camera that keeps the legs out of
frame (13) and a 480p re-encode (22). They are weaker evidence than natural footage because the E2/E3
experts were trained on the same kinds of damage; report them separately.
"""

import os
import subprocess

import numpy as np

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0 Safari/537.36")

# row -> (pexels id, short description)
PEXELS = {
    1: (4079656, "slowwalk"), 2: (6616343, "dance"), 3: (5977046, "twoapart"), 4: (17773203, "crossing"),
    5: (855564, "crowd"), 6: (7660743, "hug"), 8: (8011197, "chair"), 10: (7671778, "armscrossed"),
    11: (5100042, "box"), 12: (2376809, "floor"), 14: (4507439, "far"), 16: (855482, "tripod"),
    17: (5636206, "handheld"), 18: (32789557, "follow"), 20: (5582584, "lowlight"), 21: (7570258, "blur"),
    23: (20597684, "long"), 24: (8301754, "market"),
}
EXTRA = {10: [(4812503, "stretch")], 11: [(5043690, "parcels")]}


def name(row, desc, k=1, edit=False):
    return f"row{row:02d}_{desc}{'_edit' if edit else ''}_{k:02d}.mp4"


def download(pid, dst, max_h=1080):
    """Pexels download endpoint -> mp4, re-encoded to <= max_h (keeps files small, 30 fps max)."""
    import requests
    if os.path.exists(dst):
        return dst
    tmp = dst + ".src"
    r = requests.get(f"https://www.pexels.com/download/video/{pid}/", headers={"User-Agent": UA},
                     allow_redirects=True, stream=True, timeout=120)
    r.raise_for_status()
    if "video" not in r.headers.get("content-type", ""):
        raise RuntimeError(f"{pid}: got {r.headers.get('content-type')} from {r.url}")
    with open(tmp, "wb") as f:
        for ch in r.iter_content(1 << 20):
            f.write(ch)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-vf",
                    f"scale=-2:'min({max_h},ih)'", "-fpsmax", "30", "-an", "-c:v", "libx264",
                    "-crf", "20", "-preset", "veryfast", dst], check=True)
    os.remove(tmp)
    return dst


def to_480p(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=-2:480", "-an",
                    "-c:v", "libx264", "-b:v", "600k", "-preset", "veryfast", dst], check=True)
    return dst


def upper_body_cam(src, dst, yolo, out_wh=(1280, 720), hip=0.55, smooth=0.8):
    """Virtual camera that follows the person and frames head..hips: the legs are out of frame (row 13)."""
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = out_wh
    vw = cv2.VideoWriter(dst + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    win = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = yolo(f, classes=[0], conf=0.35, verbose=False)[0]
        if len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy()
            b = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
            h = b[3] - b[1]
            y0, y1 = b[1] - 0.1 * h, b[1] + hip * h
            hh = y1 - y0
            ww = hh * W / H
            cx = (b[0] + b[2]) / 2
            new = np.array([cx - ww / 2, y0, cx + ww / 2, y1])
            win = new if win is None else smooth * win + (1 - smooth) * new
        if win is None:
            continue
        x0, y0, x1, y1 = win
        M = cv2.getAffineTransform(np.float32([[x0, y0], [x1, y0], [x0, y1]]), np.float32([[0, 0], [W, 0], [0, H]]))
        vw.write(cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
    cap.release()
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", dst + ".tmp.mp4", "-c:v", "libx264", "-crf", "20",
                    "-preset", "veryfast", dst], check=True)
    os.remove(dst + ".tmp.mp4")
    return dst


def check(path, yolo, n=8, sheet=None):
    """Quick fit check on n evenly spaced frames: people per frame, main box height / frame height,
    fraction of frames where the main box touches the border, mean brightness, duration. Optionally
    writes a contact sheet (jpg) with the boxes drawn."""
    import cv2
    cap = cv2.VideoCapture(path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tiles, people, hfrac, touch, bright = [], [], [], [], []
    for t in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]
        r = yolo(f, classes=[0], conf=0.35, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        people.append(len(b))
        bright.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
        if len(b):
            m = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
            hfrac.append((m[3] - m[1]) / H)
            touch.append(bool(m[0] < 3 or m[1] < 3 or m[2] > W - 3 or m[3] > H - 3))
        for bb in b:
            cv2.rectangle(f, tuple(int(v) for v in bb[:2]), tuple(int(v) for v in bb[2:]), (0, 255, 0), 3)
        tiles.append(cv2.resize(f, (320, int(320 * H / W))))
    cap.release()
    if sheet and tiles:
        th = min(t.shape[0] for t in tiles)
        tiles = [t[:th] for t in tiles]
        rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])))
                for i in range(0, len(tiles), 4)]
        cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"seconds": round(N / fps, 1), "people_median": float(np.median(people)) if people else 0,
            "people_max": int(max(people)) if people else 0,
            "box_h_frac": round(float(np.median(hfrac)), 2) if hfrac else np.nan,
            "touches_border": round(float(np.mean(touch)), 2) if touch else np.nan,
            "brightness": round(float(np.mean(bright)), 1) if bright else np.nan,
            "frames_with_person": round(len(hfrac) / max(len(people), 1), 2)}
