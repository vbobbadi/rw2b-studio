"""
Turn generator output into the three training/test conditions (plain Python: numpy + Pillow).

    python synth_post.py --gen /content/synth/train --cutouts /content/synth/cutouts \
        --out /content/drive/MyDrive/CS747_project/data/train [--bg-dir /content/backgrounds]

For every sample i, the SAME background is used in all three conditions:
    B_clean/    person only                                (no occlusion)
    C_paste2d/  person + 2D cutouts pasted on top          (cheap baseline, same object types as D)
    D_occ3d/    person + rendered 3D occluders             (our method)
Each folder gets images/XXXXXX.jpg and labels.npz (generator labels + per-condition `vis` and `occfrac`).
For C, the number of objects per sample matches D, and pastes are retried until the hidden fraction
matches D's for that same sample (within 0.05), so both conditions hide the same amount of body.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageFilter

RES = 256


def procedural_bg(rng):
    """Cluttered random background: gradient + blurred rectangles + noise."""
    c1, c2 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    t = np.linspace(0, 1, RES)[:, None, None]
    img = (c1 * (1 - t) + c2 * t) * np.ones((RES, RES, 3))
    for _ in range(rng.integers(3, 12)):
        x0, y0 = rng.integers(-RES // 2, RES, 2)
        w, h = rng.integers(10, RES, 2)
        img[max(y0, 0):max(y0 + h, 0), max(x0, 0):max(x0 + w, 0)] = rng.uniform(0, 255, 3)
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0, 4)))
    arr = np.asarray(im).astype(np.float32) + rng.normal(0, rng.uniform(0, 8), (RES, RES, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def photo_bg(paths, rng):
    im = Image.open(paths[rng.integers(len(paths))]).convert("RGB")
    s = rng.uniform(RES, max(RES + 1, min(im.size)))
    x, y = rng.uniform(0, im.size[0] - s), rng.uniform(0, im.size[1] - s)
    return im.crop((int(x), int(y), int(x + s), int(y + s))).resize((RES, RES), Image.BILINEAR)


def load_cutouts(folder):
    cuts = []
    for p in sorted(glob.glob(os.path.join(folder, "cutout_*.png"))):
        im = Image.open(p).convert("RGBA")
        bbox = im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
        if bbox:
            cuts.append(im.crop(bbox))
    if not cuts:
        raise RuntimeError(f"no cutouts in {folder}")
    return cuts


def alpha_at(alpha, pts):
    """alpha (H,W) in [0,1]; pts (K,2) pixel coords -> (K,) alpha, 0 outside the image."""
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < RES) & (y >= 0) & (y < RES)
    out = np.zeros(len(pts))
    out[inside] = alpha[y[inside], x[inside]]
    return out


def paste_2d(person, cuts, n, verts2d, selfvis, joints2d, rng, target, tol=0.05, tries=25):
    """Paste n cutouts near visible body points, retrying until the hidden fraction is within `tol`
    of `target` (the 3D version's value for the same sample), so C and D hide the same amount.
    Returns RGBA image, joint vis, hidden fraction."""
    vis_pts = verts2d[selfvis] if selfvis.any() else verts2d
    lo, hi = verts2d.min(0), verts2d.max(0)
    body_h = max(hi[1] - lo[1], 20)
    best = None
    for attempt in range(tries):
        layer = Image.new("RGBA", (RES, RES), (0, 0, 0, 0))
        for _ in range(n):
            c = cuts[rng.integers(len(cuts))]
            target_h = body_h * rng.uniform(0.25, 0.9)
            s = target_h / c.size[1]
            w, h = max(2, int(c.size[0] * s)), max(2, int(c.size[1] * s))
            if rng.uniform() < 0.5:
                c = c.transpose(Image.FLIP_LEFT_RIGHT)
            ci = c.resize((w, h), Image.BILINEAR)
            cx, cy = vis_pts[rng.integers(len(vis_pts))] + rng.normal(0, 0.08 * body_h, 2)
            layer.alpha_composite(ci, (int(cx - w / 2), int(cy - h / 2))) if (
                0 <= int(cx - w / 2) and 0 <= int(cy - h / 2) and int(cx - w / 2) + w <= RES and int(cy - h / 2) + h <= RES
            ) else _paste_clipped(layer, ci, int(cx - w / 2), int(cy - h / 2))
        a = np.asarray(layer.getchannel("A")).astype(np.float32) / 255
        hidden = selfvis & (alpha_at(a, verts2d) > 0.5)
        frac = hidden.sum() / max(selfvis.sum(), 1)
        if best is None or abs(frac - target) < abs(best[2] - target):
            best = (layer, a, frac)
        if abs(frac - target) <= tol:
            break
    layer, a, frac = best
    out = person.copy()
    out.alpha_composite(layer)
    vis = alpha_at(a, joints2d) <= 0.5
    return out, vis, frac


def _paste_clipped(layer, im, x, y):
    """alpha_composite that tolerates partially off-canvas positions."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + im.size[0], RES), min(y + im.size[1], RES)
    if x1 <= x0 or y1 <= y0:
        return
    layer.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def on_bg(rgba, bg):
    out = bg.convert("RGBA")
    out.alpha_composite(rgba)
    return out.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True, help="generator output folder(s)")
    ap.add_argument("--cutouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg-dir", default=None, help="optional folder of background photos (no people)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    cuts = load_cutouts(a.cutouts)
    bgs = sorted(glob.glob(os.path.join(a.bg_dir, "**", "*.jpg"), recursive=True)) if a.bg_dir else []
    conds = {"B_clean": {}, "C_paste2d": {}, "D_occ3d": {}}
    for c in conds:
        os.makedirs(os.path.join(a.out, c, "images"), exist_ok=True)

    merged, names = {}, []
    for g in a.gen:
        L = dict(np.load(os.path.join(g, "labels.npz")))
        start = int(L.pop("start"))
        n = len(L["focal"])
        for i in range(n):
            stem = f"{start + i:06d}"
            person = Image.open(os.path.join(g, f"clean_{stem}.png")).convert("RGBA")
            occ3d = Image.open(os.path.join(g, f"occ3d_{stem}.png")).convert("RGBA")
            bg = photo_bg(bgs, rng) if bgs and rng.uniform() < 0.8 else procedural_bg(rng)

            on_bg(person, bg).save(os.path.join(a.out, "B_clean", "images", stem + ".jpg"), quality=a.quality)
            on_bg(occ3d, bg).save(os.path.join(a.out, "D_occ3d", "images", stem + ".jpg"), quality=a.quality)
            pasted, vis_c, frac_c = paste_2d(person, cuts, int(L["n_occ"][i]), L["verts2d"][i], L["selfvis"][i],
                                             L["joints2d"][i], rng, float(L["occfrac_occ3d"][i]))
            on_bg(pasted, bg).save(os.path.join(a.out, "C_paste2d", "images", stem + ".jpg"), quality=a.quality)

            conds["B_clean"].setdefault("vis", []).append(np.ones(24, bool))
            conds["B_clean"].setdefault("occfrac", []).append(0.0)
            conds["C_paste2d"].setdefault("vis", []).append(vis_c)
            conds["C_paste2d"].setdefault("occfrac", []).append(frac_c)
            conds["D_occ3d"].setdefault("vis", []).append(L["vis_occ3d"][i])
            conds["D_occ3d"].setdefault("occfrac", []).append(L["occfrac_occ3d"][i])
            names.append(stem)
        for k, v in L.items():
            if k in ("vis_occ3d", "occfrac_occ3d", "verts2d", "selfvis"):  # per-condition versions saved below
                continue
            merged.setdefault(k, []).append(v)
        print(f"[post] {g}: {n} samples")

    base = {k: np.concatenate(v) for k, v in merged.items()}
    for c, extra in conds.items():
        np.savez(os.path.join(a.out, c, "labels.npz"), names=np.array(names), **base,
                 **{k: np.asarray(v) for k, v in extra.items()})
        fr = np.asarray(extra["occfrac"])
        if "occ_ok" in base:
            print(f"[post] {c}: {int((~base['occ_ok']).sum())} samples flagged occ_ok=False (filter these out for training/eval)")
        print(f"[post] {c}: {len(names)} images | hidden fraction mean {fr.mean():.2f} "
              f"(min {fr.min():.2f}, max {fr.max():.2f}) | joints hidden {1 - np.mean(extra['vis']):.2f}")


if __name__ == "__main__":
    main()
