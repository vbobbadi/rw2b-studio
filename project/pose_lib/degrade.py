"""Synthetic training/test conditions for experts E2 (quality) and E3 (framing), made from the clean
renders in B_clean. No extra Blender rendering is needed: these are image-space effects, and their
labels follow exactly (photometric effects change no labels; framing is a 2D similarity transform, so
2D joints move with it and 3D root-relative joints don't change).

Each transform has the signature used by hmr2_ft.SynthDataset:
    fn(img_uint8 (256,256,3), labels {'joints2d','vis','occfrac',...}, rng) -> (img_uint8, updates)
`updates` may contain new 'joints2d' / 'vis' and condition labels 'cond_quality' / 'cond_framing'
(severity 0..1) that the problem detector learns to predict.

Rows covered: E2 quality = rows 20 (low light), 21 (motion blur), 22 (480p);
              E3 framing = rows 13 (partly out of frame), 14 (far / small), 15 (very close).
Row 12 (sitting/lying) is a pose distribution, not an image effect: AMASS already contains it.
"""

import numpy as np
from PIL import Image, ImageFilter

RES = 256
VIEW_X0, VIEW_X1 = 32, 224        # HMR2.0 only sees the central 192 columns of the 256 crop

# ---------------------------------------------------------------- E2: quality


def motion_blur(img, length, angle_deg):
    import cv2
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def low_light(img, rng, gain, gamma):
    x = img.astype(np.float32) / 255
    x = (x ** gamma) * gain
    # sensor noise grows when the image is dark: shot noise ~ sqrt(signal) + read noise
    x = x + rng.normal(0, 1, x.shape) * (np.sqrt(np.clip(x, 0, 1)) * 0.04 + 0.01)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def low_res(img, size):
    im = Image.fromarray(img)
    return np.asarray(im.resize((size, size), Image.BILINEAR).resize((RES, RES), Image.BILINEAR))


def jpeg(img, q):
    import io
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(q))
    return np.asarray(Image.open(buf).convert("RGB"))


def quality(kind=None, severity=None):
    """E2 transform. kind in {'blur','motion','dark','lowres','mixed'} or None (random); severity 0..1 or
    None (random). Fixing kind+severity gives the test grid and the RQ3 'arms'."""
    def fn(img, lab, rng):
        k = kind or rng.choice(["blur", "motion", "dark", "lowres", "mixed"])
        s = float(rng.uniform(0.2, 1.0) if severity is None else severity)
        out = img
        ks = [k] if k != "mixed" else list(rng.choice(["blur", "motion", "dark", "lowres"], 2, replace=False))
        for kk in ks:
            if kk == "blur":
                out = np.asarray(Image.fromarray(out).filter(ImageFilter.GaussianBlur(0.5 + 3.5 * s)))
            elif kk == "motion":
                out = motion_blur(out, int(3 + 14 * s) | 1, rng.uniform(0, 180))
            elif kk == "dark":
                out = low_light(out, rng, gain=1.0 - 0.85 * s, gamma=1.0 + 1.2 * s)
            elif kk == "lowres":
                out = low_res(out, int(round(128 - 88 * s)))      # 128 px (~480p person) down to 40 px
        if rng.uniform() < 0.5:
            out = jpeg(out, rng.uniform(25, 70))
        return out, {"cond_quality": s}
    return fn


# ---------------------------------------------------------------- E3: framing
def _fill(rng):
    c = rng.uniform(0, 255, 3)
    base = np.ones((RES, RES, 3)) * c + rng.normal(0, 12, (RES, RES, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))


def similarity(img, j2d, scale, shift, rng):
    """Zoom about the image centre by `scale` and shift by `shift` pixels. Returns new img, joints2d."""
    c = np.array([RES / 2, RES / 2])
    # PIL affine maps output -> input: x_in = (x_out - c - shift) / scale + c
    a = 1 / scale
    tx, ty = c - (c + shift) * a
    out = Image.fromarray(img).transform((RES, RES), Image.AFFINE, (a, 0, tx, 0, a, ty), Image.BILINEAR,
                                         fillcolor=None)
    mask = Image.new("L", (RES, RES), 255).transform((RES, RES), Image.AFFINE, (a, 0, tx, 0, a, ty),
                                                     Image.NEAREST, fillcolor=0)
    bg = _fill(rng)
    bg.paste(out, (0, 0), mask)
    j = (np.asarray(j2d, np.float64) - c) * scale + c + shift
    return np.asarray(bg), j


def in_view(j2d):
    return (j2d[:, 0] >= VIEW_X0) & (j2d[:, 0] < VIEW_X1) & (j2d[:, 1] >= 0) & (j2d[:, 1] < RES)


def framing(kind=None, severity=None):
    """E3 transform. kind in {'cutoff','far','close'} or None (random); severity 0..1.
    cutoff: shift so part of the body leaves the view (row 13)
    far:    zoom out, person becomes small (row 14)
    close:  zoom in, body overflows the frame (row 15)"""
    def fn(img, lab, rng):
        k = kind or rng.choice(["cutoff", "far", "close"])
        s = float(rng.uniform(0.2, 1.0) if severity is None else severity)
        j2d = lab["joints2d"]
        lo, hi = j2d.min(0), j2d.max(0)
        if k == "far":
            scale, shift = 1.0 - 0.65 * s, rng.normal(0, 10, 2)
        elif k == "close":
            scale = 1.0 + 1.0 * s
            centre = j2d[rng.integers(len(j2d))]                 # zoom towards a random body part
            shift = (np.array([RES / 2, RES / 2]) - centre) * (scale - 1) * rng.uniform(0.3, 1.0)
        else:
            scale = rng.uniform(0.9, 1.15)
            d = rng.integers(4)                                  # which side to cut
            span = (hi - lo) * scale
            amt = (0.15 + 0.5 * s)
            shift = np.zeros(2)
            if d == 0:   shift[1] = -(lo[1] - 0) - amt * span[1] + 0          # cut the top... move up
            elif d == 1: shift[1] = (RES - hi[1]) + amt * span[1]             # cut the bottom (legs)
            elif d == 2: shift[0] = -(lo[0] - VIEW_X0) - amt * span[0]        # cut the left
            else:        shift[0] = (VIEW_X1 - hi[0]) + amt * span[0]         # cut the right
            shift = shift + rng.normal(0, 4, 2)
        out, j = similarity(img, j2d, scale, shift, rng)
        inside = in_view(j)
        vis = np.asarray(lab["vis"], bool) & inside
        return out, {"joints2d": j.astype(np.float32), "vis": vis, "cond_framing": s,
                     "frac_out_of_view": float(1 - inside.mean())}
    return fn


def compose(*fns):
    def fn(img, lab, rng):
        upd_all = {}
        for f in fns:
            img, upd = f(img, {**lab, **upd_all}, rng)
            upd_all.update(upd)
        return img, upd_all
    return fn


def sometimes(fn, p):
    def g(img, lab, rng):
        if rng.uniform() < p:
            return fn(img, lab, rng)
        return img, {}
    return g
