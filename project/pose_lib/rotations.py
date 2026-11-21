"""Rotation helpers. numpy versions work on any leading shape; quaternions are (w, x, y, z)."""

import numpy as np


# ---------------- numpy ----------------
def aa_to_rotmat(aa):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    aa = np.asarray(aa, dtype=np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.divide(aa, th, out=np.zeros_like(aa), where=th > 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rotmat_to_quat(R):
    """(..., 3, 3) -> (..., 4) w-first, w >= 0. Vectorised Shepperd method."""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((len(R), 4))
    tr = np.trace(R, axis1=1, axis2=2)
    m0 = tr > 0
    s = np.sqrt(np.maximum(tr[m0] + 1, 1e-12)) * 2
    q[m0] = np.stack([s / 4, (R[m0, 2, 1] - R[m0, 1, 2]) / s, (R[m0, 0, 2] - R[m0, 2, 0]) / s,
                      (R[m0, 1, 0] - R[m0, 0, 1]) / s], -1)
    for idx in np.nonzero(~m0)[0]:
        r = R[idx]
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s_ = 2 * np.sqrt(max(1 + r[i, i] - r[j, j] - r[k, k], 1e-12))
        qq = np.empty(4)
        qq[0] = (r[k, j] - r[j, k]) / s_
        qq[1 + i] = s_ / 4
        qq[1 + j] = (r[j, i] + r[i, j]) / s_
        qq[1 + k] = (r[k, i] + r[i, k]) / s_
        q[idx] = qq
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.reshape(shp + (4,))


def quat_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def quat_to_aa(q):
    q = np.asarray(q, dtype=np.float64)
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, q[..., :1])
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-12) * ang


def aa_to_quat(aa):
    aa = np.asarray(aa, dtype=np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.divide(aa, th, out=np.zeros_like(aa), where=th > 1e-12)
    return np.concatenate([np.cos(th / 2), k * np.sin(th / 2)], -1)


def rotmat_to_aa(R):
    return quat_to_aa(rotmat_to_quat(R))


def geodesic_deg(R1, R2):
    """Angle between rotations, degrees. (..., 3, 3) each."""
    c = (np.einsum("...ij,...ij->...", R1, R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def project_to_so3(M):
    """Nearest rotation (chordal L2) to each (..., 3, 3) matrix, via SVD."""
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(M.shape[:-2] + (3,))
    D[..., 2] = d
    return (U * D[..., None, :]) @ Vt


def quat_fix_signs(q):
    """(T, ..., 4): flip signs so consecutive frames stay in the same hemisphere."""
    q = np.array(q, dtype=np.float64, copy=True)
    for t in range(1, len(q)):
        flip = np.sum(q[t] * q[t - 1], axis=-1) < 0
        q[t][flip] *= -1
    return q


# ---------------- torch ----------------
def aa_to_rotmat_t(aa):
    import torch
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = aa / theta
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    th = theta[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def gram_schmidt_t(M):
    """Rotation from the first two columns of M (the '6D' map). Unlike SVD it has stable gradients
    when M is already (nearly) a rotation, so it is used where we backpropagate through the projection."""
    import torch
    M = M.float()
    a, b = M[..., :, 0], M[..., :, 1]
    x = torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
    y = torch.nn.functional.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1, eps=1e-8)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], -1)


def project_to_so3_t(M):
    """SVD projection. Always fp32 with autocast off: SVD/det have no fp16 CUDA kernels."""
    import torch
    with torch.autocast(device_type=M.device.type, enabled=False):
        M = M.float()
        U, _, Vh = torch.linalg.svd(M)
        d = torch.sign(torch.det(U @ Vh))
        D = torch.ones(M.shape[:-2] + (3,), dtype=M.dtype, device=M.device)
        D[..., 2] = d
        return (U * D[..., None, :]) @ Vh
