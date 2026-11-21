"""Toy stand-ins for CPU tests: a model with HMR2.0's structure and a dataset folder laid out like
notebook 04's output (images/ + labels.npz). No licensed data, no GPU."""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pose_lib.rotations import aa_to_rotmat_t
from pose_lib.smpl import SMPLJointsNP, fake_smpl_npz


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, -1)
        a = torch.softmax(q @ k.transpose(1, 2) / q.shape[-1] ** 0.5, -1)
        return self.proj(a @ v)


class Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn, self.mlp = Attn(d), Mlp(d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Backbone(nn.Module):
    def __init__(self, d=32, n=6):
        super().__init__()
        self.patch = nn.Conv2d(3, d, 32, 32)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.last_norm = nn.LayerNorm(d)

    def forward(self, x):                      # x: (B,3,256,192)
        f = self.patch(x).flatten(2).transpose(1, 2)
        for b in self.blocks:
            f = b(f)
        return self.last_norm(f)


class Head(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.tok = nn.Linear(d, d)
        self.decpose = nn.Linear(d, 24 * 3)
        self.decshape = nn.Linear(d, 10)
        self.deccam = nn.Linear(d, 3)
        nn.init.zeros_(self.decpose.weight); nn.init.zeros_(self.decpose.bias)

    def forward(self, f):
        h = torch.relu(self.tok(f.mean(1)))
        aa = self.decpose(h).view(-1, 24, 3)
        R = aa_to_rotmat_t(aa)
        cam = self.deccam(h) + torch.tensor([0.9, 0.0, 0.0])
        return {"global_orient": R[:, :1], "body_pose": R[:, 1:], "betas": self.decshape(h)}, cam, None


class ToyHMR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.smpl_head = Backbone(), Head()


def make_dataset(root, smpl_npz, n_train=48, n_test=24, seed=0):
    """Write data/train|test/{B_clean,C_paste2d,D_occ3d}/{images,labels.npz} like synth_post.py."""
    rng = np.random.default_rng(seed)
    fk = SMPLJointsNP(smpl_npz)
    for split, n in (("train", n_train), ("test", n_test)):
        go = rng.normal(0, 0.3, (n, 3)); go[:, 0] += np.pi
        bp = rng.normal(0, 0.25, (n, 23, 3))
        betas = rng.normal(0, 1, (n, 10)).astype(np.float32)
        j3 = fk(go, bp, betas) + np.array([0, 0, 4.0])
        j2 = 600 * j3[..., :2] / j3[..., 2:] + 128
        occ = rng.uniform(0.05, 0.7, n)
        vis = rng.uniform(size=(n, 24)) > occ[:, None]
        base = dict(names=np.array([f"{i:06d}" for i in range(n)]), global_orient=go, body_pose=bp, betas=betas,
                    joints3d=j3, joints2d=j2, focal=np.full(n, 600.0), n_occ=np.ones(n, int),
                    pose_idx=np.arange(n), occ_ok=np.ones(n, bool),
                    selfvis_joint=rng.uniform(size=(n, 24)) > 0.2)
        for c in ("B_clean", "C_paste2d", "D_occ3d"):
            d = os.path.join(root, split, c, "images")
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                img = (rng.uniform(0, 255, (256, 256, 3))).astype(np.uint8)
                for (x, y) in j2[i]:
                    xi, yi = int(np.clip(x, 2, 253)), int(np.clip(y, 2, 253))
                    img[yi - 2:yi + 3, xi - 2:xi + 3] = 255
                Image.fromarray(img).save(os.path.join(d, f"{i:06d}.jpg"))
            o = occ if c != "B_clean" else np.zeros(n)
            v = vis if c != "B_clean" else np.ones((n, 24), bool)
            np.savez(os.path.join(root, split, c, "labels.npz"), **base, vis=v, occfrac=o)
    return root


def setup(tmp):
    smpl = fake_smpl_npz(os.path.join(tmp, "smpl_fake.npz"))
    make_dataset(os.path.join(tmp, "data"), smpl)
    return smpl, os.path.join(tmp, "data")
