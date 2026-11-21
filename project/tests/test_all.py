"""CPU tests for pose_lib. Run:  python -m pytest -q tests"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pose_lib import (convert, degrade, detector, experts, fusion, hmr2_ft, lora, metrics,  # noqa: E402
                      rotations, selector, smoothing, stats)
from pose_lib.format import PoseSeq, parse_name  # noqa: E402
from pose_lib.smpl import SMPLJointsNP, fake_smpl_npz, smpl_joints_torch  # noqa: E402

import toy  # noqa: E402

SAMPLES = "/home/claude/samples"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("rw2b"))
    smpl, data = toy.setup(tmp)
    return {"tmp": tmp, "smpl": smpl, "data": data, "fk": smpl_joints_torch(smpl), "fknp": SMPLJointsNP(smpl)}


# ---------------------------------------------------------------- rotations / smpl
def test_rotations_roundtrip():
    rng = np.random.default_rng(0)
    aa = rng.normal(0, 0.8, (500, 3))
    aa = aa[np.linalg.norm(aa, axis=1) < np.pi - 1e-3]
    assert np.allclose(rotations.rotmat_to_aa(rotations.aa_to_rotmat(aa)), aa, atol=1e-6)
    q = rotations.aa_to_quat(aa)
    assert np.allclose(rotations.quat_to_rotmat(q), rotations.aa_to_rotmat(aa), atol=1e-9)
    R = rotations.aa_to_rotmat(aa[:10])
    noisy = R + rng.normal(0, 1e-3, R.shape)
    assert np.allclose(rotations.project_to_so3(noisy), R, atol=5e-3)


def test_fk_torch_matches_numpy(env):
    rng = np.random.default_rng(1)
    go, bp, b = rng.normal(0, .5, (4, 3)), rng.normal(0, .3, (4, 23, 3)), rng.normal(0, 1, (4, 10))
    j_np = env["fknp"](go, bp, b)
    j_t = env["fk"](rotations.aa_to_rotmat_t(torch.tensor(go, dtype=torch.float32)),
                    rotations.aa_to_rotmat_t(torch.tensor(bp, dtype=torch.float32)), torch.tensor(b, dtype=torch.float32))
    assert np.abs(j_t.numpy() - j_np).max() < 1e-5


# ---------------------------------------------------------------- format / convert (real files from Drive)
@pytest.mark.skipif(not os.path.exists(f"{SAMPLES}/video_smpl.npz"), reason="sample files not present")
def test_convert_real_files(env, tmp_path):
    s = convert.from_smpl_npz(f"{SAMPLES}/video_smpl.npz", env["fknp"], model="hmr2", clip="video")
    assert s.T == 551 and s.joints3d_rel            # PHALP camera depth (~50 m) is flagged as unreliable
    p = s.save(tmp_path / "video__hmr2.npz")
    s2 = PoseSeq.load(p)
    assert np.allclose(s2.body_pose, s.body_pose) and s2.model == "hmr2"
    z = np.load(p)
    for k in ("global_orient", "body_pose", "camera", "frame_idx", "fps"):   # still readable by notebook 02
        assert k in z.files
    m = convert.from_mediapipe(f"{SAMPLES}/clipA_full.npz", fps=24, model="mediapipe_full", clip="clipA")
    assert m.n_frames == 378 and m.T < 378 and np.isfinite(m.joints3d).all()
    r = metrics.proxy_report(m)
    assert r["bone_cv"] > 0.01 and 0.85 < r["detection_rate"] < 0.95
    assert parse_name("/x/row07_desk_01__hmr2.npz") == ("row07_desk_01", "hmr2", 7)


def test_proxy_metrics_behave(env):
    T = 120
    rng = np.random.default_rng(0)
    go = np.tile([np.pi, 0, 0], (T, 1)) + np.linspace(0, 0.5, T)[:, None] * [0, 1, 0]
    bp = np.zeros((T, 23, 3)); bp[:, 17, 2] = np.sin(np.linspace(0, 6, T))
    cam = np.stack([np.zeros(T), np.zeros(T), np.full(T, 4.0)], 1)
    clean = convert.from_params(go, bp, env["fknp"], model="x", clip="c", fps=30, camera=cam)
    noisy = convert.from_params(go, bp + rng.normal(0, .08, bp.shape), env["fknp"], model="x", clip="c", fps=30,
                                camera=cam)
    assert metrics.joint_jitter(noisy) > 3 * metrics.joint_jitter(clean)
    assert metrics.rotation_jitter(noisy) > 3 * metrics.rotation_jitter(clean)
    sm = smoothing.smooth_seq(noisy, fk=env["fknp"])
    assert metrics.rotation_jitter(sm) < 0.5 * metrics.rotation_jitter(noisy)
    # left/right swap detector
    sw = noisy.subset(np.ones(T, bool)); j = sw.joints3d.copy()
    j[60:, [7, 8]] = j[60:, [8, 7]]; sw.joints3d = j
    assert metrics.left_right_swaps(sw) >= 1
    # gaps
    g = noisy.subset(np.r_[np.ones(50, bool), np.zeros(10, bool), np.ones(60, bool)])
    assert metrics.gaps(g) == (1, 10)
    assert metrics.agreement(clean, clean) < 1e-6


# ---------------------------------------------------------------- stats
def test_stats():
    rng = np.random.default_rng(0)
    a = [rng.normal(50, 10, 500) for _ in range(3)]
    b = [x + 3 + rng.normal(0, 1, 500) for x in a]
    r = stats.seed_comparison(a, b)
    assert r["robust"] and r["diff_mean"] < 0
    c = [x + rng.normal(0, 1, 500) * 0 + (1 if i == 0 else -1) for i, x in enumerate(a)]
    assert not stats.seed_comparison(a, c)["same_sign_all_seeds"]
    assert stats.bootstrap_p(a[0], b[0]) < 0.01
    assert np.all(np.diff(stats.holm([0.01, 0.04, 0.03])[[0, 2, 1]]) >= 0)


# ---------------------------------------------------------------- degradations
def test_degrade_labels(env):
    ds = hmr2_ft.SynthDataset(f"{env['data']}/test/B_clean")
    img = ds._raw(int(ds.idx[0])); lab = ds.item_labels(int(ds.idx[0]))
    rng = np.random.default_rng(0)
    for kind in ("blur", "motion", "dark", "lowres", "mixed"):
        out, upd = degrade.quality(kind, 0.8)(img, lab, rng)
        assert out.shape == img.shape and out.dtype == np.uint8 and upd["cond_quality"] == 0.8
    # framing: a pure shift must move the joints by exactly the shift
    out, j = degrade.similarity(img, lab["joints2d"], 1.0, np.array([10.0, -5.0]), rng)
    assert np.allclose(j - lab["joints2d"], [10, -5])
    # and the image content moves with it (a bright joint marker stays under its label)
    x, y = lab["joints2d"][0]
    if 20 < x < 230 and 20 < y < 230:
        assert out[int(round(y - 5)), int(round(x + 10))].mean() > 150
    for kind in ("cutoff", "far", "close"):
        out, upd = degrade.framing(kind, 0.9)(img, lab, rng)
        assert upd["vis"].dtype == bool and upd["joints2d"].shape == (24, 2)
    _, upd = degrade.framing("cutoff", 1.0)(img, lab, np.random.default_rng(3))
    assert upd["frac_out_of_view"] > 0


def test_dataset_transform_deterministic(env):
    ds = hmr2_ft.SynthDataset(f"{env['data']}/test/B_clean", transform=degrade.quality(), transform_seed=5)
    assert torch.equal(ds[0]["img"], ds[0]["img"])
    assert float(ds[0]["cond_quality"]) > 0
    assert "selfvis" in ds[0]


# ---------------------------------------------------------------- LoRA
def test_lora_inject_and_param_match():
    m = toy.ToyHMR()
    x = torch.randn(2, 3, 256, 256)
    base = hmr2_ft.predict(m, x)["body_pose"]
    lora.inject(m, lora.hmr2_targets(n_blocks=2), ["E0", "E1", "E2", "E3"], r=4)
    lora.set_active(m, "E1")
    assert torch.allclose(hmr2_ft.predict(m, x)["body_pose"], base, atol=1e-6)   # B=0 at init -> identical
    n_exp = lora.adapter_numel(m, "E0")
    lora.add_adapter(m, "gen", r=lora.generalist_rank(4, 4))
    assert lora.adapter_numel(m, "gen") == 4 * n_exp                              # same total params
    # only adapter params train
    ps = lora.trainable_fn("E1")(m)
    assert all(p.requires_grad for p in ps)
    assert not any(p.requires_grad for n, p in m.named_parameters() if ".A." not in n and ".B." not in n)


def test_train_eval_blocks_and_lora(env, tmp_path):
    torch.set_num_threads(2)
    d = env["data"]
    m = toy.ToyHMR()
    h = hmr2_ft.train(m, f"{d}/train/D_occ3d", f"{d}/train/D_occ3d", env["fk"], str(tmp_path / "D.pt"),
                      epochs=2, batch_size=8, lr=1e-3, n_blocks=2, device="cpu", workers=0, amp=False, log_every=1000)
    assert len(h) == 2 and os.path.exists(tmp_path / "D.pt")
    m2 = toy.ToyHMR(); m2.load_state_dict(m.state_dict())
    hmr2_ft.load_trainable(m2, str(tmp_path / "D.pt"))
    r = hmr2_ft.evaluate(m2, hmr2_ft.SynthDataset(f"{d}/test/D_occ3d"), env["fk"], device="cpu", workers=0,
                         amp=False, quiet=True, return_preds=True)
    assert r["pa_joint"].shape[1] == 22 and "selfvis" in r and r["body_pose"].shape[1:] == (23, 3, 3)
    s = metrics.summarize(r)
    assert "PA joints: hidden by object" in s or "PA hidden joints" in s
    # LoRA expert training with a degraded dataset
    m3 = toy.ToyHMR()
    lora.inject(m3, lora.hmr2_targets(2), ["E0", "E1", "E2", "E3"], r=4)
    tr, va = experts.expert_datasets(d, "E2", seed=0)
    w_before = {n: p.clone() for n, p in m3.named_parameters()}
    hmr2_ft.train(m3, None, None, env["fk"], str(tmp_path / "E2.pt"), epochs=1, batch_size=8, lr=1e-3,
                  device="cpu", workers=0, amp=False, train_ds=tr, val_ds=va, trainable=lora.trainable_fn("E2"),
                  save_fn=lambda mm, p: lora.save_adapter(mm, "E2", p), log_every=1000)
    changed = [n for n, p in m3.named_parameters() if not torch.equal(p, w_before[n])]
    assert changed and all(".E2" in n for n in changed)
    lora.load_adapter(m3, "E2", str(tmp_path / "E2.pt"))


def test_fusion_properties(env):
    torch.manual_seed(0)
    B = 5

    def rand_pred():
        aa = torch.randn(B, 24, 3) * 0.4
        R = rotations.aa_to_rotmat_t(aa)
        return {"global_orient": R[:, 0], "body_pose": R[:, 1:], "betas": torch.randn(B, 10),
                "cam": torch.rand(B, 3) + 0.5}

    p = [rand_pred() for _ in range(4)]
    # one-hot weights reproduce that expert exactly
    w = torch.zeros(B, 4); w[:, 2] = 1
    f = fusion.fuse_weighted(p, w)
    assert torch.allclose(f["body_pose"], p[2]["body_pose"], atol=1e-5)
    # identical experts -> fused equals them for any weights
    same = [p[0]] * 4
    f2 = fusion.fuse_weighted(same, torch.rand(B, 4))
    assert torch.allclose(f2["body_pose"], p[0]["body_pose"], atol=1e-5)
    # outputs are rotations
    f3 = fusion.fuse_per_joint(p, torch.rand(B, 4), torch.rand(B, 24))
    RtR = f3["body_pose"].transpose(-1, -2) @ f3["body_pose"]
    assert torch.allclose(RtR, torch.eye(3).expand_as(RtR), atol=1e-4)
    assert torch.allclose(torch.det(f3["body_pose"]), torch.ones(B, 23), atol=1e-4)
    # per-joint: hidden joint -> occlusion expert dominates
    wj = fusion.per_joint_weights(torch.full((1, 4), .25), torch.tensor([[0.0] + [1.0] * 23]))
    assert wj[0, 1, 0] > wj[0, 0, 0] and wj[0, 1, 5] < wj[0, 0, 5]
    # learned fusion trains and runs
    N = 64
    preds = [rand_pred() for _ in range(4)]
    R, b, c = fusion._stack([{k: v.repeat(13, *[1] * (v.dim() - 1))[:N] for k, v in q.items()} for q in preds])
    gt = env["fk"](R[:, 1, 0], R[:, 1, 1:], b[:, 1])       # expert 1 is always right
    cache = {"R": R, "betas": b, "cam": c, "cond": torch.rand(N, 3), "vis": torch.rand(N, 24),
             "occfrac": torch.rand(N), "joints3d": gt}
    net, best = fusion.train_learned_fusion(cache, env["fk"], epochs=60)
    w = net(cache["cond"][:4], cache["vis"][:4], cache["occfrac"][:4])
    assert w.shape == (4, 4, 24) and float(w[:, 1].mean().detach()) > 0.5


def test_detector_and_dgs(env, tmp_path):
    d = env["data"]
    srcs = [hmr2_ft.SynthDataset(f"{d}/train/B_clean"), hmr2_ft.SynthDataset(f"{d}/train/D_occ3d"),
            hmr2_ft.SynthDataset(f"{d}/train/B_clean", transform=degrade.quality(), deterministic=False),
            hmr2_ft.SynthDataset(f"{d}/train/B_clean", transform=degrade.framing(), deterministic=False)]
    mix = detector.MixtureDataset(srcs, [24, 24, 24, 24])
    det = detector.Detector("tiny")
    detector.train(det, mix, mix, str(tmp_path / "det.pt"), epochs=1, batch_size=16, device="cpu", workers=0)
    w = detector.router_weights(torch.tensor([[0., 0., 0.], [1., 0., 0.], [.5, .5, 0.]]))
    assert torch.allclose(w.sum(1), torch.ones(3)) and w[0, 0] == 1 and w[1, 1] == 1
    assert detector.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    # full DGS on a toy model
    m = toy.ToyHMR()
    lora.inject(m, lora.hmr2_targets(2), ["E0", "E1", "E2", "E3"], r=4)
    ds = hmr2_ft.SynthDataset(f"{d}/test/D_occ3d")
    for mode in ("weighted", "per_joint", "top1"):
        dgs = experts.DGS(m, det, mode=mode)
        r = hmr2_ft.evaluate(None, ds, env["fk"], device="cpu", workers=0, amp=False, quiet=True, predict_fn=dgs)
        assert np.isfinite(r["mpjpe"]).all()
    cache = experts.cache_for_fusion(m, det, ds, device="cpu", workers=0)
    assert cache["R"].shape == (len(ds), 4, 24, 3, 3)


def test_selector_strategies():
    rng = np.random.default_rng(0)
    occ = rng.uniform(0.05, 1.0, 400)
    arms, names = selector.occfrac_arms(occ)
    pool = selector.ArmPool(arms, names)
    seen = {}

    def loss_of(idx):                      # the 'model' is bad on heavy occlusion
        return occ[idx] ** 2

    for strat in ("random", "hem", "bandit"):
        got = []
        out = selector.run(strat, pool, rounds=8, round_size=20, train_on=got.append, loss_of=loss_of, log=None)
        seen[strat] = np.mean(occ[np.concatenate(got)])
        assert sum(len(g) for g in got) == 160                     # identical budget
    assert seen["hem"] > seen["random"] and seen["bandit"] > seen["random"]
    assert out["bandit_probs"] is not None


# ---------------------------------------------------------------- video pipeline (YOLO + HMR2 crop stubbed)
def test_video_pipeline_with_stubs(env, tmp_path, monkeypatch):
    import types
    import cv2
    from pose_lib import video
    # synthetic clip: a bright rectangle 'person' walking right, 40 frames, 320x240
    W_, H_ = 320, 240
    path = str(tmp_path / "row01_walk_01.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W_, H_))
    boxes = []
    for t in range(40):
        f = np.full((H_, W_, 3), 60, np.uint8)
        x0 = 40 + 4 * t
        cv2.rectangle(f, (x0, 40), (x0 + 50, 200), (220, 220, 220), -1)
        vw.write(f); boxes.append([x0, 40, x0 + 50, 200])
    vw.release()

    class FakeBoxes:
        def __init__(self, b):
            self.xyxy = torch.tensor([b], dtype=torch.float32); self.conf = torch.tensor([0.9])
        def __len__(self):
            return 1

    class FakeYOLO:
        def __call__(self, f, **kw):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) > 150
            ys, xs = np.nonzero(g)
            return [types.SimpleNamespace(boxes=FakeBoxes([xs.min(), ys.min(), xs.max(), ys.max()]))]

    class FakeViTDet:                         # mimics hmr2.datasets.vitdet_dataset.ViTDetDataset items
        def __init__(self, cfg, img, b):
            self.img, self.b = img, np.asarray(b, np.float32)
        def __getitem__(self, i):
            x0, y0, x1, y1 = self.b[i]
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2]); s = max(x1 - x0, y1 - y0) * 1.2
            M = cv2.getAffineTransform(np.float32([c - s / 2, [c[0] + s / 2, c[1] - s / 2], c + s / 2]),
                                       np.float32([[0, 0], [256, 0], [256, 256]]))
            p = cv2.warpAffine(self.img, M, (256, 256))[:, :, ::-1].astype(np.float32) / 255
            p = (p - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            return {"img": p.transpose(2, 0, 1).astype(np.float32), "box_center": c, "box_size": s}

    mod = types.ModuleType("hmr2.datasets.vitdet_dataset"); mod.ViTDetDataset = FakeViTDet
    for name in ("hmr2", "hmr2.datasets"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "hmr2.datasets.vitdet_dataset", mod)

    m = toy.ToyHMR()
    lora.inject(m, lora.hmr2_targets(2), ["E0", "E1", "E2", "E3"], r=4)
    det = detector.Detector("tiny")
    preds = {"hmr2": video.base_predictor(m), "E1": video.expert_predictor(m, "E1"),
             "dgs": experts.DGS(m, det, mode="per_joint")}
    paths, timing = video.run(path, str(tmp_path / "out"), preds, None, env["fknp"], yolo=FakeYOLO(),
                              target_h=None, device="cpu", min_track_len=10)
    assert set(paths) == {"hmr2", "E1", "dgs"}
    s = PoseSeq.load(paths["dgs"][0])
    assert s.T == 40 and s.n_frames == 40 and s.clip == "row01_walk_01"
    assert "extra_chk_blur" in s.extra and "extra_dgs_weights" in s.extra
    assert s.extra["extra_dgs_weights"].shape == (40, 4)
    # camera x should increase as the person walks right
    assert np.corrcoef(np.arange(40), s.camera[:, 0])[0, 1] > 0.9
    r = metrics.proxy_report(s)
    assert r["detection_rate"] == 1.0


def test_tracker_and_conditions():
    from pose_lib import conditions, video
    tr = video.Tracker()
    a = tr.update(0, [np.array([0, 0, 10, 20]), np.array([50, 0, 60, 20])])
    b = tr.update(1, [np.array([51, 0, 61, 20]), np.array([1, 0, 11, 20])])
    assert b == [a[1], a[0]]
    sharp = np.zeros((200, 200, 3), np.uint8); sharp[::4] = 255
    import cv2
    blurry = cv2.GaussianBlur(sharp, (0, 0), 6)
    box = [20, 20, 180, 180]
    assert conditions.blur(sharp, box)[1] < conditions.blur(blurry, box)[1]
    assert conditions.brightness(sharp // 10, box)[1] > conditions.brightness(sharp, box)[1]
    assert conditions.cutoff([0, 10, 100, 199], 200, 200)[0] == 2
    g0 = np.random.default_rng(0).uniform(0, 255, (120, 160)).astype(np.uint8)
    g1 = np.roll(g0, 6, axis=1)
    d, _ = conditions.camera_motion(g0, g1)
    assert 4 < d < 8


def test_rq3_adapter_trainer_and_labels_merge(env, tmp_path):
    from pose_lib import labels
    d = env["data"]
    m = toy.ToyHMR()
    lora.inject(m, lora.hmr2_targets(2), ["E1"], r=4)
    pool_ds = hmr2_ft.SynthDataset(f"{d}/train/D_occ3d", augment=True)
    tr = selector.AdapterTrainer(m, "E1", pool_ds, env["fk"], lr=1e-3, batch_size=4, steps_per_round=3,
                                 device="cpu", amp=False, workers=0)
    arms, names = selector.occfrac_arms(pool_ds.L["occfrac"])
    pool = selector.ArmPool([np.intersect1d(a, pool_ds.idx) for a in arms], names)
    out = selector.run("bandit", pool, rounds=2, round_size=8, train_on=tr.train_on, loss_of=tr.loss_of,
                       probe=4, log=None)
    assert len(tr.seen) == 16 and len(out["chosen"]) == 2
    # labels merge: fake a generator output that matches the test set
    L = dict(np.load(f"{d}/test/B_clean/labels.npz"))
    g = tmp_path / "gen0"; g.mkdir()
    np.savez(g / "labels.npz", start=0, joints3d=L["joints3d"], selfvis_joint=np.ones((len(L["names"]), 24), bool),
             selfvis_frac=np.ones((len(L["names"]), 24), np.float32))
    import shutil
    c = tmp_path / "B_clean"; shutil.copytree(f"{d}/test/B_clean", c)
    rep = labels.merge_selfvis([str(g)], [str(c)])
    assert rep["B_clean"]["joints hidden by the body"] == 0.0
    bad = tmp_path / "gen_bad"; bad.mkdir()
    np.savez(bad / "labels.npz", start=0, joints3d=L["joints3d"] + 0.1, selfvis_joint=np.ones((len(L["names"]), 24), bool),
             selfvis_frac=np.ones((len(L["names"]), 24), np.float32))
    with pytest.raises(ValueError):
        labels.merge_selfvis([str(bad)], [str(c)])


def test_oracle_routing(env):
    d = env["data"]
    m = toy.ToyHMR()
    lora.inject(m, lora.hmr2_targets(2), ["E0", "E1", "E2", "E3"], r=4)
    ds = hmr2_ft.SynthDataset(f"{d}/test/D_occ3d")
    for mode in ("weighted", "per_joint"):
        dgs = experts.DGS(m, None, mode=mode)
        r = hmr2_ft.evaluate(None, ds, env["fk"], device="cpu", workers=0, amp=False, quiet=True,
                             predict_fn_batch=dgs.oracle)
        assert np.isfinite(r["pa_mpjpe"]).all()


def test_fusion_under_autocast_cpu_bf16():
    # Colab runs evaluation under autocast; fusion must stay fp32 (SVD/det have no half kernels on CUDA)
    torch.manual_seed(0)
    R = rotations.aa_to_rotmat_t(torch.randn(3, 24, 3) * 0.3)
    p = {"global_orient": R[:, 0], "body_pose": R[:, 1:], "betas": torch.randn(3, 10), "cam": torch.rand(3, 3) + .5}
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        f = fusion.fuse_per_joint([p, p], torch.rand(3, 2), torch.rand(3, 24), occ_index=1)
    assert f["body_pose"].dtype == torch.float32
    assert torch.allclose(f["body_pose"], p["body_pose"], atol=1e-4)


def test_threedpw_camera_frame(tmp_path):
    from pose_lib.smpl import fake_smpl_npz, SMPLJointsNP
    from pose_lib import threedpw as tp
    from pose_lib.rotations import aa_to_rotmat
    fake_smpl_npz(str(tmp_path / "s.npz"))
    fknp = SMPLJointsNP(str(tmp_path / "s.npz"))
    rng = np.random.default_rng(1)
    T = 4
    cam = np.tile(np.eye(4), (T, 1, 1))
    cam[:, :3, :3] = aa_to_rotmat(rng.normal(0, 0.5, (T, 3)))
    cam[:, :3, 3] = [0.1, -0.2, 5]
    seq = {"poses": [rng.normal(0, 0.3, (T, 72))], "trans": [rng.normal(0, 0.5, (T, 3))],
           "betas": [rng.normal(0, 1, 10)], "cam_poses": cam}
    j_rot, j_full, _ = tp.gt_joints(seq, 0, fknp, np.arange(T))
    rel_a = j_rot - j_rot[:, :1]
    rel_b = j_full - j_full[:, :1]
    assert np.allclose(rel_a, rel_b, atol=1e-9)        # camera-orientation joints match the projected ones
