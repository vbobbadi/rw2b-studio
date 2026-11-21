"""
Synthetic occlusion data generator (Blender, headless).

[selfocc] marks the additions for the per-joint self-occlusion label (--selfvis-joints). Without the
flag this script behaves exactly like notebook 04's version.

    blender --background --python synth_gen.py -- \
        --model smpl_neutral.npz --poses poses_train.npz --out /content/synth/train \
        --n 5000 --seed 0

Per sample it renders the SAME person/pose/shape/camera/lighting twice:
    clean_XXXXXX.png   person only (transparent background)
    occ3d_XXXXXX.png   same scene with 1-3 random 3D occluders (boxes, poles, boards, chairs, tables, balls)
and records ground truth in labels.npz (camera = OpenCV convention: x right, y down, z forward):
    global_orient (N,3)  body_pose (N,23,3)  betas (N,10)     SMPL params, orient in camera frame
    joints3d (N,24,3)    joints2d (N,24,2)   focal (N,)       SMPL joints in camera coords / pixels
    verts2d (N,V,2)      selfvis (N,V)                         subsampled vertices + visible-without-occluders flag
    vis_occ3d (N,24)     occfrac_occ3d (N,)  n_occ (N,)        joint visibility / fraction of visible body hidden
    pose_idx (N,)

    --mode cutouts renders occluders alone (for the 2D-paste baseline):  cutout_XXXXXX.png
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smplx_mocap import (build_body, enable_gpu, joint_names, load_body_model, quat_mul,  # noqa: E402
                         quat_to_rotmat, rotmat_to_rotvec, rotvec_to_quat)

RES = 256
N_VERT_SAMPLES = 600
OCC_TYPES = ["box", "pole", "board", "chair", "table", "ball"]
JV_PER_JOINT = 16          # [selfocc] rays per joint for the self-occlusion label
SELFVIS_MIN = 0.25         # [selfocc] joint counts as visible-to-camera if >= 25% of its surface rays are
# AMASS root orientation is in a Z-up world; our armature space is SMPL's Y-up frame.
Q_ZUP_TO_YUP = np.array([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0])

# dominant joint -> clothing region (SMPL joint indices)
REGION = np.zeros(24, dtype=np.int64)          # 0 skin, 1 shirt, 2 pants, 3 shoes
REGION[[3, 6, 9, 13, 14, 16, 17, 18, 19]] = 1
REGION[[0, 1, 2, 4, 5]] = 2
REGION[[7, 8, 10, 11]] = 3
REGION[[12, 15, 20, 21, 22, 23]] = 0


# ---------------- helpers ----------------
def look_at(cam_pos, target):
    """Blender camera rotation (columns = camera X right, Y up, Z back) looking from cam_pos at target."""
    f = target - cam_pos
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    if abs(f @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(f, up); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.stack([x, y, -f], axis=1)


def to_cv(points, cam_pos, R_bl):
    """world points -> OpenCV camera coords."""
    p = (points - cam_pos) @ R_bl          # Blender camera coords (x right, y up, z back)
    return p * np.array([1.0, -1.0, -1.0])


def project(p_cv, focal):
    return focal * p_cv[:, :2] / p_cv[:, 2:3] + RES / 2


def rand_color(rng, sat=(0.2, 0.9), val=(0.15, 0.9)):
    import colorsys
    return (*colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(*sat), rng.uniform(*val)), 1.0)


def skin_color(rng):
    t = rng.uniform()
    light, dark = np.array([0.93, 0.76, 0.65]), np.array([0.30, 0.18, 0.11])
    return (*(light * (1 - t) + dark * t), 1.0)


def make_material(bpy, name, color, rng, noise=True):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.3, 0.9)
    if noise:  # subtle procedural variation so surfaces aren't flat
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = rng.uniform(2, 30)
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["Factor"].default_value = rng.uniform(0.05, 0.35)
        mix.inputs["A"].default_value = color
        mix.inputs["B"].default_value = tuple(c * 0.5 for c in color[:3]) + (1.0,)
        if rng.uniform() < 0.3:
            nt.links.new(tex.outputs["Fac"], mix.inputs["Factor"])
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = color
    return mat


def set_material_color(mat, color):
    nt = mat.node_tree
    mix = next((n for n in nt.nodes if n.type == "MIX"), None)
    if mix is not None:
        mix.inputs["A"].default_value = color
        mix.inputs["B"].default_value = tuple(c * 0.5 for c in color[:3]) + (1.0,)
    else:
        nt.nodes["Principled BSDF"].inputs["Base Color"].default_value = color


# ---------------- occluders ----------------
def _prim(bpy, kind, size, loc, name):
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    elif kind == "cyl":
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.5, depth=1.0, location=loc)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.5, location=loc)
    o = bpy.context.active_object
    o.scale = size
    o.name = name
    return o


def make_occluder(bpy, kind, rng, idx):
    """Returns a list of objects (local origin at the bottom-centre for floor objects), all named occ_*."""
    n = f"occ_{idx}"
    parts = []
    if kind == "box":
        s = rng.uniform(0.2, 0.8, 3)
        parts.append(_prim(bpy, "cube", s, (0, 0, s[2] / 2), n))
    elif kind == "pole":
        r, h = rng.uniform(0.05, 0.18), rng.uniform(1.6, 2.6)
        parts.append(_prim(bpy, "cyl", (2 * r, 2 * r, h), (0, 0, h / 2), n))
    elif kind == "board":
        w, h = rng.uniform(0.4, 1.2), rng.uniform(0.4, 1.4)
        parts.append(_prim(bpy, "cube", (w, 0.04, h), (0, 0, h / 2), n))
    elif kind == "chair":
        sw, sh = rng.uniform(0.4, 0.55), rng.uniform(0.4, 0.5)
        bh = rng.uniform(0.35, 0.55)
        parts.append(_prim(bpy, "cube", (sw, sw, 0.05), (0, 0, sh), n + "_seat"))
        parts.append(_prim(bpy, "cube", (sw, 0.05, bh), (0, sw / 2, sh + bh / 2), n + "_back"))
        for i, (dx, dy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            parts.append(_prim(bpy, "cube", (0.04, 0.04, sh), (dx * sw / 2.3, dy * sw / 2.3, sh / 2), f"{n}_leg{i}"))
    elif kind == "table":
        w, d, h = rng.uniform(0.8, 1.6), rng.uniform(0.5, 0.9), rng.uniform(0.7, 0.8)
        parts.append(_prim(bpy, "cube", (w, d, 0.05), (0, 0, h), n + "_top"))
        for i, (dx, dy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            parts.append(_prim(bpy, "cube", (0.05, 0.05, h), (dx * w / 2.2, dy * d / 2.2, h / 2), f"{n}_leg{i}"))
    elif kind == "ball":
        r = rng.uniform(0.15, 0.4)
        parts.append(_prim(bpy, "sphere", (2 * r, 2 * r, 2 * r), (0, 0, r), n))
    return parts


def place_parts(parts, pos, yaw):
    from mathutils import Matrix, Vector
    M = Matrix.Translation(Vector(pos)) @ Matrix.Rotation(yaw, 4, "Z")
    for o in parts:
        o.matrix_world = M @ o.matrix_world


def remove_objects(bpy, objs):
    for o in objs:
        me = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if me is not None and me.users == 0:
            if isinstance(me, bpy.types.Mesh):
                bpy.data.meshes.remove(me)
            elif isinstance(me, bpy.types.Armature):
                bpy.data.armatures.remove(me)


# ---------------- generator ----------------
class Generator:
    def __init__(self, args):
        import bpy
        self.bpy = bpy
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.rng_sv = np.random.default_rng([args.seed, 4747])   # [selfocc]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = self.scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 4
        scene.render.film_transparent = True
        scene.render.resolution_x = scene.render.resolution_y = RES
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.view_settings.view_transform = "Standard"
        backend = enable_gpu(scene) if not args.cpu else "CPU"
        print(f"[synth] rendering on {backend}")

        world = bpy.data.worlds.new("world"); world.use_nodes = True; scene.world = world
        self.bg_node = world.node_tree.nodes["Background"]
        self.sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        scene.collection.objects.link(self.sun)
        self.cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        self.cam.data.sensor_fit = "HORIZONTAL"
        self.cam.data.sensor_width = 36.0
        self.cam.data.clip_start = 0.05
        scene.collection.objects.link(self.cam)
        scene.camera = self.cam

        self.arm = self.body = None
        self.mats = [make_material(bpy, f"region{i}", (0.5, 0.5, 0.5, 1.0), self.rng) for i in range(4)]
        self.occ_mats = [make_material(bpy, f"occ{i}", (0.5, 0.5, 0.5, 1.0), self.rng) for i in range(6)]

    # ---- scene randomization ----
    def new_body(self):
        bpy = self.bpy
        if self.arm is not None:
            remove_objects(bpy, [self.body, self.arm])
        betas = np.clip(self.rng.normal(0, 1, 10), -2.5, 2.5)
        betas[0] = np.clip(self.rng.normal(0, 1.3), -3, 3)  # height/size varies most
        v, f, J, W, parents = load_body_model(self.args.model, betas=betas)
        self.arm, self.body = build_body(bpy, v, f, J, W, parents, name="person")
        self.betas = betas
        me = self.body.data
        me.materials.clear()
        for m in self.mats:
            me.materials.append(m)
        dom = W.argmax(1)
        region_of_face = REGION[dom[f[:, 0]]]
        me.polygons.foreach_set("material_index", region_of_face.tolist())
        self.vert_idx = self.rng.choice(len(v), N_VERT_SAMPLES, replace=False)
        # [selfocc] surface vertices per joint (dominant skin weight), from a SEPARATE rng so the main
        # random stream - and therefore every image and label - is unchanged when --selfvis-joints is on
        dom = W.argmax(1)
        self.joint_verts = [self.rng_sv.choice(np.nonzero(dom == j)[0], size=min(JV_PER_JOINT, int((dom == j).sum())),
                                               replace=False) if (dom == j).any() else np.zeros(0, int)
                            for j in range(24)]
        self.names = joint_names(24)          # SMPL joint order (pose.bones order is not guaranteed)

    def pose_body(self, pose72):
        q = rotvec_to_quat(pose72.reshape(24, 3))
        q[0] = quat_mul(Q_ZUP_TO_YUP, q[0])
        # random heading so the camera sees all sides
        yaw = self.rng.uniform(0, 2 * math.pi)
        q_yaw = np.array([math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0])   # about SMPL Y (up)
        q[0] = quat_mul(q_yaw, q[0])
        for j, n in enumerate(self.names):
            self.arm.pose.bones[n].rotation_quaternion = q[j].tolist()
        self.bpy.context.view_layer.update()

    def posed_geometry(self):
        bpy = self.bpy
        dg = bpy.context.evaluated_depsgraph_get()
        ev = self.body.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        co = co.reshape(-1, 3)
        Mw = np.array(self.body.matrix_world)
        verts = co @ Mw[:3, :3].T + Mw[:3, 3]
        Ma = np.array(self.arm.matrix_world)
        joints = np.array([list(self.arm.pose.bones[n].head) for n in self.names]) @ Ma[:3, :3].T + Ma[:3, 3]
        return verts, joints

    def randomize_look(self):
        rng = self.rng
        for m, c in zip(self.mats, [skin_color(rng), rand_color(rng), rand_color(rng), rand_color(rng, val=(0.05, 0.5))]):
            set_material_color(m, c)
        el, az = math.radians(rng.uniform(20, 80)), rng.uniform(0, 2 * math.pi)
        self.sun.rotation_euler = (math.pi / 2 - el, 0, az)
        self.sun.data.energy = rng.uniform(1.5, 5.0)
        self.sun.data.angle = math.radians(rng.uniform(0.5, 15))
        g = rng.uniform(0.3, 1.0)
        tint = rng.uniform(0.85, 1.15, 3) * g
        self.bg_node.inputs["Color"].default_value = (*np.clip(tint, 0, 1), 1.0)
        self.bg_node.inputs["Strength"].default_value = rng.uniform(0.5, 1.5)

    def place_camera(self, verts):
        rng = self.rng
        center = (verts.min(0) + verts.max(0)) / 2
        height = verts[:, 2].max() - verts[:, 2].min()
        az = rng.uniform(0, 2 * math.pi)
        el = math.radians(rng.uniform(-10, 35))
        dist = rng.uniform(2.5, 7.0)
        cam_pos = center + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        target = center + rng.normal(0, 0.06 * max(height, 0.5), 3)    # crop jitter
        R_bl = look_at(cam_pos, target)
        p = to_cv(verts, cam_pos, R_bl)
        # fit: HMR2 sees the central 192 of 256 columns, so width must fit in 0.75 of the frame
        h = max(np.abs(p[:, 0] / p[:, 2]).max() / 0.75, np.abs(p[:, 1] / p[:, 2]).max())
        fill = rng.uniform(0.75, 0.95)
        focal = (RES / 2) * fill / h
        M = np.eye(4); M[:3, :3] = R_bl; M[:3, 3] = cam_pos
        from mathutils import Matrix
        self.cam.matrix_world = Matrix(M.tolist())
        self.cam.data.lens = focal * 36.0 / RES
        self.bpy.context.view_layer.update()
        return cam_pos, R_bl, focal

    def ray_first(self, origin, points):
        """first object hit along each ray origin->point: returns (names, distances)."""
        from mathutils import Vector
        dg = self.bpy.context.evaluated_depsgraph_get()
        names, dists = [], []
        o = Vector(origin.tolist())
        for p in points:
            d = Vector(p.tolist()) - o
            L = d.length
            hit, loc, _, _, obj, _ = self.scene.ray_cast(dg, o, d.normalized(), distance=L + 0.01)
            names.append(obj.name if hit else "")
            dists.append((loc - o).length if hit else np.inf)
        return names, np.array(dists), np.linalg.norm(points - origin, axis=1)

    def add_occluders(self, verts, cam_pos, k):
        rng = self.rng
        floor_z = verts[:, 2].min()
        types = self.args.occ_types
        objs, used = [], []
        for i in range(k):
            kind = types[rng.integers(len(types))]
            parts = make_occluder(self.bpy, kind, rng, f"{self.sample_i}_{i}")
            mat = self.occ_mats[rng.integers(len(self.occ_mats))]
            set_material_color(mat, rand_color(rng, sat=(0.0, 0.8), val=(0.1, 0.95)))
            for o in parts:
                o.data.materials.append(mat)
            tgt = verts[rng.integers(len(verts))]
            to_cam = cam_pos - tgt
            dist = np.linalg.norm(to_cam)
            r = rng.uniform(0.3, min(0.9, dist - 0.6))
            pos = tgt + to_cam / dist * r
            pos[:2] += rng.normal(0, 0.12, 2)
            if kind in ("pole", "chair", "table") or rng.uniform() < 0.3:
                pos[2] = floor_z                          # standing on the floor
            else:
                pos[2] -= self._half_height(parts)        # centred on the target height
            place_parts(parts, pos, rng.uniform(0, 2 * math.pi))
            objs += parts; used.append(kind)
        self.bpy.context.view_layer.update()
        return objs, used

    @staticmethod
    def _half_height(parts):
        zs = [ (o.matrix_world @ v.co).z for o in parts for v in o.data.vertices ]
        return (max(zs) - min(zs)) / 2

    def render_to(self, path):
        if getattr(self.args, "labels_only", False):   # [selfocc] labels need ray casts only, not renders
            return
        self.scene.render.filepath = path
        self.bpy.ops.render.render(write_still=True)

    # ---- main loops ----
    def run_samples(self):
        a = self.args
        poses = np.load(a.poses)["poses"].astype(np.float64)       # (P, 72) axis-angle, AMASS (Z-up) root
        os.makedirs(a.out, exist_ok=True)
        import time
        t0 = time.time()
        L = {k: [] for k in ["global_orient", "body_pose", "betas", "joints3d", "joints2d", "focal",
                             "verts2d", "selfvis", "vis_occ3d", "occfrac_occ3d", "occ_ok", "n_occ", "pose_idx"]}
        if a.selfvis_joints:
            L.update({"selfvis_frac": [], "selfvis_joint": []})   # [selfocc]
        for i in range(a.n):
            self.sample_i = a.start + i
            if i % a.shape_every == 0:
                self.new_body()
            pi = int(self.rng.integers(len(poses)))
            self.pose_body(poses[pi])
            self.randomize_look()
            verts, joints = self.posed_geometry()
            cam_pos, R_bl, focal = self.place_camera(verts)

            j_cv = to_cv(joints, cam_pos, R_bl)
            vs = verts[self.vert_idx]
            v_cv = to_cv(vs, cam_pos, R_bl)

            # visibility without occluders (self-occlusion)
            names, hd, td = self.ray_first(cam_pos, vs)
            selfvis = np.array([n == self.body.name for n in names]) & (np.abs(hd - td) < 0.02)
            if a.selfvis_joints:   # [selfocc] per joint: is its surface visible with no occluders present?
                jv = [verts[ix] for ix in self.joint_verts]
                pts = np.concatenate([p for p in jv if len(p)])
                jn, jhd, jtd = self.ray_first(cam_pos, pts)
                ok = np.array([n == self.body.name for n in jn]) & (np.abs(jhd - jtd) < 0.02)
                fr, k0 = np.ones(24), 0
                for j, p in enumerate(jv):
                    if len(p):
                        fr[j] = ok[k0:k0 + len(p)].mean(); k0 += len(p)
                L["selfvis_frac"].append(fr.astype(np.float32))
                L["selfvis_joint"].append(fr >= SELFVIS_MIN)

            stem = f"{self.sample_i:06d}"
            self.render_to(os.path.join(a.out, f"clean_{stem}.png"))

            k = int(self.rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
            for attempt in range(a.max_tries):          # keep occlusion in a useful range
                occ, _ = self.add_occluders(verts, cam_pos, k)
                names, hd, td = self.ray_first(cam_pos, vs)
                hidden = np.array([n.startswith("occ_") for n in names]) & (hd < td - 0.01) & selfvis
                frac = hidden.sum() / max(selfvis.sum(), 1)
                if a.occ_min <= frac <= a.occ_max or attempt == a.max_tries - 1:
                    break
                remove_objects(self.bpy, occ)
            jn, jhd, jtd = self.ray_first(cam_pos, joints)
            vis_j = ~(np.array([n.startswith("occ_") for n in jn]) & (jhd < jtd - 0.01))
            self.render_to(os.path.join(a.out, f"occ3d_{stem}.png"))
            remove_objects(self.bpy, occ)

            # SMPL params with root orientation in the OpenCV camera frame
            R_cv_world = (R_bl * np.array([1.0, -1.0, -1.0])).T     # world -> cv
            R_world_arm = np.array(self.arm.matrix_world)[:3, :3]
            q0 = np.array(self.arm.pose.bones["pelvis"].rotation_quaternion)
            R_root = R_cv_world @ R_world_arm @ quat_to_rotmat(q0)
            body = np.stack([rotmat_to_rotvec(quat_to_rotmat(np.array(self.arm.pose.bones[n].rotation_quaternion)))
                             for n in self.names[1:]])

            L["global_orient"].append(rotmat_to_rotvec(R_root)); L["body_pose"].append(body)
            L["betas"].append(self.betas.copy()); L["joints3d"].append(j_cv)
            L["joints2d"].append(project(j_cv, focal)); L["focal"].append(focal)
            L["verts2d"].append(project(v_cv, focal)); L["selfvis"].append(selfvis)
            L["vis_occ3d"].append(vis_j); L["occfrac_occ3d"].append(hidden.sum() / max(selfvis.sum(), 1))
            L["occ_ok"].append(bool(a.occ_min <= frac <= a.occ_max))   # False = placement never hit the range
            L["n_occ"].append(k); L["pose_idx"].append(pi)

            if (i + 1) % 50 == 0 or i + 1 == a.n:
                np.savez(os.path.join(a.out, "labels.npz"), start=a.start,
                         **{k_: np.asarray(v_) for k_, v_ in L.items()})
                el = time.time() - t0
                print(f"[synth] {i + 1}/{a.n} | {el / (i + 1):.2f}s per sample | ~{el / (i + 1) * (a.n - i - 1) / 60:.0f} min left", flush=True)

    def run_cutouts(self):
        a = self.args
        os.makedirs(a.out, exist_ok=True)
        rng = self.rng
        for i in range(a.n):
            self.sample_i = a.start + i
            self.randomize_look()
            objs, _ = self.add_occluders(np.zeros((1, 3)) + np.array([0, 0, 0.0]), np.array([0, -5.0, 1.0]), 1)
            pts = np.array([list(o.matrix_world @ v.co) for o in objs for v in o.data.vertices])
            center = (pts.min(0) + pts.max(0)) / 2
            az, el, dist = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(-10, 40)), rng.uniform(2, 5)
            cam_pos = center + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            R_bl = look_at(cam_pos, center)
            p = to_cv(pts, cam_pos, R_bl)
            h = max(np.abs(p[:, 0] / p[:, 2]).max(), np.abs(p[:, 1] / p[:, 2]).max())
            from mathutils import Matrix
            M = np.eye(4); M[:3, :3] = R_bl; M[:3, 3] = cam_pos
            self.cam.matrix_world = Matrix(M.tolist())
            self.cam.data.lens = (RES / 2) * 0.9 / h * 36.0 / RES
            self.render_to(os.path.join(a.out, f"cutout_{a.start + i:06d}.png"))
            remove_objects(self.bpy, objs)
            if (i + 1) % 50 == 0:
                print(f"[synth] cutouts {i + 1}/{a.n}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="smpl_neutral.npz (SMPL, 24 joints)")
    ap.add_argument("--poses", help="poses .npz with 'poses' (P,72)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0, help="first sample index (for running shards in parallel)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel")
    ap.add_argument("--shape-every", type=int, default=50, help="new random body shape every N samples")
    ap.add_argument("--occ-types", nargs="+", default=OCC_TYPES)
    ap.add_argument("--mode", choices=["samples", "cutouts"], default="samples")
    ap.add_argument("--occ-min", type=float, default=0.05, help="retry occluder placement until this much of the body is hidden")
    ap.add_argument("--occ-max", type=float, default=0.75)
    ap.add_argument("--max-tries", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="[selfocc] skip rendering; labels are identical (placement uses ray casts, not images)")
    ap.add_argument("--selfvis-joints", action="store_true",
                    help="[selfocc] also label each joint as hidden-by-the-body or not (16 extra rays per joint)")
    a = ap.parse_args(argv)
    g = Generator(a)
    if a.mode == "cutouts":
        g.sample_i = 0
        g.run_cutouts()
    else:
        g.run_samples()


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)
