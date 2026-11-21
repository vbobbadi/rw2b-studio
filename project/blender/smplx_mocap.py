"""
Build an animated SMPL-X body in Blender from a motion .npz (notebook 01 output).

No add-on and no screen needed, so it runs in Colab:
    blender --background --python smplx_mocap.py -- \
        --model SMPLX_NEUTRAL.npz --motion video_smpl.npz \
        --blend out.blend --video out.mp4

What it makes:
  - SMPL-X body mesh + armature (55 bones), skinned with the model's own weights
  - your motion keyframed on the 22 body bones (pelvis..wrists), lightly smoothed
  - floor, light and a camera that follows the body
  - optional: renders an .mp4 and saves the .blend
"""

import argparse
import sys

import numpy as np

# ---------------- SMPL-X skeleton ----------------
SMPLX_JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw",
    "left_eye_smplhf", "right_eye_smplhf",
] + [f"left_{f}{i}" for f in ("index", "middle", "pinky", "ring", "thumb") for i in (1, 2, 3)] \
  + [f"right_{f}{i}" for f in ("index", "middle", "pinky", "ring", "thumb") for i in (1, 2, 3)]

SMPL_JOINTS = SMPLX_JOINTS[:22] + ["left_hand", "right_hand"]


def joint_names(n):
    if n == 55:
        return SMPLX_JOINTS
    if n == 24:
        return SMPL_JOINTS
    raise ValueError(f"expected 24 (SMPL) or 55 (SMPL-X) joints, got {n}")

# SMPL body_pose[0..20] uses the same joints, in the same order, as SMPL-X joints 1..21.
# SMPL joints 22/23 (hands) are skipped: SMPL-X has finger bones there.
BODY_BONES = SMPLX_JOINTS[:22]


# ---------------- math (plain numpy) ----------------
def rotvec_to_quat(rv):
    """axis-angle (..., 3) -> quaternion (..., 4), w first (Blender order)."""
    rv = np.asarray(rv, dtype=np.float64)
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    axis = np.divide(rv, angle, out=np.zeros_like(rv), where=angle > 1e-8)
    return np.concatenate([np.cos(angle / 2), axis * np.sin(angle / 2)], axis=-1)


def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_to_rotmat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_rotvec(R):
    """3x3 rotation matrix -> axis-angle (3,). Shepperd's method via quaternion, stable near 180 deg."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    if q[0] < 0:
        q = -q
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(3)
    return v / n * 2 * np.arctan2(n, q[0])


def _moving_avg(x, window):
    """x: (T, C). Centered moving average with edge padding."""
    pad = window // 2
    p = np.concatenate([x[:1].repeat(pad, 0), x, x[-1:].repeat(pad, 0)], axis=0)
    k = np.ones(window) / window
    return np.stack([np.convolve(p[:, c], k, mode="valid") for c in range(x.shape[1])], axis=1)


def smooth_quats(q, window):
    """q: (T, J, 4). Remove sign flips, then average over time and renormalize."""
    q = q.copy()
    for t in range(1, len(q)):
        q[t][np.sum(q[t] * q[t - 1], axis=-1) < 0] *= -1
    if window <= 1:
        return q
    T, J, _ = q.shape
    out = _moving_avg(q.reshape(T, J * 4), window).reshape(T, J, 4)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def prepare_motion(data, smooth=5, flip_camera=True, use_depth=False):
    """-> frame numbers (T,), quats (T, 22, 4) for BODY_BONES, root offset (T, 3) in SMPL coords (Y up)."""
    go = np.asarray(data["global_orient"], dtype=np.float64).reshape(-1, 3)
    bp = np.asarray(data["body_pose"], dtype=np.float64).reshape(len(go), -1, 3)[:, :21]
    q = rotvec_to_quat(np.concatenate([go[:, None], bp], axis=1))

    cam = np.asarray(data["camera"], dtype=np.float64).reshape(len(go), 3)
    if flip_camera:
        # camera coords (x right, y down, z forward) -> SMPL coords (y up, z toward camera)
        q[:, 0] = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), q[:, 0])
        cam = cam * np.array([1.0, -1.0, -1.0])
    q = smooth_quats(q, smooth)

    root = cam - cam[0]
    if not use_depth:
        root[:, 2] = 0.0
    if smooth > 1:
        root = _moving_avg(root, smooth)
    frames = np.asarray(data["frame_idx"]).reshape(-1) if "frame_idx" in data else np.arange(len(go))
    return frames.astype(int), q, root


def load_body_model(path, betas=None):
    """SMPL (24 joints) or SMPL-X (55 joints) .npz -> vertices, faces, joints, skin weights, parents.
    betas: optional shape coefficients; vertices and joints are reshaped before the skeleton is built."""
    m = np.load(path, allow_pickle=True)
    v = np.asarray(m["v_template"], dtype=np.float64)
    if betas is not None and "shapedirs" in m:
        sd = np.asarray(m["shapedirs"], dtype=np.float64)
        b = np.asarray(betas, dtype=np.float64)[: sd.shape[-1]]
        v = v + sd[..., : len(b)] @ b
    f = np.asarray(m["f"], dtype=np.int64)
    J_reg = m["J_regressor"]
    if hasattr(J_reg, "toarray"):
        J_reg = J_reg.toarray()
    J = np.asarray(J_reg, dtype=np.float64) @ v
    W = np.asarray(m["weights"], dtype=np.float64)
    parents = np.asarray(m["kintree_table"])[0].astype(np.int64)
    parents[0] = -1
    joint_names(J.shape[0])  # validates count
    return v, f, J, W, parents


def build_body(bpy, v, f, J, W, parents, name="SMPLX", color=(0.65, 0.74, 0.86, 1.0)):
    """Armature (bone axes = SMPL axes, so SMPL rotations apply directly) + skinned mesh.
    The armature object is rotated +90 deg about X so the body stands up in Blender (Z up, facing -Y)."""
    from mathutils import Matrix
    names = joint_names(J.shape[0])
    scene = bpy.context.scene
    arm_data = bpy.data.armatures.new(f"{name}_skeleton")
    arm = bpy.data.objects.new(name, arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for i, n in enumerate(names):
        eb = arm_data.edit_bones.new(n)
        eb.head = J[i].tolist()
        eb.tail = (J[i] + np.array([0.0, 0.05, 0.0])).tolist()
        eb.roll = 0.0
        if parents[i] >= 0:
            eb.parent = arm_data.edit_bones[names[parents[i]]]
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.matrix_world = Matrix.Rotation(np.pi / 2, 4, "X")

    me = bpy.data.meshes.new(f"{name}_mesh")
    me.from_pydata(v.tolist(), [], f.tolist())
    me.update()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    body = bpy.data.objects.new(f"{name}_body", me)
    scene.collection.objects.link(body)
    body.parent = arm
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    for j, n in enumerate(names):
        vg = body.vertex_groups.new(name=n)
        for vi in np.nonzero(W[:, j] > 1e-4)[0]:
            vg.add([int(vi)], float(W[vi, j]), "REPLACE")
    mat = bpy.data.materials.new(f"{name}_skin")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
    me.materials.append(mat)
    for n in names:
        arm.pose.bones[n].rotation_mode = "QUATERNION"
    return arm, body


# ---------------- Blender ----------------
def build_scene(model_path, motion_path, smooth=5, flip_camera=True, use_depth=False, start=1):
    import bpy
    from mathutils import Matrix

    v, f, J, W, parents = load_body_model(model_path)
    data = dict(np.load(motion_path, allow_pickle=True))
    fps = int(round(float(data.get("fps", 30))))
    frames, quats, root = prepare_motion(data, smooth, flip_camera, use_depth)

    # empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    arm, body = build_body(bpy, v, f, J, W, parents, name="SMPLX")

    # keyframes
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions.new("mocap")
    for n in BODY_BONES:
        arm.pose.bones[n].rotation_mode = "QUATERNION"
    for t, fr in enumerate(frames):
        frame = start + int(fr)
        for j, n in enumerate(BODY_BONES):
            pb = arm.pose.bones[n]
            pb.rotation_quaternion = quats[t, j].tolist()
            pb.keyframe_insert("rotation_quaternion", frame=frame)
        pel = arm.pose.bones["pelvis"]
        pel.location = root[t].tolist()          # bone axes = SMPL axes, so this is SMPL-space offset
        pel.keyframe_insert("location", frame=frame)

    scene.render.fps = fps
    scene.frame_start = start
    scene.frame_end = start + int(frames.max())

    # floor at foot height, light, follow camera
    foot_z = float(v[:, 1].min())              # SMPL Y (up) -> Blender Z
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, foot_z))
    floor_mat = bpy.data.materials.new("floor")
    floor_mat.use_nodes = True
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    bpy.context.active_object.data.materials.append(floor_mat)

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (np.radians(50), 0, np.radians(30))
    scene.collection.objects.link(sun)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world

    target = bpy.data.objects.new("cam_target", None)
    scene.collection.objects.link(target)
    c = target.constraints.new("COPY_LOCATION")
    c.target, c.subtarget = arm, "pelvis"
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(cam)
    cam.location = (0, -6.0, 0.6)
    cam.parent = target                        # moves with the body
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    scene.camera = cam

    print(f"[smplx_mocap] built {len(frames)} frames at {fps} fps")
    return scene


def enable_gpu(scene):
    """Use OptiX or CUDA if available, else CPU. Returns the backend name or 'CPU'."""
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type == backend for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                return backend
        except TypeError:
            continue
    return "CPU"


def render(scene, video_path, res=(640, 480), samples=16, gpu=True, frame_end=None):
    import bpy
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    if gpu:
        backend = enable_gpu(scene)
        print(f"[smplx_mocap] rendering on {'GPU (' + backend + ')' if backend != 'CPU' else 'CPU (no GPU found)'}")
    if frame_end is not None:
        scene.frame_end = min(scene.frame_end, frame_end)
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.filepath = video_path
    bpy.ops.render.render(animation=True)
    print(f"[smplx_mocap] video -> {video_path}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SMPLX_NEUTRAL.npz")
    ap.add_argument("--motion", required=True, help="video_smpl.npz from notebook 01")
    ap.add_argument("--blend", default=None, help="save .blend here")
    ap.add_argument("--video", default=None, help="render .mp4 here")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--no-flip", action="store_true", help="use if the body comes out upside down")
    ap.add_argument("--use-depth", action="store_true")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--res", type=int, nargs=2, default=(640, 480))
    ap.add_argument("--max-frames", type=int, default=None, help="render only the first N frames (quick test)")
    args = ap.parse_args(argv)

    import bpy
    scene = build_scene(args.model, args.motion, args.smooth, not args.no_flip, args.use_depth)
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.blend)
        print(f"[smplx_mocap] blend -> {args.blend}")
    if args.video:
        end = scene.frame_start + args.max_frames - 1 if args.max_frames else None
        render(scene, args.video, tuple(args.res), args.samples, frame_end=end)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)
