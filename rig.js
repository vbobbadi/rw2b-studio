/* Rigged 3D characters (Mixamo FBX or GLB) driven by the same named body joints as the 2D characters.
   Files are loaded from the viewer's own device and kept only in this browser (IndexedDB). */
(() => {
const T = window.THREE;
if (!T) { window.RigChars = null; return; }

const AIMS = [ // [bone, child bone, from joint, to joint]
  ["Spine", "Neck", "pelvis", "neck"], ["Neck", "Head", "neck", "head"],
  ["LeftShoulder", "LeftArm", "neck", "lsho"], ["RightShoulder", "RightArm", "neck", "rsho"],
  ["LeftArm", "LeftForeArm", "lsho", "lelb"], ["LeftForeArm", "LeftHand", "lelb", "lwri"],
  ["RightArm", "RightForeArm", "rsho", "relb"], ["RightForeArm", "RightHand", "relb", "rwri"],
  ["LeftUpLeg", "LeftLeg", "lhip", "lkne"], ["LeftLeg", "LeftFoot", "lkne", "lank"], ["LeftFoot", "LeftToeBase", "lank", "lfoot"],
  ["RightUpLeg", "RightLeg", "rhip", "rkne"], ["RightLeg", "RightFoot", "rkne", "rank"], ["RightFoot", "RightToeBase", "rank", "rfoot"]];
const clean = n => n.replace(/^.*:/, "").replace(/^mixamorig\d*[:_]?/i, "");

let renderer = null, glCanvas = null;
function gl(){ if (renderer) return renderer; glCanvas = document.createElement("canvas");
  renderer = new T.WebGLRenderer({canvas: glCanvas, alpha: true, antialias: true, preserveDrawingBuffer: true}); renderer.setClearColor(0x000000, 0);
  if ("outputEncoding" in renderer) renderer.outputEncoding = T.sRGBEncoding; return renderer; }

const rigs = {}; // id -> {name, scene, root, model, bones, rest, height}
function prepare(id, name, object){
  const scene = new T.Scene(); scene.add(new T.HemisphereLight(0xffffff, 0x334055, 1.15)); const d = new T.DirectionalLight(0xffffff, 0.9); d.position.set(1, 2, 3); scene.add(d);
  const root = new T.Group(); scene.add(root); const holder = new T.Group(); root.add(holder); holder.add(object);
  object.traverse(o => { if (o.isMesh || o.isSkinnedMesh){ o.frustumCulled = false; const ms = Array.isArray(o.material) ? o.material : [o.material]; ms.forEach(m => { if (m){ m.side = T.FrontSide; if (m.map) m.map.encoding = T.sRGBEncoding; } }); } });
  // one bone set per distinct skeleton: some Mixamo files bind each body part to its own copy of the rig
  const sets = [], seen = new Set();
  object.traverse(o => { if (o.isSkinnedMesh && o.skeleton && !seen.has(o.skeleton.bones[0] && o.skeleton.bones[0].uuid)){ seen.add(o.skeleton.bones[0] && o.skeleton.bones[0].uuid);
    const b = {}; o.skeleton.bones.forEach(x => { const k = clean(x.name); if (!b[k]) b[k] = x; }); if (b.Hips) sets.push(b); } });
  if (!sets.length){ const b = {}; object.traverse(o => { if (o.isBone){ const k = clean(o.name); if (!b[k]) b[k] = o; } }); if (b.Hips) sets.push(b); }
  if (!sets.length) throw new Error("No Mixamo-style skeleton found (expected a bone named Hips). Download from Mixamo with Skin and in T-pose.");
  object.updateMatrixWorld(true);
  const box = new T.Box3().setFromObject(object), h = Math.max(1e-6, box.max.y - box.min.y), s = 1.75 / h;
  holder.scale.setScalar(s); root.updateMatrixWorld(true);
  const hp = new T.Vector3(); sets[0].Hips.getWorldPosition(hp); holder.position.sub(hp); root.updateMatrixWorld(true);
  const all = sets.map(b => { const rest = {}; Object.entries(b).forEach(([k, x]) => rest[k] = x.quaternion.clone()); return {bones:b, rest}; });
  rigs[id] = {id, name, scene, root, holder, object, sets:all, bones:sets[0]};
  return rigs[id];
}
const V = (p) => new T.Vector3(p[0], p[1], p[2]);
const wp = (b) => { const v = new T.Vector3(); b.getWorldPosition(v); return v; };
const wq = (b) => { const q = new T.Quaternion(); b.getWorldQuaternion(q); return q; };
function setWorldQuat(bone, qWorld){ const pq = bone.parent ? wq(bone.parent) : new T.Quaternion(); bone.quaternion.copy(pq.invert().multiply(qWorld)); bone.updateMatrixWorld(true); }
function basis(up, right){ const u = up.clone().normalize(), f = new T.Vector3().crossVectors(right, u).normalize(), r = new T.Vector3().crossVectors(u, f).normalize(); return new T.Matrix4().makeBasis(r, u, f); }

/* P: named joints in metres, three.js axes (x right, y up, z toward viewer), any origin */
function pose(rig, P){ rig.sets.forEach(set => poseSet(rig.root, set.bones, set.rest, P)); rig.root.updateMatrixWorld(true); }
function poseSet(root, bones, rest, P){
  Object.entries(bones).forEach(([k, b]) => b.quaternion.copy(rest[k])); root.updateMatrixWorld(true);
  const H = bones.Hips, tgt = k => V(P[k]);
  const lU = bones.LeftUpLeg, rU = bones.RightUpLeg, N = bones.Neck || bones.Spine2 || bones.Spine;
  if (lU && rU && N){
    const restB = basis(wp(N).sub(wp(H)), wp(lU).sub(wp(rU))), tgtB = basis(tgt("neck").sub(tgt("pelvis")), tgt("lhip").sub(tgt("rhip")));
    const q = new T.Quaternion().setFromRotationMatrix(tgtB.multiply(restB.clone().transpose()));
    setWorldQuat(H, q.multiply(wq(H)));
  }
  AIMS.forEach(([bn, cn, a, b]) => { const bone = bones[bn], child = bones[cn]; if (!bone || !child || !P[a] || !P[b]) return;
    const cur = wp(child).sub(wp(bone)), want = tgt(b).sub(tgt(a)); if (cur.lengthSq() < 1e-10 || want.lengthSq() < 1e-10) return;
    const q = new T.Quaternion().setFromUnitVectors(cur.normalize(), want.normalize()); setWorldQuat(bone, q.multiply(wq(bone))); });
}

/* Draw onto a 2D context. P3: named 3D joints in camera convention (metres, y down, z forward).
   B2: the same joints on screen ([x, y, depth] px) — used to place and scale the figure. view: {yaw, pitch, mirror} */
function draw(ctx, id, P3, B2, view = {}){
  const rig = rigs[id]; if (!rig || !P3 || !B2) return false;
  const k2 = ctx.getTransform ? ctx.getTransform().a || 1 : 1, W = ctx.canvas.width / k2, H = ctx.canvas.height / k2;
  const P = {}; Object.entries(P3).forEach(([k, p]) => P[k] = [view.mirror ? -p[0] : p[0], -p[1], -p[2]]);
  const c = mid(P.lhip, P.rhip); Object.keys(P).forEach(k => P[k] = [P[k][0] - c[0], P[k][1] - c[1], P[k][2] - c[2]]);
  pose(rig, P);
  rig.root.rotation.set(view.pitch || 0, view.yaw || 0, 0, "XYZ"); rig.root.updateMatrixWorld(true);
  // scale: screen torso length / rendered torso length
  const proj = k => { const v = V(P[k]).applyEuler(rig.root.rotation); return v; };
  const t3 = proj("neck").sub(proj("pelvis")), t3len = Math.hypot(t3.x, t3.y);
  const t2len = Math.hypot(B2.neck[0] - B2.pelvis[0], B2.neck[1] - B2.pelvis[1]); if (t3len < 1e-4 || t2len < 2) return false;
  const s = t2len / t3len, hip2 = mid(B2.lhip, B2.rhip);
  const r = gl(), dpr = Math.min(2, window.devicePixelRatio || 1); r.setPixelRatio(dpr); r.setSize(W, H, false);
  const cam = new T.OrthographicCamera(0, W, H, 0, -1e5, 1e5); cam.position.z = 1e4;
  const g = rig.root; const savedS = g.scale.x; g.scale.setScalar(s); g.position.set(hip2[0], H - hip2[1], 0); g.updateMatrixWorld(true);
  r.render(rig.scene, cam); g.scale.setScalar(savedS); g.position.set(0, 0, 0);
  ctx.drawImage(glCanvas, 0, 0, W, H); return true;
}
const mid = (a, b) => [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2];

/* loading + storage */
function parse(name, buf){
  const ext = name.toLowerCase().split(".").pop();
  if (ext === "fbx"){ if (!T.FBXLoader) throw new Error("FBX loader missing."); return Promise.resolve(new T.FBXLoader().parse(buf, "")); }
  if (ext === "glb" || ext === "gltf"){ return new Promise((res, rej) => new T.GLTFLoader().parse(buf, "", g => res(g.scene), rej)); }
  return Promise.reject(new Error("Use an .fbx file from Mixamo, or a .glb file."));
}
const DB = "rw2b_rigs";
function idb(){ return new Promise((res, rej) => { try { const q = indexedDB.open(DB, 1); q.onupgradeneeded = () => q.result.createObjectStore("rigs", {keyPath: "id"}); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); } catch (e) { rej(e); } }); }
async function store(rec){ try { const db = await idb(); await new Promise((res, rej) => { const t = db.transaction("rigs", "readwrite"); t.objectStore("rigs").put(rec); t.oncomplete = res; t.onerror = () => rej(t.error); }); return true; } catch (e) { return false; } }
async function remove(id){ delete rigs[id]; try { const db = await idb(); db.transaction("rigs", "readwrite").objectStore("rigs").delete(id); } catch (e) {} }
async function add(file){
  const buf = await file.arrayBuffer(); const name = file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim() || "Rigged character";
  const id = "rig_" + Date.now().toString(36); prepare(id, name, await parse(file.name, buf));
  const kept = await store({id, name, file: file.name, buf}); return {id, name, kept};
}
async function restore(){ try { const db = await idb(); const all = await new Promise((res, rej) => { const q = db.transaction("rigs").objectStore("rigs").getAll(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); });
  for (const r of all){ try { prepare(r.id, r.name, await parse(r.file, r.buf)); } catch (e) {} } } catch (e) {} return list(); }
function list(){ return Object.values(rigs).map(r => ({id: r.id, name: r.name, bones: Object.keys(r.bones).length})); }

window.RigChars = {add, restore, list, draw, remove};
})();
