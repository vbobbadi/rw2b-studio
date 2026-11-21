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
const ALIAS = {Abdomen:"Spine"}; // Blender/Quaternius-style names → Mixamo names
[["L", "Left"], ["R", "Right"]].forEach(([s, side]) => Object.entries({Shoulder:"Shoulder", UpperArm:"Arm", LowerArm:"ForeArm", Palm2:"Hand", UpperLeg:"UpLeg", LowerLeg:"Leg", Foot:"Foot"}).forEach(([a, b]) => { ALIAS[a + "." + s] = side + b; ALIAS[a + s] = side + b; ALIAS[a + "_" + s] = side + b; }));
const clean = n => { const k = n.replace(/^.*:/, "").replace(/^mixamorig\d*[:_]?/i, ""); return ALIAS[k] || k; };

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
  // height from the skeleton (mesh bounds are unreliable when the armature itself is scaled, e.g. Mixamo GLBs)
  const B0 = sets[0], yOf = b => { const v = new T.Vector3(); b.getWorldPosition(v); return v.y; };
  const top = B0.Head ? yOf(B0.Head) : null, feet = [B0.LeftFoot, B0.RightFoot, B0.LeftToeBase, B0.RightToeBase].filter(Boolean).map(yOf);
  let h = top != null && feet.length ? (top - Math.min(...feet)) * 1.12 : 0;
  if (!(h > 1e-6)){ const box = new T.Box3().setFromObject(object); h = Math.max(1e-6, box.max.y - box.min.y); }
  const s = 1.75 / h;
  holder.scale.setScalar(s); root.updateMatrixWorld(true);
  const hp = new T.Vector3(); sets[0].Hips.getWorldPosition(hp); holder.position.sub(hp); root.updateMatrixWorld(true);
  const all = sets.map(b => { const rest = {}, aim = {}; Object.entries(b).forEach(([k, x]) => rest[k] = x.quaternion.clone());
    AIMS.forEach(([bn, cn]) => { const bone = b[bn], child = b[cn]; if (!bone || !child) return; const v = wp(child).sub(wp(bone));
      let desc = false; for (let p = child.parent; p; p = p.parent) if (p === bone){ desc = true; break; }
      aim[bn] = {v: v.clone().applyQuaternion(wq(bone).invert()), len: v.length(), desc}; });
    return {bones:b, rest, aim}; });
  rigs[id] = {id, name, scene, root, holder, object, sets:all, bones:sets[0]};
  return rigs[id];
}
const V = (p) => new T.Vector3(p[0], p[1], p[2]);
const wp = (b) => { const v = new T.Vector3(); b.getWorldPosition(v); return v; };
const wq = (b) => { const q = new T.Quaternion(); b.getWorldQuaternion(q); return q; };
function setWorldQuat(bone, qWorld){ const pq = bone.parent ? wq(bone.parent) : new T.Quaternion(); bone.quaternion.copy(pq.invert().multiply(qWorld)); bone.updateMatrixWorld(true); }
function basis(up, right){ const u = up.clone().normalize(), f = new T.Vector3().crossVectors(right, u).normalize(), r = new T.Vector3().crossVectors(u, f).normalize(); return new T.Matrix4().makeBasis(r, u, f); }

/* P: named joints in metres, three.js axes (x right, y up, z toward viewer), any origin */
function pose(rig, P){ rig.sets.forEach(set => poseSet(rig.root, set.bones, set.rest, P, set.aim)); rig.root.updateMatrixWorld(true); }
function poseSet(root, bones, rest, P, aim = {}){
  Object.entries(bones).forEach(([k, b]) => b.quaternion.copy(rest[k])); root.updateMatrixWorld(true);
  const H = bones.Hips, tgt = k => V(P[k]);
  const lU = bones.LeftUpLeg, rU = bones.RightUpLeg, N = bones.Neck || bones.Spine2 || bones.Spine;
  if (lU && rU && N){
    const restB = basis(wp(N).sub(wp(H)), wp(lU).sub(wp(rU))), tgtB = basis(tgt("neck").sub(tgt("pelvis")), tgt("lhip").sub(tgt("rhip")));
    const q = new T.Quaternion().setFromRotationMatrix(tgtB.multiply(restB.clone().transpose()));
    setWorldQuat(H, q.multiply(wq(H)));
  }
  AIMS.forEach(([bn, cn, a, b]) => { const bone = bones[bn], child = bones[cn]; if (!bone || !child || !P[a] || !P[b]) return;
    const A = aim[bn], cur = A ? A.v.clone().applyQuaternion(wq(bone)) : wp(child).sub(wp(bone)), want = tgt(b).sub(tgt(a)); if (cur.lengthSq() < 1e-10 || want.lengthSq() < 1e-10) return;
    const q = new T.Quaternion().setFromUnitVectors(cur.clone().normalize(), want.clone().normalize()); setWorldQuat(bone, q.multiply(wq(bone)));
    if (A && !A.desc && child.parent){ // IK-style rigs: the child is not under this bone, so carry it to the bone's end
      const end = wp(bone).add(A.v.clone().applyQuaternion(wq(bone))); child.position.copy(child.parent.worldToLocal(end)); child.updateMatrixWorld(true); } });
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
  // scale = median screen/3D length over limbs that lie roughly across the view (foreshortened ones are unreliable)
  const SEG = [["neck","pelvis"],["lsho","rsho"],["lhip","rhip"],["lsho","lelb"],["lelb","lwri"],["rsho","relb"],["relb","rwri"],["lhip","lkne"],["lkne","lank"],["rhip","rkne"],["rkne","rank"]];
  const rat = [];
  SEG.forEach(([a, b]) => { if (!P[a] || !P[b] || !B2[a] || !B2[b]) return; const d = proj(b).sub(proj(a)), xy = Math.hypot(d.x, d.y), L = d.length();
    const l2 = Math.hypot(B2[b][0] - B2[a][0], B2[b][1] - B2[a][1]); if (L > 1e-4 && xy / L > 0.6 && l2 > 2) rat.push(l2 / xy); });
  if (!rat.length) return false; rat.sort((x, y) => x - y);
  const s = rat[Math.floor(rat.length / 2)], hip2 = mid(B2.lhip, B2.rhip);
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
function list(){ return Object.values(rigs).map(r => ({id: r.id, name: r.name, bones: Object.keys(r.bones).length, builtin: !!r.builtin, blurb: r.blurb, credit: r.credit})); }

async function builtin(items, onEach){
  for (const it of items){ if (rigs[it.id]) continue;
    try { const buf = await (await fetch(it.url)).arrayBuffer(); const r = prepare(it.id, it.name, await parse(it.url, buf)); Object.assign(r, {builtin:true, blurb:it.blurb, credit:it.credit}); onEach && onEach(r); }
    catch (e) { console.warn("rw2b: could not load", it.name, e); } } }
window.RigChars = {add, restore, list, draw, remove, builtin};
})();
