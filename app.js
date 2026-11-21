(() => {
const $ = s => document.querySelector(s);
const ACC = "#ff8a3d";

/* =====================================================================
   Common body format. Every source (SMPL-22 from our models, MediaPipe-33
   in the browser) is mapped to these named joints, each [x, y, depth]
   in screen pixels, so every character draws from the same skeleton.
   ===================================================================== */
const J = ["head","neck","chest","pelvis","lsho","rsho","lelb","relb","lwri","rwri","lhip","rhip","lkne","rkne","lank","rank","lfoot","rfoot"];
const mid = (a, b) => [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2];
const lerp = (a, b, t) => [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];
function fromSMPL(P){ // P: 22 points [x,y,z]
  return {head:P[15], neck:P[12], chest:P[6], pelvis:P[0], lsho:P[16], rsho:P[17], lelb:P[18], relb:P[19], lwri:P[20], rwri:P[21],
          lhip:P[1], rhip:P[2], lkne:P[4], rkne:P[5], lank:P[7], rank:P[8], lfoot:P[10], rfoot:P[11]};
}
function fromMP(L){ // L: 33 points [x,y,z]
  const neck = mid(L[11], L[12]), pelvis = mid(L[23], L[24]);
  const ears = mid(L[7], L[8]); const head = lerp(ears, L[0], 0.35);
  return {head, neck, chest:lerp(pelvis, neck, 0.62), pelvis, lsho:L[11], rsho:L[12], lelb:L[13], relb:L[14], lwri:L[15], rwri:L[16],
          lhip:L[23], rhip:L[24], lkne:L[25], rkne:L[26], lank:L[27], rank:L[28], lfoot:L[31], rfoot:L[32]};
}
const LIMBS = [["neck","head"],["chest","neck"],["pelvis","chest"],["neck","lsho"],["neck","rsho"],["lsho","lelb"],["lelb","lwri"],["rsho","relb"],["relb","rwri"],
               ["pelvis","lhip"],["pelvis","rhip"],["lhip","lkne"],["lkne","lank"],["lank","lfoot"],["rhip","rkne"],["rkne","rank"],["rank","rfoot"]];
const dist = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1]);

/* ============================ characters ============================ */
function capsule(ctx, a, b, w, fill){ ctx.lineCap = "round"; ctx.lineWidth = w; ctx.strokeStyle = fill; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
function sortedLimbs(B){ return LIMBS.map(l => ({l, z:(B[l[0]][2] + B[l[1]][2]) / 2})).sort((p, q) => q.z - p.z).map(o => o.l); } // far first (larger depth = farther)
function unit(B){ return Math.max(6, (dist(B.lsho, B.rsho) + dist(B.lhip, B.rhip)) / 2, dist(B.neck, B.pelvis) * 0.5); }
function headSize(B){ return Math.max(6, Math.max(dist(B.neck, B.head) * 1.05, unit(B) * 0.42)); }
function drawHeadImage(ctx, B, img){
  const r = headSize(B) * 1.35, ang = Math.atan2(B.head[1] - B.neck[1], B.head[0] - B.neck[0]) + Math.PI / 2;
  ctx.save(); ctx.translate(B.head[0], B.head[1]); ctx.rotate(ang); ctx.drawImage(img, -r, -r * 1.1, r * 2, r * 2); ctx.restore();
}
function shade(hex, f){ const n = parseInt(hex.slice(1), 16); let r = n >> 16, g = (n >> 8) & 255, b = n & 255; const m = c => Math.max(0, Math.min(255, Math.round(f > 0 ? c + (255 - c) * f : c * (1 + f)))); return `rgb(${m(r)},${m(g)},${m(b)})`; }

const CHARACTERS = {
  mocap: {name:"Mocap Suit", blurb:"Black suit with reflective markers, like a studio capture session.", color:"#e9edf3",
    draw(ctx, B, o){ const u = unit(B), c = o.color;
      sortedLimbs(B).forEach(([a, b]) => capsule(ctx, B[a], B[b], u * (a === "pelvis" || b === "chest" || a === "chest" ? 0.55 : 0.26), "#2b3038"));
      ctx.fillStyle = "#2b3038"; ctx.beginPath(); ctx.arc(B.head[0], B.head[1], headSize(B), 0, 7); ctx.fill();
      ctx.shadowColor = c; ctx.shadowBlur = u * 0.35; ctx.fillStyle = c;
      J.forEach(k => { const p = B[k]; ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(2, u * 0.07), 0, 7); ctx.fill(); }); ctx.shadowBlur = 0; } },
  neon: {name:"Neon Wire", blurb:"Glowing tubes that trace every limb. Best on dark footage.", color:"#ff8a3d",
    draw(ctx, B, o){ const u = unit(B); ctx.shadowColor = o.color; ctx.shadowBlur = u * 0.6;
      sortedLimbs(B).forEach(([a, b]) => capsule(ctx, B[a], B[b], u * 0.12, o.color));
      ctx.lineWidth = u * 0.1; ctx.strokeStyle = o.color; ctx.beginPath(); ctx.arc(B.head[0], B.head[1], headSize(B) * 0.85, 0, 7); ctx.stroke();
      ctx.shadowBlur = 0; sortedLimbs(B).forEach(([a, b]) => capsule(ctx, B[a], B[b], u * 0.04, "#fff7")); } },
  mannequin: {name:"Mannequin", blurb:"Wooden artist's figure with rounded, shaded limbs.", color:"#d9a46c",
    draw(ctx, B, o){ const u = unit(B), c = o.color;
      sortedLimbs(B).forEach(([a, b]) => { const torso = ["pelvis","chest"].includes(a) || b === "chest"; capsule(ctx, B[a], B[b], u * (torso ? 0.62 : 0.3), shade(c, -0.35)); capsule(ctx, B[a], B[b], u * (torso ? 0.5 : 0.22), c); capsule(ctx, B[a], B[b], u * 0.06, shade(c, 0.35)); });
      J.filter(k => /elb|kne|sho|hip/.test(k)).forEach(k => { ctx.fillStyle = shade(c, -0.2); ctx.beginPath(); ctx.arc(B[k][0], B[k][1], u * 0.15, 0, 7); ctx.fill(); });
      const h = headSize(B), g = ctx.createRadialGradient(B.head[0] - h * .3, B.head[1] - h * .3, h * .1, B.head[0], B.head[1], h);
      g.addColorStop(0, shade(c, 0.35)); g.addColorStop(1, shade(c, -0.25)); ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(B.head[0], B.head[1], h * 0.82, h, 0, 0, 7); ctx.fill(); } },
  robot: {name:"Robot", blurb:"Boxy metal limbs, a chest plate and a glowing visor.", color:"#8fb3d9",
    draw(ctx, B, o){ const u = unit(B), c = o.color;
      sortedLimbs(B).forEach(([a, b]) => { const p = B[a], q = B[b], w = u * (["pelvis","chest"].includes(a) || b === "chest" ? 0.5 : 0.24), L = dist(p, q), ang = Math.atan2(q[1]-p[1], q[0]-p[0]);
        ctx.save(); ctx.translate(p[0], p[1]); ctx.rotate(ang); ctx.fillStyle = shade(c, -0.3); ctx.fillRect(0, -w/2, L, w); ctx.fillStyle = c; ctx.fillRect(w * 0.12, -w * 0.32, L - w * 0.24, w * 0.64); ctx.restore(); });
      J.filter(k => /elb|kne|wri|ank/.test(k)).forEach(k => { ctx.fillStyle = "#2b3442"; ctx.beginPath(); ctx.arc(B[k][0], B[k][1], u * 0.13, 0, 7); ctx.fill(); ctx.fillStyle = ACC; ctx.beginPath(); ctx.arc(B[k][0], B[k][1], u * 0.05, 0, 7); ctx.fill(); });
      const h = headSize(B), ang = Math.atan2(B.head[1] - B.neck[1], B.head[0] - B.neck[0]) + Math.PI / 2;
      ctx.save(); ctx.translate(B.head[0], B.head[1]); ctx.rotate(ang); ctx.fillStyle = shade(c, -0.15); ctx.fillRect(-h * .85, -h, h * 1.7, h * 1.8);
      ctx.fillStyle = "#0d1117"; ctx.fillRect(-h * .65, -h * .45, h * 1.3, h * .5); ctx.shadowColor = "#5fd3ff"; ctx.shadowBlur = h * .6; ctx.fillStyle = "#5fd3ff"; ctx.fillRect(-h * .5, -h * .3, h, h * .18);
      ctx.shadowBlur = 0; ctx.fillStyle = shade(c, -0.4); ctx.fillRect(-h * .06, -h * 1.45, h * .12, h * .45); ctx.restore(); } },
  bones: {name:"Bones", blurb:"An anatomy-class skeleton with knobbed bones and a skull.", color:"#f1ead8",
    draw(ctx, B, o){ const u = unit(B), c = o.color;
      sortedLimbs(B).forEach(([a, b]) => { if (a === "pelvis" && b === "chest"){ for (let t = 0; t <= 1.001; t += 0.2){ const p = lerp(B.pelvis, B.neck, t); ctx.fillStyle = c; ctx.beginPath(); ctx.ellipse(p[0], p[1], u * 0.11, u * 0.07, 0, 0, 7); ctx.fill(); } return; }
        capsule(ctx, B[a], B[b], u * 0.09, c); [B[a], B[b]].forEach(p => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(p[0], p[1], u * 0.09, 0, 7); ctx.fill(); }); });
      ctx.strokeStyle = c; ctx.lineWidth = u * 0.06; for (let i = 0; i < 4; i++){ const p = lerp(B.chest, B.neck, i / 4), w = u * (0.42 - i * 0.04); ctx.beginPath(); ctx.ellipse(p[0], p[1], w, u * 0.07, 0, 0, Math.PI); ctx.stroke(); }
      const h = headSize(B); ctx.fillStyle = c; ctx.beginPath(); ctx.ellipse(B.head[0], B.head[1], h * .8, h * .95, 0, 0, 7); ctx.fill();
      ctx.fillStyle = "#14161b"; [-1, 1].forEach(s => { ctx.beginPath(); ctx.ellipse(B.head[0] + s * h * .3, B.head[1] - h * .05, h * .2, h * .24, 0, 0, 7); ctx.fill(); });
      ctx.beginPath(); ctx.moveTo(B.head[0], B.head[1] + h * .15); ctx.lineTo(B.head[0] - h * .1, B.head[1] + h * .35); ctx.lineTo(B.head[0] + h * .1, B.head[1] + h * .35); ctx.fill(); } },
  shadow: {name:"Shadow", blurb:"A solid silhouette with a coloured rim light, like a stage shadow.", color:"#7d5cff",
    draw(ctx, B, o){ const u = unit(B); ctx.shadowColor = o.color; ctx.shadowBlur = u * 0.9;
      sortedLimbs(B).forEach(([a, b]) => capsule(ctx, B[a], B[b], u * (["pelvis","chest"].includes(a) || b === "chest" ? 0.75 : 0.36), "#07080b"));
      ctx.fillStyle = "#07080b"; ctx.beginPath(); ctx.arc(B.head[0], B.head[1], headSize(B), 0, 7); ctx.fill(); ctx.shadowBlur = 0; } },
  ink: {name:"Ink Brush", blurb:"Calligraphy strokes that taper toward hands and feet.", color:"#e8433a",
    draw(ctx, B, o){ const u = unit(B), c = o.color;
      sortedLimbs(B).forEach(([a, b]) => { const p = B[a], q = B[b], n = 10; for (let i = 0; i < n; i++){ const t0 = i / n, t1 = (i + 1) / n, w = u * 0.32 * (1 - t0 * 0.75); capsule(ctx, lerp(p, q, t0), lerp(p, q, t1), w, i % 3 === 0 ? shade(c, -0.25) : c); } });
      const h = headSize(B); ctx.lineWidth = u * 0.18; ctx.strokeStyle = c; ctx.beginPath(); ctx.arc(B.head[0], B.head[1], h * .8, 0.4, 5.9); ctx.stroke(); } }
};
let CUSTOM = null; // {name, color, base, img}
try { const s = JSON.parse(localStorage.getItem("rw2b_custom") || "null"); if (s && s.img){ const im = new Image(); im.src = s.img; CUSTOM = {...s, imgEl:im}; } } catch (e) {}
const RIG = window.RigChars;
function allChars(){ const o = {...CHARACTERS}; if (RIG) RIG.list().forEach(r => o[r.id] = {name:r.name, blurb:r.blurb || ("Your rigged 3D character (" + r.bones + " bones), posed from the same body joints."), color:"#ffffff", rig:r.id, builtin:r.builtin, credit:r.credit}); if (CUSTOM) o.custom = {name:CUSTOM.name || "My character", blurb:"Your upload on a " + CHARACTERS[CUSTOM.base].name + " body.", color:CUSTOM.color, draw:(ctx, B, opt) => { CHARACTERS[CUSTOM.base].draw(ctx, B, opt); if (CUSTOM.imgEl && CUSTOM.imgEl.complete) drawHeadImage(ctx, B, CUSTOM.imgEl); }}; return o; }
function drawCharacter(ctx, key, B, opts = {}){ const ch = allChars()[key] || CHARACTERS.mocap;
  if (ch.rig){ if (opts.P3 && RIG.draw(ctx, ch.rig, opts.P3, B, opts.view || {})) return; ctx.save(); CHARACTERS.mannequin.draw(ctx, B, {color:"#9aa1ad"}); ctx.restore(); return; }
  ctx.save(); ch.draw(ctx, B, {color: opts.color || ch.color}); if (opts.headImg && opts.headImg.complete) drawHeadImage(ctx, B, opts.headImg); ctx.restore(); }
function drawSkeleton(ctx, B, col = "#5fd38a"){ const u = unit(B); ctx.save(); ctx.globalAlpha = .9; LIMBS.forEach(([a, b]) => capsule(ctx, B[a], B[b], Math.max(1.5, u * 0.04), col)); ctx.fillStyle = "#fff"; J.forEach(k => { ctx.beginPath(); ctx.arc(B[k][0], B[k][1], Math.max(2, u * 0.035), 0, 7); ctx.fill(); }); ctx.restore(); }

/* ========================= 3D stage (orbit) ========================= */
function Stage(canvas, {yaw = -0.4, pitch = 0.15, floor = true} = {}){
  const ctx = canvas.getContext("2d"); const st = {yaw, pitch, canvas, ctx};
  let drag = null;
  canvas.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, st.yaw, st.pitch]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", e => { if (!drag) return; st.yaw = drag[2] + (e.clientX - drag[0]) * .01; st.pitch = Math.max(-1.2, Math.min(1.2, drag[3] + (e.clientY - drag[1]) * .01)); st.onchange && st.onchange(); });
  canvas.addEventListener("pointerup", () => drag = null);
  st.fit = () => { const r = canvas.getBoundingClientRect(), d = devicePixelRatio || 1; if (!r.width) return false; canvas.width = r.width * d; canvas.height = r.height * d; ctx.setTransform(d, 0, 0, d, 0, 0); return true; };
  // points in metres, y down (camera convention); returns screen [x, y, depth]
  st.project = (p, sc, cy = 0.5) => { const W = canvas.clientWidth, H = canvas.clientHeight; let X = p[0], Y = -p[1], Z = -p[2];
    const a = Math.cos(st.yaw), b = Math.sin(st.yaw); [X, Z] = [X * a + Z * b, -X * b + Z * a];
    const c = Math.cos(st.pitch), s = Math.sin(st.pitch); [Y, Z] = [Y * c - Z * s, Y * s + Z * c];
    return [W / 2 + X * sc, H * cy - Y * sc, -Z]; };
  st.clear = () => ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  st.grid = (sc, y = 0.95, n = 5, step = 0.25) => { ctx.strokeStyle = "#2a2f39"; ctx.lineWidth = 1; for (let i = -n; i <= n; i++){ const a = st.project([i*step, y, -n*step], sc), b = st.project([i*step, y, n*step], sc), c = st.project([-n*step, y, i*step], sc), d = st.project([n*step, y, i*step], sc); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke(); } };
  if (window.ResizeObserver) new ResizeObserver(() => { st.fit(); st.onchange && st.onchange(); }).observe(canvas);
  return st;
}

/* ============================== router ============================== */
const PAGES = ["studio","live","characters","arch","how","team","results"];
function route(){
  const h = (location.hash || "#studio").slice(1);
  const page = PAGES.includes(h) ? h : (document.getElementById(h) && document.getElementById(h).closest(".page") ? document.getElementById(h).closest(".page").id.replace("page-", "") : "studio");
  PAGES.forEach(p => document.getElementById("page-" + p).classList.toggle("on", p === page));
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("on", page === "results" ? a.getAttribute("href") === "#" + h : a.dataset.page === page));
  $("#resultsLegend").style.display = page === "results" ? "" : "none";
  if (page === "results" && window.RQ) RQ.show(h); else document.title = "Real World to Blender Studio";
  scrollTo(0, 0);
  dispatchEvent(new Event("resize"));
  if (page === "characters") Library.start(); else Library.stop();
}
addEventListener("hashchange", route);

/* ============================ page markup ============================ */
$("#page-studio").innerHTML = `
<section>
  <div class="sec-head"><div class="eyebrow">Studio · runs in your browser</div>
    <h1>Put a character on any video</h1>
    <p class="lede">Upload a clip or a photo, or try a sample. Pose is estimated on your own device for every person in the frame. Pick who gets which character, or leave someone unchanged, then record the result or export the motion for Blender.</p></div>
  <div class="studio">
    <div style="display:grid;gap:12px">
      <div class="stage" id="stage"><div class="empty" id="stageEmpty">Drop a video or photo here, or pick a sample on the right.</div><canvas id="out" hidden></canvas>
        <div class="status" id="status"></div></div>
      <div class="row-ctrl">
        <button class="btn" id="playBtn" disabled>Play</button>
        <button class="btn" id="recBtn" disabled>Record video</button>
        <button class="btn" id="jsonBtn" disabled>Export motion (JSON)</button>
        <button class="btn" id="pngBtn" disabled>Save frame</button>
        <span class="muted" id="saveMsg" style="font-size:12px"></span>
      </div>
      <div class="panel" style="padding:12px 14px;display:grid;gap:8px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap"><h3>3D view of the detected pose</h3><span class="muted" style="font-size:12px">drag to turn</span></div>
        <canvas id="pose3d" style="width:100%;height:260px;display:block;cursor:grab;touch-action:none;border-radius:10px;background:radial-gradient(100% 80% at 50% 15%,#232833,#11141a)"></canvas>
      </div>
    </div>
    <div class="ctrl panel">
      <div class="drop" id="drop"><b>Upload</b><span class="muted" style="font-size:12.5px">Video (mp4, webm, mov) or photo (jpg, png), with up to 5 people. Files stay on your device.</span>
        <label class="btn" for="file" style="display:inline-block">Choose a file</label><input type="file" id="file" accept="video/*,image/*" hidden></div>
      <div><div class="muted" style="font-size:12.5px;margin-bottom:6px">Samples from our test plan</div>
        <div class="row-ctrl" id="samples"></div></div>
      <div><div class="muted" style="font-size:12.5px;margin-bottom:6px">People in the video · click a person here or on the video</div><div class="people" id="people"><span class="muted" style="font-size:12.5px">Load a video to see who is in it.</span></div>
        <div class="row-ctrl" style="margin-top:8px"><button class="btn" id="allSame">Same character for everyone</button><button class="btn" id="mixChars">A different character each</button></div></div>
      <div><div class="muted" style="font-size:12.5px;margin-bottom:6px" id="pickLabel">Character for Person 1</div><div class="charpick" id="pick"></div></div>
      <div class="row-ctrl">
        <label>Colour <input type="color" id="color"></label>
        <label style="flex:1">Mode <select id="mode"><option value="overlay">Character over the person</option><option value="replace">Replace the person</option><option value="solo">Character only (black stage)</option><option value="side">Video and character side by side</option><option value="three">Three screens: camera · pose · both</option></select></label>
      </div>
      <div class="row-ctrl">
        <label style="flex:1">Model <select id="complex"><option value="lite">Lite (fastest)</option><option value="full" selected>Full (more accurate)</option></select></label>
        <label style="flex:1">Up to <select id="maxp"><option value="1">1 person</option><option value="2">2 people</option><option value="3">3 people</option><option value="5" selected>5 people</option></select></label>
        <label style="flex:1">Smoothing <input type="range" id="smooth" min="0" max="0.9" step="0.05" value="0.5"></label>
      </div>
      <div class="row-ctrl"><label style="flex-direction:row;display:flex;gap:6px;align-items:center"><input type="checkbox" id="labels" checked> Person labels</label><label style="flex-direction:row;display:flex;gap:6px;align-items:center"><input type="checkbox" id="skel"> Show detected skeleton</label>
        <label style="flex-direction:row;display:flex;gap:6px;align-items:center"><input type="checkbox" id="mirror"> Mirror</label></div>
      <p class="note" style="margin:0">Uses MediaPipe Pose Landmarker in the browser (up to 5 people). Our HMR2.0 / DGS models need a GPU and run in the Colab notebooks; their output is on the Results pages.</p>
    </div>
  </div>
</section>
<section>
  <div class="sec-head"><div class="eyebrow">Send to Blender</div><h2>Bring the exported motion into Blender</h2>
  <p class="lede">The JSON export holds 33 body points per frame in metres. Paste this script into Blender's Scripting tab, set the path, and run it: it builds an animated rig of empties you can parent a character to.</p></div>
  <div class="panel" style="display:grid;gap:10px"><div class="row-ctrl"><button class="btn" id="copyScript">Copy script</button><span class="muted" id="copyMsg" style="font-size:12px"></span></div><pre class="code" id="bpy"></pre></div>
</section>`;

$("#page-live").innerHTML = `
<section>
  <div class="sec-head"><div class="eyebrow">Live · webcam</div><h1>Become the character in real time</h1>
  <p class="lede">Stand back so your whole body is in view. Everyone in front of the camera, up to five people, gets their own character and copies their own moves live. Pick a character per person, leave someone unchanged, and record the result.</p></div>
  <div class="studio">
    <div style="display:grid;gap:12px">
      <div class="stage" id="liveStage"><div class="empty" id="liveEmpty"></div><canvas id="liveOut" hidden></canvas><div class="status" id="liveStatus"></div></div>
      <div class="row-ctrl"><button class="btn on" id="camBtn">Start camera</button><button class="btn" id="camFlip" hidden title="Switch between the front and back camera">Switch camera</button><button class="btn" id="liveRec" disabled>Record video</button><span class="muted" id="liveMsg" style="font-size:12px"></span></div>
    </div>
    <div class="ctrl panel">
      <div><div class="muted" style="font-size:12.5px;margin-bottom:6px">Character for everyone (change one person below)</div><div class="charpick" id="livePick"></div></div>
      <label>Mode <select id="liveMode"><option value="overlay">Character over me</option><option value="replace">Replace me</option><option value="solo">Character only</option><option value="three" selected>Three screens: camera · pose · both</option></select></label>
      <label>Colour <input type="color" id="liveColor"></label>
      <label>People to track <select id="liveMax"><option value="1">1 (just me)</option><option value="2">2</option><option value="3">3</option><option value="4" selected>4</option><option value="5">5</option></select></label>
      <label style="flex-direction:row;display:flex;gap:6px;align-items:center"><input type="checkbox" id="liveMix"> A different character for each new person</label>
      <div><div class="muted" style="font-size:12.5px;margin-bottom:6px">People in view · choose a character for each</div><div class="people" id="livePeople"><p class="note" style="margin:0">Start the camera to see who is in view.</p></div></div>
      <div class="panel phoneqr" style="background:var(--panel-2)"><div><h3>Open on your phone</h3>
        <p class="note" style="margin:0">Scan with your phone's camera to open this page there. Your phone's camera works with the front and back lens.</p></div><div id="qr" aria-label="QR code for the live camera page"></div></div>
      <div class="panel" style="background:var(--panel-2);display:grid;gap:8px"><h3>If the camera does not start</h3>
        <ul class="note" style="margin:0;padding-left:18px"><li>Allow the camera when the browser asks. If you said no, turn it back on in the site settings (the icon next to the address).</li>
        <li>On a phone, open the link in Safari or Chrome, not inside Instagram, WhatsApp or another app.</li>
        <li>Close other apps that are using the camera.</li>
        <li>Pages shown inside Claude cannot use a camera. Use <a href="https://vbobbadi.github.io/rw2b-studio/#live" target="_blank" rel="noopener">vbobbadi.github.io/rw2b-studio</a>.</li></ul></div>
    </div>
  </div>
</section>`;

$("#page-characters").innerHTML = `
<section>
  <div class="sec-head"><div class="eyebrow">Library</div><h1>Pick a character</h1>
  <p class="lede">Every character is driven by the same 18-point body, so any of them works on uploads, the live camera and our model output. The cards marked 3D are fully rigged characters downloaded from Mixamo and the three.js examples; you can add your own below. These previews replay our DGS model's 3D pose from the dance clip (row 2). Drag any preview to turn it.</p></div>
  <div class="lib" id="lib"></div>
</section>
<section>
  <div class="sec-head"><div class="eyebrow">Mixamo characters</div><h2>Add rigged 3D characters from Mixamo</h2>
  <p class="lede">Mixamo characters are free with an Adobe account, and its licence lets you use them in your own projects but not hand out the files. So you download them yourself and drop them here: they load from your device, are kept only in this browser, and then appear in the library, the Pose Studio and the Live Camera.</p></div>
  <div class="studio">
    <div class="panel" style="display:grid;gap:10px">
      <h3>Get a character</h3>
      <ol class="note" style="margin:0;padding-left:18px;display:grid;gap:4px">
        <li>Open <a href="https://www.mixamo.com/#/?type=Character" target="_blank" rel="noopener">mixamo.com</a> and sign in with a free Adobe account.</li>
        <li>Characters tab: pick one, for example Y Bot, X Bot, Remy, Kaya or Mutant.</li>
        <li>Download with <b>Format: FBX Binary</b>, <b>Pose: T-pose</b>, <b>Skin: With Skin</b>. No animation is needed; the pose comes from your video.</li>
        <li>Drop the .fbx file on the right. A .glb with a Mixamo-style skeleton also works.</li>
      </ol>
    </div>
    <div class="panel ctrl">
      <div class="drop" id="rigDrop"><b>Drop a Mixamo .fbx here</b><span class="muted" style="font-size:12.5px">Up to 60 MB · stays on this device</span><label class="btn" for="rigFile">Choose a file</label><input type="file" id="rigFile" accept=".fbx,.glb,.gltf" hidden></div>
      <div class="muted" id="rigMsg" style="font-size:12.5px"></div>
      <div id="rigList" style="display:grid;gap:6px"></div>
    </div>
  </div>
</section>
<section>
  <div class="sec-head"><div class="eyebrow">Make your own</div><h2>Upload a face or logo for the head</h2>
  <p class="lede">Choose a body style and colour, upload a square image (PNG with transparency looks best), and it becomes a character in every picker. It is stored only in this browser.</p></div>
  <div class="studio">
    <div class="panel ctrl">
      <label>Name <input id="cName" value="My character" style="font:500 13px var(--body);color:var(--ink);background:var(--panel-2);border:1px solid var(--line);border-radius:8px;padding:7px 9px"></label>
      <label>Body style <select id="cBase"></select></label>
      <label>Colour <input type="color" id="cColor" value="#ff8a3d"></label>
      <div class="drop" id="cDrop"><b>Head image</b><span class="muted" style="font-size:12.5px">PNG or JPG, up to 2 MB</span><label class="btn" for="cFile">Choose an image</label><input type="file" id="cFile" accept="image/*" hidden></div>
      <div class="row-ctrl"><button class="btn on" id="cSave" disabled>Add to library</button><button class="btn" id="cDel" ${CUSTOM ? "" : "hidden"}>Remove my character</button><span class="muted" id="cMsg" style="font-size:12px"></span></div>
    </div>
    <div class="libcard"><canvas id="cPrev" style="aspect-ratio:1"></canvas><div class="body"><h3>Preview</h3><p>Updates as you change the options.</p></div></div>
  </div>
</section>`;

$("#page-how").innerHTML = `
<section>
  <div class="sec-head"><div class="eyebrow">Pipeline</div><h1>From a phone video to an animated Blender character</h1>
  <p class="lede">The research system runs on a Colab GPU. The Studio pages use a lighter browser model so anyone can try the idea without a GPU.</p></div>
  <div class="flow">
    <div class="step"><b>01 · Capture</b><h3>Phone video</h3><p>Any clip; the 29-row test plan covers people, occlusion, body position, camera and video quality.</p></div>
    <div class="step"><b>02 · Find people</b><h3>YOLO + tracker</h3><p>Boxes per frame, linked into one track per person. Replaces the broken PHALP setup.</p></div>
    <div class="step"><b>03 · Diagnose</b><h3>Problem detector</h3><p>ResNet-18 scores occlusion, image quality and framing, plus which joints are hidden.</p></div>
    <div class="step"><b>04 · Specialize</b><h3>4 LoRA experts</h3><p>HMR2.0 adapters trained on Blender renders: clean, occlusion, quality, framing.</p></div>
    <div class="step"><b>05 · Fuse</b><h3>Rotation fusion</h3><p>Weighted, per-joint or learned averaging of the experts' SMPL rotations on SO(3).</p></div>
    <div class="step"><b>06 · Smooth</b><h3>Savitzky–Golay</h3><p>Removes frame-to-frame jitter: raw 14 mm down to about 2 mm.</p></div>
    <div class="step"><b>07 · Animate</b><h3>Blender</h3><p>SMPL rotations drive a rig; the scene is saved as .blend and rendered to .mp4.</p></div>
  </div>
</section>
<section>
  <div class="sec-head"><div class="eyebrow">Training data</div><h2>Synthetic people from Blender</h2>
  <p class="lede">AMASS motions rendered on SMPL bodies with 3D-placed occluders (D), 2D-pasted occluders (C) or none (B), plus image damage for the quality and framing experts. The detector and experts never see a real image during training, which is why the Results pages test them on 3DPW and stock video.</p></div>
  <div class="grid2">
    <div class="panel"><h3>What runs where</h3><table><thead><tr><th>Part</th><th>Where</th><th>Model</th></tr></thead><tbody>
      <tr><td>Pose Studio, Live Camera</td><td>your browser</td><td>MediaPipe Pose (33 points)</td></tr>
      <tr><td>Research pipeline</td><td>Colab GPU</td><td>HMR2.0 + 4 experts + fusion</td></tr>
      <tr><td>Evaluation</td><td>Colab GPU</td><td>synthetic test, 3DPW, 29-row clips</td></tr>
      <tr><td>Animation</td><td>Blender 4.2</td><td>SMPL-driven rig</td></tr></tbody></table></div>
    <div class="panel"><h3>Notebooks</h3><table><tbody>
      <tr><td>04</td><td>Blender synthetic data</td></tr><tr><td>06</td><td>RQ2 · three seeds, 2D vs 3D occlusion</td></tr><tr><td>07</td><td>RQ4 · experts and generalist</td></tr>
      <tr><td>08</td><td>RQ5/RQ6 · detector and fusion</td></tr><tr><td>09</td><td>RQ1 · real clips, Blender export</td></tr><tr><td>10</td><td>RQ3 · adaptive data selection</td></tr>
      <tr><td>11</td><td>Stock clips for the test plan</td></tr><tr><td>12</td><td>3DPW evaluation</td></tr></tbody></table></div>
  </div>
</section>`;


/* =========================== contributions ========================== */
const TEAM = [
  {name:"Nishitha", c:"#ff8a3d", area:"Synthetic data in Blender", tags:["Notebook 04","RQ2 data"], work:["AMASS motions on SMPL bodies rendered in Blender","3D-placed occluders (D) and 2D-pasted occluders (C)","Per-joint visibility and self-occlusion labels","Image damage for the quality and framing experts"]},
  {name:"Salma", c:"#7fb0e8", area:"Model training", tags:["Notebooks 06, 07","RQ2, RQ4"], work:["HMR2.0 fine-tuning without PHALP or detectron2","Three-seed runs for 2D vs 3D occlusion","Four LoRA experts and the matched generalist","Seed-level bootstrap statistics"]},
  {name:"Gnanitha", c:"#c3a6ff", area:"Diagnose and fuse", tags:["Notebooks 08, 10","RQ3, RQ5, RQ6"], work:["ResNet-18 problem detector and visibility head","Weighted, per-joint and learned rotation fusion","Hedge bandit vs random vs hard-example mining","Detector checks on synthetic and real data"]},
  {name:"Venugopal", c:"#5fd3b5", area:"Real-video evaluation", tags:["Notebooks 09, 11, 12","RQ1"], work:["YOLO + tracker video pipeline","29-row test plan clips and proxy metrics","3DPW evaluation with ground truth","Baselines: MediaPipe, PARE, WHAM, GVHMR, Multi-HMR"]},
  {name:"Madeeha", c:"#f2c14e", area:"Blender export, studio and report", tags:["Blender 4.2","This site"], work:["SMPL-driven Blender rig, .blend and .mp4 export","Pose Studio, Live Camera and Character Library","Mixamo character support and motion export","Results pages and final report"]}
];
$("#page-team").innerHTML = `
<section>
  <div class="sec-head"><div class="eyebrow">Team</div><h1>Contributions</h1>
  <p class="lede">The project split into five work packages, one per team member. Everyone reviewed the results together.</p></div>
  <div class="team">${TEAM.map((m, i) => `<article class="member" style="--c:${m.c}"><div class="who"><div class="av" aria-hidden="true">${m.name[0]}</div><div><h3>${m.name}</h3><div class="muted" style="font-size:12.5px">Work package ${i + 1} · ${m.area}</div></div></div>
    <div class="row-ctrl">${m.tags.map(t => `<span class="chip">${t}</span>`).join("")}</div><ul>${m.work.map(w => `<li>${w}</li>`).join("")}</ul></article>`).join("")}</div>
</section>`;


/* =========================== architecture =========================== */
const ST = {ok:["Built","good"], weak:["Built · weak on real video","warn"], used:["Used, not trained by us","acc"], plan:["Not built yet",""]};
const ARCH = window.ARCH = [
  {n:"00", t:"Input", d:"Anything with people in it.", nodes:[
    {id:"in_video", name:"Phone video", s:"ok", what:"Any clip. The 29-row test plan covers people, occlusion, body position, camera and video quality.", io:"→ frames", ev:"24 test clips in videos/rows/, 3DPW for ground truth."},
    {id:"in_photo", name:"Photo", s:"ok", what:"Single images go through the same path, without tracking or smoothing.", io:"→ one frame", ev:"Pose Studio accepts jpg and png."},
    {id:"in_cam", name:"Webcam", s:"ok", what:"Live frames in the browser studio.", io:"→ frames at ~30 fps", ev:"Works in the offline copy; hosted pages block cameras."}]},
  {n:"01", t:"Find every person", d:"Detect people in each frame and keep one identity per person over time.", nodes:[
    {id:"yolo", name:"YOLO11 person detector", s:"used", what:"Finds a box around every person in each frame.", io:"frame → person boxes", ev:"Notebook 09. Replaced the broken PHALP / detectron2 setup."},
    {id:"iou", name:"IoU tracker", s:"weak", what:"Links boxes into one track per person.", io:"boxes → person tracks", ev:"Keeps the main person in only 34–61% of frames on far, low-light, crowd and hug clips; splits people into several tracks."},
    {id:"mpmulti", name:"Browser multi-person pose", s:"used", what:"MediaPipe Pose Landmarker finds up to 5 people per frame; our matcher keeps their ids stable.", io:"frame → 33 points per person + mask", ev:"Pose Studio and Live Camera."},
    {id:"bytetrack", name:"Re-identifying tracker", s:"plan", what:"Appearance + motion tracking (ByteTrack / PHALP style) so people keep their id after crossing or occlusion.", io:"boxes + crops → stable tracks", ev:"Top fix from RQ1: tracking is the biggest real-video failure."}]},
  {n:"02", t:"Choose people and characters", d:"Decide who becomes which character, or who stays unchanged.", nodes:[
    {id:"select", name:"Person and character selection", s:"ok", what:"Click a person, pick a character, or give everyone the same or a different one.", io:"tracks → assignment {person: character}", ev:"Pose Studio people list; Live Camera mix option."},
    {id:"library", name:"Character library", s:"ok", what:"7 built-in characters, your own head image, and rigged Mixamo characters you load yourself.", io:"→ character to render", ev:"Character Library page."}]},
  {n:"03", t:"Identify the problem", d:"Score what is wrong with this person in this frame, so the right specialist is trusted.", nodes:[
    {id:"detector", name:"Problem identifier (ResNet-18)", s:"weak", what:"Predicts occlusion, image quality and framing problems, plus which joints are hidden.", io:"person crop → 3 problem scores + 24 visibility scores", ev:"AUC 0.9998 on Blender renders, but 0.50 / 0.33 / 0.54 on real clips (chance is 0.5). Trained only on synthetic images."},
    {id:"cheap", name:"Simple image checks", s:"ok", what:"Brightness, blur, box size and border contact, computed directly from pixels.", io:"frame + box → quality / framing flags", ev:"Beat the trained detector on real clips (AUC 0.56 quality, 0.68 framing)."},
    {id:"camdet", name:"Camera-motion check", s:"plan", what:"Detect handheld shake, following and orbiting cameras (rows 17–19).", io:"frame pair → camera motion", ev:"Raw optical-flow check exists in notebook 09; no model or expert uses it yet."},
    {id:"contactdet", name:"Contact / interaction check", s:"plan", what:"Detect people touching or overlapping (rows 4–6).", io:"two tracks → contact score", ev:"Not built."}]},
  {n:"04", t:"Route", d:"Turn the problem scores into how much each specialist is trusted.", nodes:[
    {id:"router", name:"Router", s:"ok", what:"No problem → clean expert; each problem score → weight of its expert.", io:"problem scores → expert weights", ev:"pose_lib/detector.router_weights; oracle routing used as the upper bound."}]},
  {n:"05", t:"Specialists on the base model", d:"Small LoRA adapters on top of HMR2.0, one per problem family.", nodes:[
    {id:"e0", name:"E0 · clean", s:"ok", what:"Trained on clean Blender renders.", io:"crop → SMPL pose", ev:"Synthetic clean 32.2 mm · 3DPW 44.1 mm PA-MPJPE. Rows 1, 16."},
    {id:"e1", name:"E1 · occlusion", s:"ok", what:"Trained on 3D-placed occluders.", io:"crop → SMPL pose", ev:"Best expert on occlusion (62.1 mm) · 3DPW 45.3 mm, best orientation (73.6 mm MPJPE). Rows 7, 8, 10, 11."},
    {id:"e2", name:"E2 · quality", s:"ok", what:"Trained on blur, low light, low resolution and JPEG damage.", io:"crop → SMPL pose", ev:"Synthetic quality 39.7 mm · 3DPW 45.1 mm. Rows 20–22."},
    {id:"e3", name:"E3 · framing", s:"ok", what:"Trained on cut-off, far and close framing.", io:"crop → SMPL pose", ev:"Synthetic framing 46.7 mm · 3DPW 43.9 mm, best expert. Rows 12–15."},
    {id:"e4", name:"Fast-motion expert", s:"plan", what:"For motion blur and limb flips in fast movement (row 2).", io:"crop sequence → pose", ev:"Not built; E2 covers blur only."},
    {id:"e5", name:"Contact expert", s:"plan", what:"For hugs and handshakes where bodies overlap (row 6).", io:"two crops → two poses", ev:"Not built."},
    {id:"e6", name:"World-space / camera expert", s:"plan", what:"Separates body motion from camera motion (rows 17–19).", io:"sequence → pose + global path", ev:"Not built; WHAM / GVHMR were planned as baselines only."}]},
  {n:"06", t:"Base model", d:"The network every specialist plugs into, and the alternative that replaces all of them.", nodes:[
    {id:"hmr2", name:"HMR2.0 (ViT-H)", s:"used", what:"Released pretrained model; frozen except for the adapters.", io:"256×256 crop → SMPL pose, shape, camera", ev:"Synthetic 46.8 / 61.5 / 73.1 mm (clean / 2D / 3D occlusion) · 3DPW 47.9 mm."},
    {id:"gen", name:"GEN generalist", s:"ok", what:"One adapter trained on all four experts' data, same total size and compute.", io:"crop → SMPL pose", ev:"Best everywhere: synthetic 28.4–68.1 mm, 3DPW 42.9 mm. Beats every expert and every fusion."},
    {id:"ft", name:"Full fine-tunes B / C / D", s:"ok", what:"Last 4 blocks fine-tuned on clean, 2D-pasted and 3D-rendered occlusion (RQ2).", io:"crop → SMPL pose", ev:"3D-rendered beats 2D-pasted by 2.1 mm on occluded test images, three seeds."}]},
  {n:"07", t:"Fuse", d:"Combine the trusted specialists into one pose per person.", nodes:[
    {id:"fw", name:"Weighted rotation average", s:"ok", what:"Average of SMPL joint rotations, projected back to valid rotations.", io:"expert poses + weights → pose", ev:"Synthetic mean 51.0 mm · 3DPW 44.3 mm."},
    {id:"fpj", name:"Per-joint fusion", s:"ok", what:"Hidden joints lean on the occlusion expert, visible joints on the others.", io:"+ joint visibility → pose", ev:"51.1 mm · 3DPW 44.3 mm."},
    {id:"fl", name:"Learned fusion", s:"ok", what:"Small network that picks per-joint expert weights.", io:"problem + visibility scores → per-joint weights", ev:"Best fusion on synthetic (50.6 mm); tied on 3DPW."}]},
  {n:"08", t:"Smooth over time", d:"Remove frame-to-frame shaking.", nodes:[
    {id:"sg", name:"Savitzky–Golay smoothing", s:"ok", what:"Smooths joint rotations along each track.", io:"pose sequence → smooth sequence", ev:"Raw jitter ~14 mm → ~2 mm on the 29-row clips."},
    {id:"temporal", name:"Learned temporal model", s:"plan", what:"Uses neighbouring frames to fix poses, not just smooth them.", io:"sequence → corrected sequence", ev:"Not built."}]},
  {n:"09", t:"Animate", d:"Turn each person's pose into a character.", nodes:[
    {id:"blender", name:"Blender SMPL rig", s:"ok", what:"SMPL rotations drive an animated body; saved as .blend and rendered to .mp4.", io:"pose sequence → .blend + .mp4", ev:"outputs/clipB__dgs_perjoint+sg.blend and .mp4."},
    {id:"web", name:"Browser characters", s:"ok", what:"Every person gets their chosen character drawn over, instead of, or beside the video.", io:"33 points per person → character video", ev:"Pose Studio: record .mp4, export motion JSON for Blender."},
    {id:"mixblend", name:"Mixamo retarget in Blender", s:"plan", what:"Map SMPL rotations onto a Mixamo character inside Blender automatically.", io:"pose sequence + Mixamo rig → animated character", ev:"Done in the browser only; Blender gets empties to parent a rig to."}]}
];
$("#page-arch").innerHTML = `
<section>
  <div class="sec-head"><div class="eyebrow">Architecture</div><h1>Identify the problem, route to a specialist, run the base model</h1>
  <p class="lede">Watch one frame travel through the system. Each person is found, the problem identifier scores what is wrong, the router splits the work across the specialists on top of the base model, and their outputs are fused into one pose that drives a character. Change the scenario or the scores and the flow re-routes.</p></div>
  <div id="flow"></div>
</section>
<section>
  <div class="sec-head"><div class="eyebrow">Stage by stage</div><h2>Every part, built or not</h2>
  <p class="lede">The same system as a list. Every box is coloured by whether we built it. Click a box for what it does, its input and output, and the evidence.</p></div>
  <div class="row-ctrl" id="archLegend">${Object.values(ST).map(([l, c]) => `<span class="chip ${c}">${l}</span>`).join("")}</div>
  <div class="archwrap"><div class="arch" id="arch"></div>
    <aside class="panel archinfo" id="archInfo"><div class="eyebrow">Details</div><h3>Click any box</h3><p class="note">The panel shows what the part does, what goes in and out, and the numbers behind its colour.</p></aside></div>
</section>
<section>
  <div class="sec-head"><div class="eyebrow">Inventory</div><h2>What is built and what is not</h2></div>
  <div class="grid2">
    <div class="panel"><h3>Trained by us</h3><table><tbody>
      <tr><td>Problem identifier (ResNet-18)</td><td>notebook 08</td></tr><tr><td>Experts E0, E1, E2, E3 (LoRA)</td><td>notebook 07</td></tr>
      <tr><td>GEN generalist (LoRA)</td><td>notebook 07</td></tr><tr><td>Full fine-tunes B, C, D × 3 seeds</td><td>notebook 06</td></tr>
      <tr><td>Learned fusion network</td><td>notebook 08</td></tr><tr><td>Hedge data selector (RQ3)</td><td>notebook 10</td></tr></tbody></table></div>
    <div class="panel"><h3>Used as-is</h3><table><tbody>
      <tr><td>HMR2.0 ViT-H (base model)</td><td>4D-Humans release</td></tr><tr><td>YOLO11n person detector</td><td>Ultralytics</td></tr>
      <tr><td>MediaPipe Pose Landmarker</td><td>browser studio</td></tr><tr><td>MediaPipe heavy / full / lite</td><td>real-video baseline</td></tr></tbody></table></div>
    <div class="panel"><h3>Not built yet</h3><table><tbody>
      <tr><td>Re-identifying tracker</td><td>rows 3–5, 9, 23</td></tr><tr><td>Camera-motion check + world-space expert</td><td>rows 17–19</td></tr>
      <tr><td>Contact check + contact expert</td><td>rows 4–6</td></tr><tr><td>Fast-motion expert</td><td>row 2</td></tr>
      <tr><td>Problem identifier trained on real video</td><td>RQ6</td></tr><tr><td>Learned temporal model</td><td>all video rows</td></tr>
      <tr><td>Mixamo retarget inside Blender</td><td>rows 25–29</td></tr></tbody></table></div>
    <div class="panel"><h3>What the results say about this design</h3><p class="note" style="margin:0">The routing design is built end to end, but two findings argue for a simpler path today: the GEN generalist beats every specialist and every fusion (3DPW 42.9 vs 44.3 mm), and the problem identifier does not recognise problems in real video. The practical pipeline is tracker → GEN → smoothing → Blender, with the identifier kept for reporting what went wrong rather than for routing, until it is retrained on real frames.</p></div>
  </div>
</section>`;
(() => {
  const el = $("#arch");
  ARCH.forEach((stg, i) => {
    const row = document.createElement("div"); row.className = "stage-row";
    row.innerHTML = `<div class="stage-lab"><span class="num">${stg.n}</span><h3>${stg.t}</h3><p>${stg.d}</p></div><div class="stage-nodes"></div>`;
    stg.nodes.forEach(nd => { const b = document.createElement("button"); b.className = "anode " + nd.s; b.innerHTML = `<b>${nd.name}</b><span>${ST[nd.s][0]}</span>`; b.onclick = () => {
        el.querySelectorAll(".anode").forEach(x => x.classList.toggle("sel", x === b));
        $("#archInfo").innerHTML = `<div class="eyebrow">Stage ${stg.n} · ${stg.t}</div><h3>${nd.name}</h3><span class="chip ${ST[nd.s][1]}">${ST[nd.s][0]}</span><p>${nd.what}</p><div class="io"><span class="muted">Input → output</span><b class="num">${nd.io}</b></div><div class="io"><span class="muted">Evidence</span><p style="margin:0">${nd.ev}</p></div>`; };
      row.querySelector(".stage-nodes").appendChild(b); });
    el.appendChild(row);
    if (i < ARCH.length - 1){ const a = document.createElement("div"); a.className = "arrow"; a.setAttribute("aria-hidden", "true"); el.appendChild(a); }
  });

})();

/* ======================= shared pose engine ========================= */
/* MediaPipe Tasks PoseLandmarker: up to 5 people per frame, with a segmentation mask per person. */
let engine = null, engineKey = "";
const mpReady = new Promise(res => { if (window.MPV) res(window.MPV); else addEventListener("mpv-ready", () => res(window.MPV), {once:true}); });
async function getEngine(model, numPoses){
  const key = model + "/" + numPoses; if (engine && engineKey === key) return engine;
  const MPV = await Promise.race([mpReady, new Promise((_, rej) => setTimeout(() => rej(new Error("the pose runtime did not load")), 20000))]);
  const files = await MPV.FilesetResolver.forVisionTasks(new URL("mp", location.href).href);
  const buf = new Uint8Array(await (await fetch(`mp/pose_landmarker_${model}.task.wasm`)).arrayBuffer());   // the host serves the model under a .wasm name
  const opts = d => ({baseOptions:{modelAssetBuffer:buf, delegate:d}, runningMode:"VIDEO", numPoses:+numPoses, outputSegmentationMasks:true, minPoseDetectionConfidence:0.5, minPosePresenceConfidence:0.5, minTrackingConfidence:0.5});
  // some phones hang on the GPU delegate instead of failing, so give it 12 s before falling back to the CPU
  let lm; try { lm = await Promise.race([MPV.PoseLandmarker.createFromOptions(files, opts("GPU")), new Promise((_, rej) => setTimeout(() => rej(new Error("GPU timeout")), 12000))]); }
  catch (e) { lm = await MPV.PoseLandmarker.createFromOptions(files, opts("CPU")); }
  if (engine && engine.close) engine.close(); engine = lm; engineKey = key; engine.mode = "VIDEO"; engine.t = 0; return engine;
}
/* one call for both videos and photos -> [{lm:[[x,y,z,vis]], world:[[x,y,z]], mask:{w,h,data}}] */
async function detect(src, isVideo){
  if (!engine) return [];
  const want = isVideo ? "VIDEO" : "IMAGE"; if (engine.mode !== want){ await engine.setOptions({runningMode:want}); engine.mode = want; }
  const r = isVideo ? engine.detectForVideo(src, engine.t = Math.max(engine.t + 1, Math.round(performance.now()))) : engine.detect(src);
  const people = (r.landmarks || []).map((l, i) => {
    let mask = null; const m = r.segmentationMasks && r.segmentationMasks[i];
    if (m){ try { mask = {w:m.width, h:m.height, data:m.getAsFloat32Array().slice()}; } catch (e) {} }
    return {lm:l.map(p => [p.x, p.y, p.z, p.visibility == null ? 1 : p.visibility]), world:(r.worldLandmarks && r.worldLandmarks[i] || []).map(p => [p.x, p.y, p.z]), mask};
  });
  if (r.segmentationMasks) r.segmentationMasks.forEach(m => m && m.close && m.close());
  return people;
}
function chip(text, cls = ""){ return `<span class="chip ${cls}" style="background:#0f1115e6">${text}</span>`; }

/* EMA smoothing of landmark arrays */
function smoother(){ let prev = null; return (L, a) => { if (!L){ prev = null; return null; } if (!prev || a <= 0){ prev = L.map(p => [...p]); return prev; } prev = L.map((p, i) => p.map((v, j) => prev[i][j] * a + v * (1 - a))); return prev; }; }

/* Keeps the same id for the same person across frames (nearest hip centre). */
function PersonTracker(){
  const tracks = new Map(); let next = 1;
  return {
    update(people){
      const hip = p => [(p.lm[23][0] + p.lm[24][0]) / 2, (p.lm[23][1] + p.lm[24][1]) / 2];
      const free = new Set(tracks.keys()), out = [];
      people.map(p => ({p, h:hip(p)})).sort((a, b) => a.h[0] - b.h[0]).forEach(({p, h}) => {
        let best = null, bd = 0.25; free.forEach(id => { const t = tracks.get(id), d = Math.hypot(t.h[0] - h[0], t.h[1] - h[1]); if (d < bd){ bd = d; best = id; } });
        if (best == null){ best = next++; tracks.set(best, {h, sm:smoother(), sm3:smoother(), miss:0}); }
        free.delete(best); const t = tracks.get(best); t.h = h; t.miss = 0; out.push({id:best, ...p, t});
      });
      free.forEach(id => { const t = tracks.get(id); if (++t.miss > 30) tracks.delete(id); });
      return out.sort((a, b) => a.id - b.id);
    },
    reset(){ tracks.clear(); next = 1; }
  };
}
const PCOLORS = ["#ff8a3d","#7fb0e8","#c3a6ff","#5fd3b5","#f2c14e","#ff6b9a","#9ad15a"];
const pcol = id => PCOLORS[(id - 1) % PCOLORS.length];

/* Compose one output frame: video, then for each person their character */
const PANES = [["cam", "Camera"], ["pose", "Pose only"], ["both", "Both"]];
function compose(ctx, W, H, src, people, opt){
  if (opt.mode === "three"){ // camera and pose side by side on top, both together large below
    const h = H / 1.5, R = [[0, 0, W / 2, h / 2], [W / 2, 0, W / 2, h / 2], [0, h / 2, W, h]];
    ctx.fillStyle = "#0c0e12"; ctx.fillRect(0, 0, W, H);
    PANES.forEach(([k, label], i) => { const [x, y, w, ph] = R[i]; ctx.save(); ctx.translate(x, y); ctx.beginPath(); ctx.rect(0, 0, w, ph); ctx.clip();
      if (k === "cam") compose(ctx, w, ph, src, people.map(p => ({...p, char:null})), {...opt, mode:"overlay", skel:false, labels:false});
      else if (k === "pose") compose(ctx, w, ph, src, people, {...opt, mode:"solo", skel:true, labels:false});
      else compose(ctx, w, ph, src, people, {...opt, mode:"overlay"});
      const f = Math.max(12, Math.round(h / 34)); ctx.font = `600 ${f}px IBM Plex Mono, ui-monospace, monospace`; const tw = ctx.measureText(label).width;
      ctx.fillStyle = "#0f1115d0"; ctx.fillRect(f * .6, f * .6, tw + f * 1.2, f * 1.9); ctx.fillStyle = i === 2 ? "#ff8a3d" : "#e9ebef"; ctx.fillText(label, f * 1.2, f * 1.95);
      ctx.restore(); });
    ctx.fillStyle = "#2a2f39"; ctx.fillRect(W / 2 - 1, 0, 2, h / 2); ctx.fillRect(0, h / 2 - 1, W, 2);
    return; }
  const side = opt.mode === "side", vw = side ? W / 2 : W;
  ctx.save(); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  if (opt.mirror){ ctx.translate(vw, 0); ctx.scale(-1, 1); }
  if (opt.mode !== "solo") ctx.drawImage(src, 0, 0, vw, H);
  if (opt.mode === "replace") people.forEach(p => { if (!p.char || !p.mask) return;
    const m = compose.tmp || (compose.tmp = document.createElement("canvas")); m.width = p.mask.w; m.height = p.mask.h; const mc = m.getContext("2d");
    const img = mc.createImageData(p.mask.w, p.mask.h); for (let i = 0; i < p.mask.data.length; i++){ const a = p.mask.data[i]; img.data[i*4] = 16; img.data[i*4+1] = 19; img.data[i*4+2] = 24; img.data[i*4+3] = a > 0.5 ? 235 : a * 400; }
    mc.putImageData(img, 0, 0); ctx.drawImage(m, 0, 0, vw, H); });
  ctx.restore();
  if (side){ ctx.fillStyle = "#0c0e12"; ctx.fillRect(W / 2, 0, W / 2, H); }
  const toPx = (L, x0) => L.map(p => [x0 + (opt.mirror ? 1 - p[0] : p[0]) * vw, p[1] * H, p[2] * vw]);
  [...people].sort((a, b) => (b.L2[0][2]) - (a.L2[0][2])).forEach(p => {
    if (!p.L2) return; const B = fromMP(toPx(p.L2, side ? W / 2 : 0));
    if (p.char) drawCharacter(ctx, p.char, B, {color:p.color, P3:p.W3 ? fromMP(p.W3) : null, view:{mirror:opt.mirror}});
    if (opt.skel) drawSkeleton(ctx, fromMP(toPx(p.L2, 0)), pcol(p.id));
    if (opt.labels){ const h = fromMP(toPx(p.L2, 0)).head, sel = opt.selected === p.id; ctx.save(); ctx.font = "600 13px IBM Plex Sans, system-ui, sans-serif"; const t = `Person ${p.id}`, w = ctx.measureText(t).width + 14;
      ctx.fillStyle = sel ? pcol(p.id) : "#0f1115dd"; ctx.strokeStyle = pcol(p.id); ctx.lineWidth = 2; const x = h[0] - w / 2, y = Math.max(4, h[1] - unit(fromMP(toPx(p.L2, 0))) * 1.1 - 26);
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, 22, 11) : ctx.rect(x, y, w, 22); ctx.fill(); ctx.stroke(); ctx.fillStyle = sel ? "#111" : "#e9ebef"; ctx.fillText(t, x + 7, y + 15); ctx.restore(); }
  });
}
function draw3D(stage, W3, charKey, color){
  stage.fit(); stage.clear(); const sc = Math.min(stage.canvas.clientWidth, stage.canvas.clientHeight) / 2.3; stage.grid(sc, 0.9);
  if (!W3 || !W3.length) return; const B = fromMP(W3.map(p => stage.project(p, sc, 0.45)));
  drawCharacter(stage.ctx, charKey || "mannequin", B, {color, P3:fromMP(W3), view:{yaw:stage.yaw, pitch:stage.pitch}});
}

/* ========================= character pickers ======================== */
function buildPicker(el, get, set, withNone){
  el.innerHTML = ""; const chars = allChars(); const sample = Library.sampleBody();
  const entries = Object.entries(chars); if (withNone) entries.unshift(["none", {name:"Leave as is", blurb:"Show this person unchanged", color:"#9aa1ad"}]);
  entries.forEach(([k, ch]) => { const b = document.createElement("button"); b.innerHTML = `<canvas width="120" height="120"></canvas>${ch.name}`; b.dataset.k = k; b.title = ch.blurb;
    b.onclick = () => { set(k); [...el.children].forEach(c => c.classList.toggle("on", c === b)); }; el.appendChild(b);
    const c = b.querySelector("canvas"), cx = c.getContext("2d");
    if (k === "none"){ cx.fillStyle = "#9aa1ad"; cx.font = "40px system-ui"; cx.textAlign = "center"; cx.fillText("∅", 60, 74); }
    else if (sample){ const s = sample(120, 120); drawCharacter(cx, k, s.B, {color:ch.color, P3:s.P3, view:{}}); } });
  [...el.children].forEach(c => c.classList.toggle("on", c.dataset.k === get()));
}

/* ============================== studio =============================== */
const studio = {src:null, kind:null, frames:[], playing:false, people:[], assign:{}, selected:1, defaultChar:"neon", policy:"same"};
const out = $("#out"), octx = out.getContext("2d"), st3 = Stage($("#pose3d"), {yaw:-0.5, pitch:0.15}), tracker = PersonTracker();
st3.onchange = () => render3D();
function assignOf(id){ if (!studio.assign[id]){ const keys = Object.keys(allChars()); studio.assign[id] = {char: studio.policy === "mix" ? keys[(id - 1) % keys.length] : studio.defaultChar, color: null, auto: true}; } return studio.assign[id]; }
function render3D(){ const p = studio.people.find(q => q.id === studio.selected) || studio.people[0]; const a = p ? assignOf(p.id) : null;
  draw3D(st3, p && p.W3, a && a.char !== "none" ? a.char : "mannequin", a && (a.color || (allChars()[a.char] || {}).color)); }
function setStatus(html){ $("#status").innerHTML = html; }
function opts(){ return {mode:$("#mode").value, skel:$("#skel").checked, mirror:$("#mirror").checked, labels:$("#labels").checked, selected:studio.selected}; }
let lastFrameT = 0, fps = 0, lastIds = "";
async function process(){
  const src = studio.src; if (!src) return;
  const raw = await detect(src, studio.kind === "video"); const a = +$("#smooth").value;
  studio.people = tracker.update(raw).map(p => { const as = assignOf(p.id); const ch = as.char === "none" ? null : as.char;
    return {id:p.id, mask:p.mask, L2:p.t.sm(p.lm, a), W3:p.t.sm3(p.world, a), char:ch, color:as.color || (ch ? allChars()[ch].color : null), vis:p.lm.filter(q => q[3] > 0.5).length}; });
  compose(octx, out.width, out.height, src, studio.people, opts()); render3D();
  if (studio.kind === "video" && studio.people.length) studio.frames.push({t:+src.currentTime.toFixed(3), people:studio.people.map(p => ({id:p.id, character:p.char, world:p.W3.map(q => q.map(v => +v.toFixed(4))), image:p.L2.map(q => q.slice(0, 3).map(v => +v.toFixed(4)))}))});
  const ids = studio.people.map(p => p.id).join(","); if (ids !== lastIds){ lastIds = ids; if (studio.people.length && !studio.people.some(p => p.id === studio.selected)) selectPerson(studio.people[0].id); else renderPeople(); }
  const now = performance.now(); fps = fps * .85 + (1000 / Math.max(1, now - lastFrameT)) * .15; lastFrameT = now;
  setStatus(studio.people.length ? chip(`${studio.people.length} ${studio.people.length === 1 ? "person" : "people"} found`, "good") + (studio.kind === "video" ? chip(`${fps.toFixed(0)} fps`) : "") : chip("no person found", "bad"));
}
function renderPeople(){
  const el = $("#people"); el.innerHTML = "";
  if (!studio.people.length){ el.innerHTML = `<span class="muted" style="font-size:12.5px">No one detected yet.</span>`; return; }
  studio.people.forEach(p => { const a = assignOf(p.id); const row = document.createElement("button"); row.className = "personrow" + (p.id === studio.selected ? " on" : ""); row.style.setProperty("--c", pcol(p.id));
    row.innerHTML = `<i></i><b>Person ${p.id}</b><span>${a.char === "none" ? "unchanged" : (allChars()[a.char] || {name:"?"}).name}</span>`;
    row.onclick = () => selectPerson(p.id); el.appendChild(row); });
}
function selectPerson(id){ studio.selected = id; renderPeople(); const k = assignOf(id).char; [...$("#pick").children].forEach(c => c.classList.toggle("on", c.dataset.k === k)); $("#pickLabel").textContent = `Character for Person ${id}`; if (studio.kind === "image" || !studio.playing) redraw(); }
function redraw(){ if (studio.src) compose(octx, out.width, out.height, studio.src, studio.people.map(p => { const a = assignOf(p.id); const ch = a.char === "none" ? null : a.char; return {...p, char:ch, color:a.color || (ch ? allChars()[ch].color : null)}; }), opts()); render3D(); }
out.addEventListener("click", e => { if (!studio.people.length) return; const r = out.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  const md = $("#mode").value, side = md === "side", three = md === "three", mir = $("#mirror").checked; let best = null, bd = 1e9;
  studio.people.forEach(p => { const h = fromMP(p.L2.map(q => { const u = mir ? 1 - q[0] : q[0]; return [three ? u : u * (side ? .5 : 1), three ? (1 + 2 * q[1]) / 3 : q[1], 0]; })); const pts = [h.head, h.neck, h.pelvis, h.lsho, h.rsho, h.lhip, h.rhip];
    const d = Math.min(...pts.map(q => Math.hypot(q[0] - x, q[1] - y))); if (d < bd){ bd = d; best = p.id; } });
  if (best != null) selectPerson(best); });
async function loadSource(file, url, name){
  const isVid = file ? file.type.startsWith("video") : /\.(mp4|webm|mov)$/i.test(url);
  if (file && file.size > 300e6){ setStatus(chip("That file is over 300 MB; try a shorter clip.", "bad")); return; }
  const u = file ? URL.createObjectURL(file) : url;
  studio.frames = []; studio.assign = {}; studio.selected = 1; studio.policy = "same"; tracker.reset(); lastIds = ""; stopPlay();
  $("#stageEmpty").hidden = true; out.hidden = false; setStatus(chip("loading the pose model…", "acc"));
  try { await getEngine($("#complex").value, $("#maxp").value); } catch (e) { setStatus(chip("The pose model could not start here: " + e.message, "bad")); return; }
  if (isVid){
    const v = document.createElement("video"); v.src = u; v.muted = true; v.playsInline = true; v.loop = true;
    try { await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error("This video format is not supported by your browser.")); }); } catch (e) { setStatus(chip(e.message, "bad")); return; }
    studio.src = v; studio.kind = "video"; sizeOut(v.videoWidth, v.videoHeight); startPlay();
  } else {
    const im = new Image(); im.src = u; try { await im.decode(); } catch (e) { setStatus(chip("This image could not be read.", "bad")); return; }
    studio.src = im; studio.kind = "image"; sizeOut(im.naturalWidth, im.naturalHeight); await process();
  }
  $("#pngBtn").disabled = false; $("#playBtn").disabled = !isVid; $("#jsonBtn").disabled = !isVid; $("#recBtn").disabled = !isVid;
  $("#saveMsg").textContent = name ? "Loaded: " + name : ""; selectPerson(1);
}
function sizeOut(w, h){ const max = 1280, s = Math.min(1, max / Math.max(w, h)); const side = $("#mode").value === "side"; const three = $("#mode").value === "three"; out.width = Math.round(w * s) * (side ? 2 : 1); out.height = Math.round(h * s * (three ? 1.5 : 1)); }
let rafId = 0, busy = false;
async function tick(){ if (!studio.playing || studio.kind !== "video") return; const v = studio.src;
  if (!busy && !v.paused && v.readyState >= 2){ busy = true; try { await process(); } catch (e) {} busy = false; } rafId = requestAnimationFrame(tick); }
function startPlay(){ const v = studio.src; if (!v) return; v.play().catch(() => {}); studio.playing = true; $("#playBtn").textContent = "Pause"; cancelAnimationFrame(rafId); tick(); }
function stopPlay(){ if (studio.src && studio.kind === "video") studio.src.pause(); studio.playing = false; $("#playBtn").textContent = "Play"; cancelAnimationFrame(rafId); }
$("#playBtn").onclick = () => studio.playing ? stopPlay() : startPlay();
$("#file").onchange = e => { const f = e.target.files[0]; if (f) loadSource(f, null, f.name); };
const drop = $("#drop"); ["dragenter","dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave","drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) loadSource(f, null, f.name); });
$("#stage").addEventListener("dragover", e => e.preventDefault()); $("#stage").addEventListener("drop", e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadSource(f, null, f.name); });
[["Two people crossing (row 4)","samples/crossing.mp4"],["Dancer (row 21)","samples/dance.mp4"],["Floor exercise (row 12)","samples/floor.mp4"],["Behind a chair (row 8)","samples/chair.mp4"]].forEach(([n, u]) => { const b = document.createElement("button"); b.className = "btn"; b.textContent = n; b.onclick = () => loadSource(null, u, n); $("#samples").appendChild(b); });
$("#color").oninput = e => { assignOf(studio.selected).color = e.target.value; if (!studio.playing) redraw(); };
$("#mode").onchange = () => { if (studio.src){ const w = studio.kind === "video" ? studio.src.videoWidth : studio.src.naturalWidth, h = studio.kind === "video" ? studio.src.videoHeight : studio.src.naturalHeight; sizeOut(w, h); if (!studio.playing) redraw(); } };
["skel","mirror","labels"].forEach(id => $("#" + id).onchange = () => { if (!studio.playing) redraw(); });
$("#allSame").onclick = () => { studio.policy = "same"; const a = assignOf(studio.selected); studio.defaultChar = a.char === "none" ? studio.defaultChar : a.char; studio.people.forEach(p => { studio.assign[p.id] = {char:a.char, color:a.color}; }); renderPeople(); if (!studio.playing) redraw(); };
$("#mixChars").onclick = () => { studio.policy = "mix"; const keys = Object.keys(allChars()); studio.people.forEach((p, i) => { studio.assign[p.id] = {char:keys[i % keys.length], color:null}; }); renderPeople(); refreshPickers(); if (!studio.playing) redraw(); };
["complex","maxp"].forEach(id => $("#" + id).onchange = async () => { if (!studio.src) return; const was = studio.playing; stopPlay(); setStatus(chip("switching model…", "acc")); try { await getEngine($("#complex").value, $("#maxp").value); } catch (e) {} if (was) startPlay(); else await process(); });

/* downloads */
let downloads = null; if (window.claude && window.claude.use) window.claude.use("downloads").then(d => downloads = d).catch(() => {});
async function save(filename, data, msgEl){
  if (!downloads){ if (!window.claude){ const a = document.createElement("a"); a.href = URL.createObjectURL(data); a.download = filename; a.click(); msgEl.textContent = "Saved " + filename; } else msgEl.textContent = "Saving files is not available in this view."; return; }
  try { await downloads.save({filename, data}); msgEl.textContent = "Saved " + filename; } catch (e) { msgEl.textContent = "Not saved: " + (e && e.message ? e.message : "the download was declined"); }
}
/* Recording: MP4 (H.264) everywhere we can.
   1) WebCodecs + mp4-muxer -> a normal, seekable .mp4 (Chrome, Edge, Safari 16.4+, Firefox 130+)
   2) MediaRecorder with an MP4 type (older Safari, some Chrome builds)
   3) WebM only if the browser can do nothing else */
async function avcConfig(w, h){
  if (!window.VideoEncoder) return null;
  for (const codec of ["avc1.640033", "avc1.4d0033", "avc1.640028", "avc1.42001f"]){
    const c = {codec, width:w, height:h, bitrate:Math.round(w * h * 30 * 0.12), framerate:30, avc:{format:"avc"}};
    try { if ((await VideoEncoder.isConfigSupported(c)).supported) return c; } catch (e) {}
  }
  return null;
}
function recorder(canvas, btn, msg, getName){
  let rec = null;
  const done = () => { rec = null; btn.textContent = "Record video"; btn.classList.remove("on"); btn.disabled = false; };
  btn.onclick = async () => {
    if (rec){ btn.disabled = true; msg.textContent = "Finishing the video…"; try { const {blob, ext} = await rec.stop(); done(); save(getName() + "." + ext, blob, msg); } catch (e) { done(); msg.textContent = "Recording failed: " + e.message; } return; }
    // even size, at most 1920 wide (H.264 needs even dimensions)
    const k = Math.min(1, 1920 / canvas.width, 1920 / canvas.height), w = Math.max(2, Math.floor(canvas.width * k / 2) * 2), h = Math.max(2, Math.floor(canvas.height * k / 2) * 2);
    const cfg = window.Mp4Muxer ? await avcConfig(w, h) : null;
    if (cfg){
      const target = new Mp4Muxer.ArrayBufferTarget();
      const muxer = new Mp4Muxer.Muxer({target, video:{codec:"avc", width:w, height:h, frameRate:30}, fastStart:"in-memory", firstTimestampBehavior:"offset"});
      let err = null; const enc = new VideoEncoder({output:(chunk, meta) => muxer.addVideoChunk(chunk, meta), error:e => err = e}); enc.configure(cfg);
      const cv = document.createElement("canvas"); cv.width = w; cv.height = h; const cx = cv.getContext("2d");
      const t0 = performance.now(); let n = 0, last = -1, stopped = false;
      const tick = () => { if (stopped) return; const t = performance.now() - t0;
        if (t - last >= 1000 / 30 - 1 && enc.encodeQueueSize < 8){ last = t; cx.drawImage(canvas, 0, 0, w, h);
          const f = new VideoFrame(cv, {timestamp:Math.round(t * 1000), duration:Math.round(1e6 / 30)}); enc.encode(f, {keyFrame:n % 60 === 0}); f.close(); n++; }
        requestAnimationFrame(tick); };
      tick();
      rec = {stop: async () => { stopped = true; await enc.flush(); enc.close(); if (err) throw err; if (!n) throw new Error("no frames were captured"); muxer.finalize(); return {blob:new Blob([target.buffer], {type:"video/mp4"}), ext:"mp4"}; }};
    } else {
      if (!canvas.captureStream || !window.MediaRecorder){ msg.textContent = "This browser cannot record video. Try Chrome, Edge or Safari."; return; }
      const type = ["video/mp4;codecs=avc1.42E01E", "video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"].find(t => MediaRecorder.isTypeSupported(t)) || "";
      const mr = new MediaRecorder(canvas.captureStream(30), type ? {mimeType:type, videoBitsPerSecond:8e6} : {}), chunks = [];
      mr.ondataavailable = e => e.data.size && chunks.push(e.data);
      const isMp4 = (mr.mimeType || type).includes("mp4");
      rec = {stop: () => new Promise(r => { mr.onstop = () => r({blob:new Blob(chunks, {type:isMp4 ? "video/mp4" : "video/webm"}), ext:isMp4 ? "mp4" : "webm"}); mr.stop(); })};
      mr.start(1000);
      if (!isMp4) msg.textContent = "This browser can only record WebM. Use Chrome, Edge or Safari for MP4.";
    }
    btn.textContent = "Stop recording"; btn.classList.add("on"); if (!msg.textContent.startsWith("This browser")) msg.textContent = "Recording (MP4)…";
  };
}
recorder(out, $("#recBtn"), $("#saveMsg"), () => `rw2b_${Date.now()}`);
$("#pngBtn").onclick = () => out.toBlob(b => save(`rw2b_frame_${Date.now()}.png`, b, $("#saveMsg")), "image/png");
$("#jsonBtn").onclick = () => {
  const fr = [...new Map(studio.frames.map(f => [f.t, f])).values()].sort((a, b) => a.t - b.t);
  if (!fr.length){ $("#saveMsg").textContent = "Play the video first so there are frames to export."; return; }
  const data = {format:"rw2b-mediapipe-33-multi", units:"metres, hip-centred per person, y down", fps_hint: Math.round(fr.length / Math.max(0.1, fr[fr.length-1].t - fr[0].t)), frames:fr};
  save(`rw2b_motion_${Date.now()}.json`, new Blob([JSON.stringify(data)], {type:"application/json"}), $("#saveMsg"));
};
$("#bpy").textContent = `# Blender 4.x: Scripting tab -> New -> paste -> set PATH -> Run Script
import bpy, json
PATH = "/path/to/rw2b_motion.json"     # the file from "Export motion (JSON)"
SPACING = 1.0                           # metres between people (positions are hip-centred per person)
d = json.load(open(PATH))
names = ["nose","l_eye_in","l_eye","l_eye_out","r_eye_in","r_eye","r_eye_out","l_ear","r_ear","mouth_l","mouth_r",
         "l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist","l_pinky","r_pinky","l_index","r_index",
         "l_thumb","r_thumb","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle","l_heel","r_heel","l_foot","r_foot"]
fps = d.get("fps_hint", 30); bpy.context.scene.render.fps = int(fps)
rigs = {}
def rig(pid):
    if pid not in rigs:
        root = bpy.data.objects.new(f"rw2b_person_{pid}", None); bpy.context.collection.objects.link(root)
        root.location.x = (pid - 1) * SPACING
        es = []
        for n in names:
            e = bpy.data.objects.new(f"p{pid}_{n}", None); e.empty_display_size = 0.03; e.parent = root
            bpy.context.collection.objects.link(e); es.append(e)
        rigs[pid] = es
    return rigs[pid]
t0 = d["frames"][0]["t"]; frame = 1
for f in d["frames"]:
    frame = 1 + round((f["t"] - t0) * fps)
    for p in f["people"]:
        for e, (x, y, z) in zip(rig(p["id"]), p["world"]):
            e.location = (x, -z, -y)            # MediaPipe (y down, z toward camera) -> Blender (z up)
            e.keyframe_insert("location", frame=frame)
bpy.context.scene.frame_end = frame
print("rw2b: animated", len(rigs), "people over", len(d["frames"]), "frames")`;
$("#copyScript").onclick = async () => { try { await navigator.clipboard.writeText($("#bpy").textContent); $("#copyMsg").textContent = "Copied."; } catch (e) { $("#copyMsg").textContent = "Copy was blocked; select the text and copy it."; } };

/* =============================== live =============================== */
const live = {char:"robot", color:CHARACTERS.robot.color, stream:null, video:null, running:false};
const lout = $("#liveOut"), lctx = lout.getContext("2d"), ltracker = PersonTracker();
$("#liveEmpty").textContent = "Press Start camera. Your video never leaves this device.";
{ const rail = document.querySelector(".rail"), mb = $("#menuBtn"); if (mb){ mb.onclick = () => { const o = rail.classList.toggle("open"); mb.setAttribute("aria-expanded", o); mb.textContent = o ? "Close" : "Menu"; };
  $("#nav").addEventListener("click", e => { if (e.target.closest("a")){ rail.classList.remove("open"); mb.setAttribute("aria-expanded", false); mb.textContent = "Menu"; } }); } }
(() => { const el = $("#qr"); if (!el || !window.qrcode) return; try { const u = /github\.io|localhost|127\.0\.0\.1/.test(location.host) && !/localhost|127\.0\.0\.1/.test(location.host) ? location.origin + location.pathname + "#live" : "https://vbobbadi.github.io/rw2b-studio/#live";
  const q = qrcode(0, "M"); q.addData(u); q.make(); el.innerHTML = q.createSvgTag({cellSize:4, margin:2, scalable:true}); el.title = u; } catch (e) {} })();
$("#liveColor").value = live.color; $("#liveColor").oninput = e => live.color = e.target.value;
live.assign = {}; live.seen = "";
const liveCharOf = id => { if (!live.assign[id]) { const keys = Object.keys(allChars()); live.assign[id] = $("#liveMix").checked ? keys[(id - 1) % keys.length] : live.char; } return live.assign[id]; };
live.last = {};
function renderLivePeople(now){
  const el0 = $("#livePeople"); if (el0.contains(document.activeElement) && document.activeElement.tagName === "SELECT") return;
  const people = Object.keys(live.last).filter(id => now - live.last[id] < 2000).map(id => ({id:+id})).sort((a, b) => a.id - b.id);
  const sig = people.map(p => p.id + ":" + live.assign[p.id]).join(","); if (sig === live.seen) return; live.seen = sig;
  const el = $("#livePeople");
  if (!people.length){ el.innerHTML = '<p class="note" style="margin:0">Nobody in view yet.</p>'; return; }
  const opts = [["none", "Leave as is"], ...Object.entries(allChars()).map(([k, c]) => [k, c.name])];
  el.innerHTML = people.map(p => `<label class="personrow" style="--c:${pcol(p.id)};cursor:default"><i></i><b>Person ${p.id}</b><select data-pid="${p.id}" aria-label="Character for person ${p.id}">${opts.map(([k, n]) => `<option value="${k}" ${k === live.assign[p.id] ? "selected" : ""}>${n}</option>`).join("")}</select></label>`).join("");
  el.querySelectorAll("select").forEach(s => s.onchange = () => { live.assign[+s.dataset.pid] = s.value; live.seen = ""; });
}
async function liveLoop(){ if (!live.running) return;
  try { const raw = (await detect(live.video, true)).slice(0, +$("#liveMax").value);
    const people = ltracker.update(raw).map(p => { const ch = liveCharOf(p.id), c = ch === "none" ? null : ch; return {id:p.id, mask:p.mask, L2:p.t.sm(p.lm, 0.5), W3:p.t.sm3(p.world, 0.5), char:c, color:c ? (c === live.char && !$("#liveMix").checked ? live.color : allChars()[c].color) : null}; });
    compose(lctx, lout.width, lout.height, live.video, people, {mode:$("#liveMode").value, mirror:live.facing === "user", labels:people.length > 1});
    const now = performance.now(); people.forEach(p => live.last[p.id] = now); renderLivePeople(now);
    $("#liveStatus").innerHTML = people.length ? chip(people.length > 1 ? `${people.length} people` : "tracking you", "good") : chip("step back so your body is in view", "warn"); } catch (e) {}
  requestAnimationFrame(liveLoop); }
function sizeLive(){ if (!live.vw) return; const three = $("#liveMode").value === "three"; lout.width = live.vw; lout.height = Math.round(live.vh * (three ? 1.5 : 1)); }
$("#liveMode").onchange = sizeLive;
$("#liveMix").onchange = () => { live.assign = {}; live.seen = ""; };
live.facing = "user";
const CAM_ERR = {NotAllowedError:"Camera permission was blocked. Allow the camera for this site in your browser settings and try again.",
  NotFoundError:"No camera was found on this device.", NotReadableError:"The camera is busy. Close other apps or tabs that use it and try again.",
  OverconstrainedError:"This camera does not support the requested settings.", SecurityError:"This page is not allowed to use the camera here. Open https://vbobbadi.github.io/rw2b-studio/#live in Safari or Chrome."};
function stopCam(){ live.running = false; if (live.stream) live.stream.getTracks().forEach(t => t.stop()); live.stream = null;
  if (live.video){ live.video.srcObject = null; live.video.remove(); live.video = null; } }
async function openCam(){
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){ const e = new Error(window.isSecureContext ? "This browser has no camera access." : "The camera needs a secure (https) page."); e.name = window.isSecureContext ? "NotFoundError" : "SecurityError"; throw e; }
  try { return await navigator.mediaDevices.getUserMedia({video:{facingMode:live.facing, width:{ideal:1280}, height:{ideal:720}}, audio:false}); }
  catch (e) { if (e.name === "OverconstrainedError" || e.name === "TypeError") return await navigator.mediaDevices.getUserMedia({video:true, audio:false}); throw e; }
}
async function startCam(){
  $("#liveMsg").textContent = ""; $("#liveEmpty").hidden = false; $("#liveEmpty").textContent = "Starting the camera…";
  try { live.stream = await openCam(); }
  catch (e) { $("#liveEmpty").textContent = CAM_ERR[e.name] || ("The camera could not start: " + (e.message || e.name) + "."); return false; }
  stopPlay();
  // iOS Safari only feeds frames to a video that is inline, muted and in the page
  const v = document.createElement("video"); v.setAttribute("playsinline", ""); v.setAttribute("muted", ""); v.setAttribute("autoplay", ""); v.muted = true; v.playsInline = true;
  v.style.cssText = "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none"; document.body.appendChild(v);
  v.srcObject = live.stream;
  await new Promise(r => { if (v.readyState >= 1) r(); else v.onloadedmetadata = r; });
  try { await v.play(); } catch (e) {}
  for (let i = 0; i < 40 && !v.videoWidth; i++) await new Promise(r => setTimeout(r, 50));
  live.video = v; live.vw = v.videoWidth || 640; live.vh = v.videoHeight || 480; sizeLive(); lout.hidden = false; $("#liveEmpty").hidden = true;
  const tr = live.stream.getVideoTracks()[0], fm = tr && tr.getSettings ? tr.getSettings().facingMode : null; if (fm) live.facing = fm === "environment" ? "environment" : "user";
  try { const devs = await navigator.mediaDevices.enumerateDevices(); $("#camFlip").hidden = devs.filter(d => d.kind === "videoinput").length < 2; } catch (e) {}
  setStatus2(chip("loading the pose model…", "acc"));
  try { await getEngine("lite", 5); } catch (e) { $("#liveMsg").textContent = "The pose model could not start: " + e.message; stopCam(); return false; }
  live.running = true; live.assign = {}; live.last = {}; live.seen = ""; $("#camBtn").textContent = "Stop camera"; $("#liveRec").disabled = false; ltracker.reset(); liveLoop(); return true;
}
const setStatus2 = html => $("#liveStatus").innerHTML = html;
$("#camBtn").onclick = async () => {
  if (live.running || live.stream){ stopCam(); $("#camBtn").textContent = "Start camera"; $("#liveRec").disabled = true; $("#camFlip").hidden = true; lout.hidden = true; $("#liveEmpty").hidden = false; $("#liveEmpty").textContent = "Camera stopped."; setStatus2(""); return; }
  $("#camBtn").disabled = true; await startCam(); $("#camBtn").disabled = false;
};
$("#camFlip").onclick = async () => { live.facing = live.facing === "user" ? "environment" : "user"; stopCam(); await startCam(); };
recorder(lout, $("#liveRec"), $("#liveMsg"), () => `rw2b_live_${Date.now()}`);

/* ============================== library ============================= */
const Library = (() => {
  let SK = null, raf = 0, cards = [], f = 0, lastT = 0;
  const ready = fetch("skeletons.json").then(r => r.json()).then(d => { SK = d.dance.models.dgs; }).catch(() => {});
  function pts(frame){ return SK[frame % SK.length].map(p => [p[0] / 1000, p[1] / 1000, p[2] / 1000]); }
  function bodyAt(stage, frame, W, H){ const P = pts(frame); const sc = Math.min(W, H) / 2.2; return fromSMPL(P.map(p => stage ? stage.project(p, sc, 0.47) : [W / 2 + p[0] * sc, H * 0.47 + p[1] * sc, -p[2]])); }
  function sampleBody(){ if (!SK) return null; return (W, H) => ({B:bodyAt(null, 40, W, H), P3:fromSMPL(pts(40))}); }
  function build(){
    const lib = $("#lib"); lib.innerHTML = ""; cards = [];
    Object.entries(allChars()).forEach(([k, ch]) => { const d = document.createElement("article"); d.className = "libcard";
      d.innerHTML = `<canvas style="cursor:grab;touch-action:none" aria-label="${ch.name} preview, drag to turn"></canvas><div class="body"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px"><h3>${ch.name}</h3>${k === "custom" ? '<span class="chip acc">yours</span>' : ch.rig ? `<span class="chip acc">${ch.builtin ? "3D" : "3D · yours"}</span>` : ""}</div><p>${ch.blurb}</p>${ch.credit ? `<p class="muted" style="font-size:11.5px;margin:0">${ch.credit}</p>` : ""}<div class="row-ctrl"><a class="btn" href="#studio" data-use="${k}">Use in studio</a><a class="btn" href="#live" data-use="${k}">Use live</a></div></div>`;
      lib.appendChild(d); const stg = Stage(d.querySelector("canvas"), {yaw:-0.6 + cards.length * 0.25, pitch:0.12}); cards.push([k, stg]); });
    lib.querySelectorAll("[data-use]").forEach(a => a.addEventListener("click", () => { const k = a.dataset.use; if (a.getAttribute("href") === "#live") live.char = k; else { studio.defaultChar = k; assignOf(studio.selected).char = k; } refreshPickers(); }));
  }
  function frame(t){ if (!SK){ raf = requestAnimationFrame(frame); return; } if (t - lastT > 66){ f++; lastT = t; }
    cards.forEach(([k, stg]) => { if (!stg.canvas.clientWidth) return; if (stg.canvas.width !== Math.round(stg.canvas.clientWidth * (devicePixelRatio || 1))) stg.fit(); stg.clear(); const W = stg.canvas.clientWidth, H = stg.canvas.clientHeight; stg.grid(Math.min(W, H) / 2.2, 0.95, 4, 0.25); drawCharacter(stg.ctx, k, bodyAt(stg, f, W, H), {P3:fromSMPL(pts(f)), view:{yaw:stg.yaw, pitch:stg.pitch}}); });
    const cp = $("#cPrev"); if (cp && cp.clientWidth){ const stg = cPrevStage; if (cp.width !== Math.round(cp.clientWidth * (devicePixelRatio || 1))) stg.fit(); stg.clear(); const W = cp.clientWidth, H = cp.clientHeight; stg.grid(Math.min(W, H) / 2.2, 0.95, 4, 0.25);
      const B = bodyAt(stg, f, W, H); CHARACTERS[$("#cBase").value].draw(stg.ctx, B, {color:$("#cColor").value}); if (pendingImg && pendingImg.complete) drawHeadImage(stg.ctx, B, pendingImg); }
    raf = requestAnimationFrame(frame); }
  return {ready, sampleBody, build, start(){ ready.then(() => { if (!cards.length) build(); cancelAnimationFrame(raf); raf = requestAnimationFrame(frame); }); }, stop(){ cancelAnimationFrame(raf); }};
})();
const cPrevStage = Stage($("#cPrev"), {yaw:-0.4, pitch:0.1});
let pendingImg = CUSTOM ? CUSTOM.imgEl : null;
Object.entries(CHARACTERS).forEach(([k, c]) => { const o = document.createElement("option"); o.value = k; o.textContent = c.name; $("#cBase").appendChild(o); });
if (CUSTOM){ $("#cName").value = CUSTOM.name; $("#cBase").value = CUSTOM.base; $("#cColor").value = CUSTOM.color; $("#cSave").disabled = false; }
function readHead(file){ if (!file || !file.type.startsWith("image/")) return $("#cMsg").textContent = "Choose an image file."; if (file.size > 2e6) return $("#cMsg").textContent = "That image is over 2 MB.";
  const fr = new FileReader(); fr.onload = () => { const im = new Image(); im.onload = () => { // shrink to 256 px so it fits in browser storage
      const c = document.createElement("canvas"), s = 256 / Math.max(im.width, im.height); c.width = im.width * s; c.height = im.height * s; c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
      const small = new Image(); small.src = c.toDataURL("image/png"); pendingImg = small; $("#cSave").disabled = false; $("#cMsg").textContent = "Looks good. Add it to the library."; }; im.src = fr.result; }; fr.readAsDataURL(file); }
$("#cFile").onchange = e => readHead(e.target.files[0]);
const cd = $("#cDrop"); ["dragenter","dragover"].forEach(t => cd.addEventListener(t, e => { e.preventDefault(); cd.classList.add("over"); })); ["dragleave","drop"].forEach(t => cd.addEventListener(t, e => { e.preventDefault(); cd.classList.remove("over"); }));
cd.addEventListener("drop", e => readHead(e.dataTransfer.files[0]));
$("#cSave").onclick = () => { if (!pendingImg) return; CUSTOM = {name:$("#cName").value.trim() || "My character", base:$("#cBase").value, color:$("#cColor").value, img:pendingImg.src, imgEl:pendingImg};
  try { localStorage.setItem("rw2b_custom", JSON.stringify({name:CUSTOM.name, base:CUSTOM.base, color:CUSTOM.color, img:CUSTOM.img})); $("#cMsg").textContent = "Added. It is now in every character picker."; } catch (e) { $("#cMsg").textContent = "Added for this visit (this browser would not store it)."; }
  $("#cDel").hidden = false; Library.build(); refreshPickers(); };
$("#cDel").onclick = () => { CUSTOM = null; try { localStorage.removeItem("rw2b_custom"); } catch (e) {} Object.values(studio.assign).forEach(a => { if (a.char === "custom") a.char = "neon"; }); if (live.char === "custom") live.char = "robot"; $("#cDel").hidden = true; $("#cMsg").textContent = "Removed."; Library.build(); refreshPickers(); };

function renderRigList(){ const el = $("#rigList"); if (!el || !RIG) return; el.innerHTML = "";
  RIG.list().filter(r => !r.builtin).forEach(r => { const row = document.createElement("div"); row.className = "row-ctrl"; row.style.justifyContent = "space-between";
    row.innerHTML = `<span><b>${r.name.replace(/[<>&]/g, "")}</b> <span class="muted" style="font-size:12px">${r.bones} bones</span></span>`;
    const x = document.createElement("button"); x.className = "btn"; x.textContent = "Remove"; x.onclick = async () => { await RIG.remove(r.id); Object.values(studio.assign).forEach(a => { if (a.char === r.id) a.char = "neon"; }); if (live.char === r.id) live.char = "robot"; renderRigList(); Library.build(); refreshPickers(); };
    row.appendChild(x); el.appendChild(row); }); }
async function addRig(file){ const msg = $("#rigMsg"); if (!RIG){ msg.textContent = "3D characters could not start in this browser (WebGL is needed)."; return; }
  if (!file) return; if (file.size > 60e6){ msg.textContent = "That file is over 60 MB."; return; }
  msg.textContent = "Loading " + file.name + "…";
  try { const r = await RIG.add(file); msg.textContent = `Added ${r.name}.` + (r.kept ? " It will still be here next time you open this page in this browser." : " This browser would not store it, so it lasts for this visit only.");
    studio.defaultChar = r.id; assignOf(studio.selected).char = r.id; renderRigList(); Library.build(); refreshPickers(); }
  catch (e) { msg.textContent = "Could not use that file: " + (e && e.message ? e.message : e); } }
if ($("#rigFile")){ $("#rigFile").onchange = e => addRig(e.target.files[0]); const rd = $("#rigDrop");
  ["dragenter","dragover"].forEach(t => rd.addEventListener(t, e => { e.preventDefault(); rd.classList.add("over"); })); ["dragleave","drop"].forEach(t => rd.addEventListener(t, e => { e.preventDefault(); rd.classList.remove("over"); }));
  rd.addEventListener("drop", e => addRig(e.dataTransfer.files[0])); }
function refreshPickers(){
  buildPicker($("#pick"), () => assignOf(studio.selected).char, k => { const a = assignOf(studio.selected); a.char = k; a.color = null; a.auto = false; Object.keys(studio.assign).forEach(id => { if (studio.assign[id].auto && !studio.people.some(p => p.id === +id)) delete studio.assign[id]; }); if (k !== "none") $("#color").value = allChars()[k].color; if (k !== "none") studio.defaultChar = k; renderPeople(); if (!studio.playing) redraw(); }, true);
  buildPicker($("#livePick"), () => live.char, k => { live.char = k; live.color = allChars()[k].color; $("#liveColor").value = live.color; if (!$("#liveMix").checked){ for (const id in live.assign) live.assign[id] = k; live.seen = ""; } });
}
Library.ready.then(refreshPickers); refreshPickers();
const BUILTIN_RIGS = [
  {id:"b_xbot", name:"X Bot", url:"models/xbot.glb", blurb:"Mixamo's classic training dummy, fully rigged in 3D.", credit:"Mixamo (Adobe), via the three.js examples."},
  {id:"b_michelle", name:"Michelle", url:"models/michelle.glb", blurb:"A textured Mixamo character in streetwear.", credit:"Mixamo (Adobe), via the three.js examples."},
  {id:"b_soldier", name:"Soldier", url:"models/soldier.glb", blurb:"A Mixamo soldier in combat gear.", credit:"Mixamo (Adobe), via the three.js examples."},
  {id:"b_robot", name:"Expressive Robot", url:"models/robot.glb", blurb:"A friendly low-poly robot.", credit:"Tomás Laulhé (Quaternius), CC0, via the three.js examples."},
  {id:"b_samba", name:"Samba Dancer", url:"models/samba.fbx", blurb:"A slim Mixamo dancer figure, fully rigged in 3D.", credit:"Mixamo (Adobe), via the three.js examples."}];
renderRigList();
if (RIG){ RIG.restore().then(l => { if (l.length){ Library.build(); refreshPickers(); renderRigList(); } });
  RIG.builtin(BUILTIN_RIGS, () => { Library.build(); refreshPickers(); }); }
route();
})();
