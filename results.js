(() => {
const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C = {hmr:css("--hmr"), dgs:css("--accent"), gen:css("--gen"), sg:css("--sg"), good:css("--good"), warn:css("--warn"), bad:css("--bad"), line:css("--line"), muted:css("--muted"), ink:css("--ink")};
const tip = $("#tip");
function showTip(e, html){ tip.innerHTML = html; tip.style.opacity = 1; const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8); tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px"; }
function hideTip(){ tip.style.opacity = 0; }
function el(tag, attrs = {}, parent){ const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; }
function txt(parent, x, y, s, attrs = {}){ const t = el("text", {x, y, ...attrs}, parent); t.textContent = s; return t; }
function clear(svg){ while (svg.firstChild) svg.removeChild(svg.firstChild); }
function widthOf(svg){ return Math.max(320, svg.parentElement.clientWidth); }
function hov(node, html){ node.addEventListener("mousemove", e => showTip(e, html)); node.addEventListener("mouseleave", hideTip); }
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

/* ---------- synthetic results (notebooks 06, 07, 08, 10) ---------- */
const SYN = {
  selfocc: [{m:"Released",v:[51.6,96.9,75.1]},{m:"B clean",v:[44.4,92.4,67.8]},{m:"C 2D-pasted",v:[42.1,87.6,65.1]},{m:"D 3D-rendered",v:[40.2,85.1,63.4]}],
  rq2gaps: [{l:"D − C, on 3D-occ test",d:-2.1,lo:-2.6,hi:-1.6},{l:"C − D, on 2D-pasted test",d:-2.3},{l:"D − C, on clean test",d:3.1},{l:"D − B, on 3D-occ test",d:-5.4}],
  conds: ["clean","3D occ","2D occ","quality","framing","occ+quality"],
  heat: [
    ["Released",[46.8,73.1,61.5,52.2,63.6,80.3]],["E0 clean",[32.2,66.9,53.0,39.9,53.8,75.7]],["E1 occlusion",[38.1,62.1,53.5,44.8,53.4,71.0]],
    ["E2 quality",[33.8,68.8,53.8,39.7,54.8,77.3]],["E3 framing",[35.4,66.4,53.4,42.2,46.7,74.5]],["GEN",[28.4,58.6,48.3,35.6,41.7,68.1]],
    ["DGS oracle",[32.2,62.1,53.5,39.7,46.7,72.3]],["DGS learned",[32.3,62.1,51.6,39.6,46.4,71.4]]],
  fusion: [["Learned",50.6],["Weighted",51.0],["Per-joint",51.1],["Top-1",51.1],["Oracle routing",51.1]],
  rq3: {tests:["3D occ","2D occ","clean"], s:[["Random",[63.7,55.1,39.9],[0.4,0.2,0.5]],["Bandit (Hedge)",[64.8,55.8,41.2],[0.3,0.3,0.5]],["Hard-example mining",[66.8,58.0,43.0],[0.3,0.5,0.5]]]},
  aucSyn: {occlusion:0.9999, quality:0.9998, framing:0.9998}
};
const NAME = {hmr2:"Released HMR2.0", B_clean:"B clean", C_paste2d:"C 2D-pasted", D_occ3d:"D 3D-rendered", E0:"E0 clean", E1:"E1 occlusion", E2:"E2 quality", E3:"E3 framing", GEN:"GEN generalist", dgs_weighted:"DGS weighted", dgs_perjoint:"DGS per-joint", dgs_learned:"DGS learned", dgs_top1:"DGS top-1"};
const COL = m => m === "hmr2" ? C.hmr : m === "GEN" ? C.gen : m.startsWith("dgs") ? C.dgs : "#b8bec9";

Promise.all([fetch("data.json").then(r => r.json()), fetch("skeletons.json").then(r => r.json())]).then(([D, SK]) => {
  window.__D = D;
  buildViewer(SK, D);
  buildFindings(D);
  draw3dpw(D); drawSeqTable(D); drawOcc(D);
  buildClips(D);
  drawSelfOcc(); drawGaps(); drawHeat(); drawFusion(); drawRq4Real(D); drawRq3(); drawAuc(D);
  let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { draw3dpw(D); drawOcc(D); drawSelfOcc(); drawGaps(); drawHeat(); drawFusion(); drawRq4Real(D); drawRq3(); drawAuc(D); }, 120); });
}).catch(err => { $("#findGrid").textContent = "Could not load the result files: " + err; });

/* ================= 3D viewer ================= */
function buildViewer(SK, D){
  const PAR = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];
  const CLIPS = {
    dance:{title:"Fast motion: dance", row:"row 2", img:"img/row02.jpg", clip:"row02_dance_01", note:"Full body, fast spins. Both models track her; raw jitter is about equal (14.3 vs 14.4 mm)."},
    stretch:{title:"Self-occlusion: seated stretch", row:"row 10", img:"img/row10b.jpg", clip:"row10_stretch_02", note:"Head and arms fold over the legs. Our model's clearest win: raw jitter drops from 14.1 to 10.6 mm."},
    floor:{title:"Lying on the floor", row:"row 12", img:"img/row12.jpg", clip:"row12_floor_01", note:"Unusual viewpoint from above. Both models are shaky here (26.1 vs 25.3 mm)."},
    crossing:{title:"Multi-person: crossing", row:"row 4", img:"img/row04.jpg", clip:"row04_crossing_01", note:"Five tracked pedestrians, each its own color, placed in depth from their box size. People appear and vanish where the tracker starts and drops them; identity swaps when they cross are the known tracker weakness."},
    twopeople:{title:"Multi-person: two people", row:"row 3", img:"img/row03.jpg", clip:"row03_twoapart_01", note:"Two people talking, both tracked for the whole clip. Only their upper bodies are in view, so the legs are the model's guess."}
  };
  const PCOL = ["#ff8a3d","#7fb0e8","#c3a6ff","#5fd3b5","#f2c14e","#ff6b9a","#9ad15a"];
  const MODELS = [["hmr2","HMR2.0",C.hmr],["dgs","DGS",C.dgs],["dgs_sg","DGS + smoothing",C.sg]];
  const shown = {hmr2:true, dgs:true, dgs_sg:false};
  let clip = "stretch", f = 0, playing = !matchMedia("(prefers-reduced-motion: reduce)").matches, yaw = -0.5, pitch = 0.18, last = 0, acc = 0;
  const cv = $("#cv"), ctx = cv.getContext("2d");
  const clipSeg = $("#clipSeg"), modelSeg = $("#modelSeg");
  Object.entries(CLIPS).forEach(([k, c]) => { const b = document.createElement("button"); b.className = "btn"; b.textContent = c.title.startsWith("Multi") ? c.title.split(": ")[1].replace(/^./, m => m.toUpperCase()) : c.title.split(":")[0]; b.dataset.k = k; b.onclick = () => setClip(k); clipSeg.appendChild(b); });
  MODELS.forEach(([k, label, col]) => { const b = document.createElement("button"); b.className = "btn"; b.innerHTML = `<span style="display:inline-block;width:10px;height:3px;border-radius:2px;background:${col};margin-right:6px;vertical-align:middle"></span>${label}`; b.dataset.k = k; b.onclick = () => { shown[k] = !shown[k]; sync(); render(); }; modelSeg.appendChild(b); });
  function sync(){ [...clipSeg.children].forEach(b => b.classList.toggle("on", b.dataset.k === clip)); const multi = !!SK[clip].multi; modelSeg.style.display = multi ? "none" : ""; $("#multiNote").style.display = multi ? "" : "none"; [...modelSeg.children].forEach(b => { if (multi) return; const has = !!SK[clip].models[b.dataset.k]; b.disabled = !has; b.style.opacity = has ? 1 : .35; b.classList.toggle("on", has && shown[b.dataset.k]); }); $("#play").textContent = playing ? "Pause" : "Play"; }
  function setClip(k){
    clip = k; f = 0; const c = CLIPS[k], S = SK[k];
    $("#sheet").src = c.img; $("#clipTitle").textContent = c.title; $("#clipRow").textContent = c.row; $("#clipNote").textContent = c.note;
    $("#scrub").max = S.frames.length - 1; $("#scrub").value = 0;
    const row = D.clips.find(r => r.clip === c.clip);
    const st = $("#heroStats"); st.innerHTML = "";
    const add = (v, l, col) => { const d = document.createElement("div"); d.className = "stat"; d.innerHTML = `<b style="color:${col}">${v}</b><span>${l}</span>`; st.appendChild(d); };
    if (S.multi){ const per = S.frames.map((_, i) => S.people.filter(p => p.J[i]).length); add(S.people.length, "people tracked (HMR2.0, one track each)", C.dgs); add(Math.max(...per), "most people in one frame", C.ink); add(row ? row.jit_hmr2.toFixed(1) : "—", "main person raw jitter (mm)", C.hmr); add(row ? Math.round(row.det * 100) + "%" : "—", "frames with the main person", C.ink); }
    else if (row){ add(row.jit_hmr2.toFixed(1), "HMR2.0 raw jitter (mm)", C.hmr); add(row.jit_dgs.toFixed(1), "DGS raw jitter (mm)", C.dgs); add(row.jit_sg != null ? row.jit_sg.toFixed(1) : "—", "DGS + smoothing (mm)", C.sg); add(Math.round(row.det * 100) + "%", "frames with the person tracked", C.ink); }
    $("#sparkLabel").textContent = S.multi ? "People tracked in each frame." : "Joint speed per frame (mm/frame). Spikes are jitter.";
    if (S.multi){ yaw = -0.2; pitch = 0.12; } else { yaw = -0.5; pitch = 0.18; }
    drawSpark(); sync(); render();
  }
  function speeds(J){ const out = [0]; for (let t = 1; t < J.length; t++){ let s = 0; for (let j = 0; j < 22; j++){ const a = J[t][j], b = J[t-1][j]; s += Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]); } out.push(s / 22); } return out; }
  function drawSpark(){
    const svg = $("#spark"); clear(svg); const W = svg.clientWidth || 400, H = 46; svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const S = SK[clip]; let series;
    if (S.multi){ series = [["count", S.frames.map((_, i) => S.people.filter(p => p.J[i]).length), C.dgs]]; shown.count = true; }
    else series = MODELS.filter(([k]) => S.models[k]).map(([k,,col]) => [k, speeds(S.models[k]), col]);
    const mx = Math.max(...series.flatMap(s => s[1])) || 1; const n = S.frames.length;
    series.forEach(([k, v, col]) => { if (!shown[k]) return; let d = ""; v.forEach((y, i) => { d += (i ? "L" : "M") + (i / (n - 1) * W).toFixed(1) + "," + (H - 3 - y / mx * (H - 8)).toFixed(1); }); el("path", {d, fill:"none", stroke:col, "stroke-width":1.4, opacity:.9}, svg); });
    el("line", {id:"sparkCursor", x1:0, x2:0, y1:0, y2:H, stroke:C.ink, "stroke-width":1, opacity:.6}, svg);
  }
  function resize(){ const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1; cv.width = r.width * dpr; cv.height = r.height * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); render(); }
  function project(p, W, H, sc){ // camera coords: x right, y down, z forward -> world: X=x, Y=-y, Z=-z
    let X = p[0], Y = -p[1], Z = -p[2];
    const cy = Math.cos(yaw), sy = Math.sin(yaw); [X, Z] = [X*cy + Z*sy, -X*sy + Z*cy];
    const cp = Math.cos(pitch), sp = Math.sin(pitch); [Y, Z] = [Y*cp - Z*sp, Y*sp + Z*cp];
    return [W/2 + X*sc, H*(SK[clip].multi ? 0.42 : 0.52) - Y*sc, Z];
  }
  function render(){
    const W = cv.clientWidth, H = cv.clientHeight; ctx.clearRect(0, 0, W, H);
    const S = SK[clip]; const sc = S.multi ? Math.min(W, H) / (2 * S.extent) * 1.7 : Math.min(W, H) / 1500;
    // floor grid
    ctx.strokeStyle = "#2a2f39"; ctx.lineWidth = 1;
    const G = S.multi ? Math.ceil(S.extent / 1000) * 1000 : 1000, gs = G / 5;
    for (let i = -5; i <= 5; i++){ const a = project([i*gs, 950, -G], W, H, sc), b = project([i*gs, 950, G], W, H, sc), c = project([-G, 950, i*gs], W, H, sc), d = project([G, 950, i*gs], W, H, sc); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke(); }
    const segs = [];
    if (S.multi) S.people.forEach((p, pi) => { const J = p.J[f]; if (!J) return; const o = p.P[f], col = PCOL[pi % PCOL.length];
      const P = J.map(q => project([q[0] + o[0], q[1], q[2] + o[1]], W, H, sc));
      for (let j = 1; j < 22; j++) segs.push({a:P[PAR[j]], b:P[j], col, z:(P[j][2] + P[PAR[j]][2]) / 2});
      segs.push({head:P[15], col, z:P[15][2]}); });
    else MODELS.forEach(([k,, col], mi) => { if (!shown[k] || !S.models[k]) return; const J = S.models[k][f]; const P = J.map(p => project(p, W, H, sc));
      for (let j = 1; j < 22; j++) segs.push({a:P[PAR[j]], b:P[j], col, z:(P[j][2] + P[PAR[j]][2]) / 2});
      segs.push({head:P[15], col, z:P[15][2]}); });
    segs.sort((a, b) => a.z - b.z);
    segs.forEach(s => { ctx.strokeStyle = s.col; ctx.fillStyle = s.col; if (s.head){ ctx.globalAlpha = .9; ctx.beginPath(); ctx.arc(s.head[0], s.head[1], 7, 0, 7); ctx.fill(); return; }
      ctx.globalAlpha = .92; ctx.lineWidth = 4; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(s.a[0], s.a[1]); ctx.lineTo(s.b[0], s.b[1]); ctx.stroke(); });
    ctx.globalAlpha = 1;
    const n = S.frames.length; $("#hud").textContent = `frame ${S.frames[f]} · ${f + 1}/${n} · ${S.fps.toFixed(1)} fps · drag to rotate`;
    $("#scrub").value = f; const cur = document.getElementById("sparkCursor"); if (cur){ const x = (f / (n - 1)) * ($("#spark").clientWidth || 400); cur.setAttribute("x1", x); cur.setAttribute("x2", x); }
  }
  function loop(ts){ if (playing){ acc += ts - last; const step = 1000 / SK[clip].fps; while (acc > step){ f = (f + 1) % SK[clip].frames.length; acc -= step; } render(); } last = ts; requestAnimationFrame(loop); }
  let drag = null;
  cv.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, yaw, pitch]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => { if (!drag) return; yaw = drag[2] + (e.clientX - drag[0]) * 0.01; pitch = Math.max(-1.2, Math.min(1.2, drag[3] + (e.clientY - drag[1]) * 0.01)); render(); });
  cv.addEventListener("pointerup", () => drag = null);
  $("#play").onclick = () => { playing = !playing; sync(); };
  $("#scrub").oninput = e => { f = +e.target.value; playing = false; sync(); render(); };
  addEventListener("resize", () => { resize(); drawSpark(); });
  new ResizeObserver(() => resize()).observe(cv);
  setClip(clip); resize(); requestAnimationFrame(t => { last = t; requestAnimationFrame(loop); });
}

/* ================= findings ================= */
function buildFindings(D){
  const s = Object.fromEntries(D.tdpw.summary.map(r => [r.model, r]));
  const F = [
    ["good","RQ1 · 3DPW","Synthetic training transfers to real video", `${s.hmr2.pa.toFixed(1)} → ${s.GEN.pa.toFixed(1)} mm`, "PA-MPJPE drops for every fine-tuned model. Body orientation gets worse, though: unaligned error rises up to 10 mm."],
    ["warn","RQ1 · Real clips","Real-world failures are mostly tracking", "34–61%", "of frames keep the person on far, low-light, crowded, long and hugging clips. Our model is smoother on only 11 of 24 clips."],
    ["good","RQ2","3D-rendered occlusion beats 2D-pasted", "−2.1 mm", "on occluded test images, robust across 3 seeds. It costs 3.1 mm on clean images, and on 3DPW it trails 2D-pasted by 2.1 mm."],
    ["bad","RQ4","Specialists lose to one generalist", "−3.3 to −4.8 mm", "GEN beats every expert and every DGS variant on every condition, and by 1.4 mm on 3DPW."],
    ["warn","RQ5","Fusion method barely matters", "50.6 vs 51.1 mm", "Learned fusion is best on synthetic data and tied with weighted fusion on 3DPW."],
    ["bad","RQ3","Picking hard data did not help", "random wins", "Random > bandit > hard-example mining on all three test sets, all seeds."],
    ["bad","RQ6","The problem detector does not transfer", `AUC ${D.rq6real.occlusion.toFixed(2)}`, "Near-perfect on Blender images (0.9998), chance on real clips. Simple image checks do better (up to 0.68)."]
  ];
  const g = $("#findGrid");
  F.forEach(([cls, tag, head, big, body]) => { const d = document.createElement("article"); d.className = "find " + cls;
    d.innerHTML = `<div class="top"><span class="chip ${cls}">${tag}</span><span class="big num">${esc(big)}</span></div><h3>${esc(head)}</h3><p>${esc(body)}</p>`; g.appendChild(d); });
}

/* ================= 3DPW charts ================= */
function draw3dpw(D){
  const svg = $("#tdpwDots"); clear(svg); const W = widthOf(svg); const order = ["hmr2","B_clean","C_paste2d","D_occ3d","E0","E1","E2","E3","GEN","dgs_weighted","dgs_perjoint","dgs_learned","dgs_top1"];
  const rows = order.map(m => D.tdpw.summary.find(r => r.model === m)); const L = 128, R = 20, T = 26, rh = 24, H = T + rows.length * rh + 30;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  const x0 = 40, x1 = 90, sx = v => L + (v - x0) / (x1 - x0) * (W - L - R);
  for (let v = 40; v <= 90; v += 10){ el("line", {x1:sx(v), x2:sx(v), y1:T-8, y2:H-24, stroke:C.line}, svg); txt(svg, sx(v), H-8, v, {"text-anchor":"middle"}); }
  txt(svg, L, 12, "● PA-MPJPE   ○ MPJPE   (mm)");
  rows.forEach((r, i) => { const y = T + i * rh + 6, col = COL(r.model);
    txt(svg, L - 10, y + 4, NAME[r.model], {"text-anchor":"end", class:"lab"});
    el("line", {x1:sx(r.pa), x2:sx(r.mpjpe), y1:y, y2:y, stroke:col, "stroke-width":2, opacity:.35}, svg);
    const a = el("circle", {cx:sx(r.pa), cy:y, r:6, fill:col}, svg); const b = el("circle", {cx:sx(r.mpjpe), cy:y, r:5.5, fill:"none", stroke:col, "stroke-width":2}, svg);
    [a, b].forEach(n => hov(n, `${esc(NAME[r.model])}<br>PA-MPJPE ${r.pa.toFixed(1)} mm<br>MPJPE ${r.mpjpe.toFixed(1)} mm`)); });
  // forest
  const seg = $("#cmpSeg"); if (!seg.children.length){ ["PA-MPJPE","MPJPE"].forEach((m, i) => { const b = document.createElement("button"); b.className = "btn" + (i ? "" : " on"); b.textContent = m; b.onclick = () => { [...seg.children].forEach(c => c.classList.remove("on")); b.classList.add("on"); drawForest(D, m); }; seg.appendChild(b); }); }
  drawForest(D, [...seg.children].find(c => c.classList.contains("on")).textContent);
}
function drawForest(D, metric){
  const svg = $("#forest"); clear(svg); const W = widthOf(svg); const rows = D.tdpw.cmp.filter(r => r.metric === metric);
  const L = 190, R = 60, T = 14, rh = 26, H = T + rows.length * rh + 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  const ci = r => r.ci.replace(/[\[\]\s]/g, "").split(",").map(Number); const lim = Math.max(...rows.flatMap(r => ci(r).map(Math.abs))) * 1.08;
  const sx = v => L + (v + lim) / (2 * lim) * (W - L - R);
  el("line", {x1:sx(0), x2:sx(0), y1:T - 6, y2:H - 20, stroke:C.muted, "stroke-dasharray":"3 3"}, svg);
  [-lim, -lim/2, 0, lim/2, lim].forEach(v => txt(svg, sx(v), H - 4, (v > 0 ? "+" : "") + v.toFixed(0), {"text-anchor":"middle"}));
  rows.forEach((r, i) => { const y = T + i * rh + 6, [lo, hi] = ci(r), sig = r.p < 0.05, col = r.d < 0 ? C.good : C.bad;
    const nm = r.c.split(" - ").map(m => (NAME[m] || m).replace("Released ", "").replace(" generalist", "")).join(" − ");
    txt(svg, L - 10, y + 4, nm, {"text-anchor":"end", class:"lab", opacity: sig ? 1 : .45});
    const g = el("g", {opacity: sig ? 1 : .35}, svg);
    el("line", {x1:sx(lo), x2:sx(hi), y1:y, y2:y, stroke:col, "stroke-width":2}, g); el("circle", {cx:sx(r.d), cy:y, r:5, fill:col}, g);
    txt(svg, W - R + 8, y + 4, (r.d > 0 ? "+" : "") + r.d.toFixed(1), {opacity: sig ? 1 : .45});
    hov(g, `${esc(r.c)}<br>${metric}: ${r.d > 0 ? "+" : ""}${r.d.toFixed(2)} mm<br>95% CI ${esc(r.ci)}<br>p (Holm) ${r.p}`); });
}
function drawSeqTable(D){
  const cols = [["seq","Sequence"],["hmr2","HMR2.0"],["D_occ3d","D 3D-occ"],["E1","E1"],["dgs_learned","DGS"],["GEN","GEN"],["frames","Frames"]];
  let key = "seq", dir = 1; const box = $("#seqTable");
  function render(){
    const rows = [...D.tdpw.seq].sort((a, b) => (a[key] > b[key] ? 1 : -1) * dir);
    const t = document.createElement("table"); const hr = t.createTHead().insertRow();
    cols.forEach(([k, l]) => { const th = document.createElement("th"); th.textContent = l + (k === key ? (dir > 0 ? " ↑" : " ↓") : ""); th.tabIndex = 0; th.onclick = th.onkeydown = e => { if (e.type === "keydown" && e.key !== "Enter") return; if (key === k) dir = -dir; else { key = k; dir = 1; } render(); }; hr.appendChild(th); });
    const tb = t.createTBody();
    rows.forEach(r => { const tr = tb.insertRow(); const vals = cols.slice(1, 6).map(([k]) => r[k]); const best = Math.min(...vals);
      cols.forEach(([k]) => { const td = tr.insertCell(); if (k === "seq"){ td.textContent = r.seq.replace(/_/g, " "); return; } if (k === "frames"){ td.textContent = r.frames; td.style.color = C.muted; return; }
        td.textContent = r[k].toFixed(1); if (r[k] === best){ td.style.color = COL(k === "dgs_learned" ? "dgs" : k) ; td.style.fontWeight = 600; } }); });
    box.innerHTML = ""; box.appendChild(t);
  }
  render();
}
function drawOcc(D){
  const svg = $("#occChart"); clear(svg); const W = widthOf(svg), H = 250, L = 44, R = 14, T = 16, B = 46;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  const bins = D.tdpw.occ, ms = [["hmr2","HMR2.0",C.hmr],["E0","E0 clean","#b8bec9"],["E1","E1 occlusion","#8f96a3"],["dgs_learned","DGS",C.dgs],["GEN","GEN",C.gen]];
  const y0 = 40, y1 = 52, sy = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B), bw = (W - L - R) / bins.length;
  for (let v = 40; v <= 52; v += 4){ el("line", {x1:L, x2:W - R, y1:sy(v), y2:sy(v), stroke:C.line}, svg); txt(svg, L - 6, sy(v) + 4, v, {"text-anchor":"end"}); }
  bins.forEach((b, i) => { const gx = L + i * bw, w = (bw - 18) / ms.length;
    ms.forEach(([k, l, col], j) => { const v = b[k], x = gx + 9 + j * w; const r = el("rect", {x, y:sy(v), width:w - 3, height:H - B - sy(v), fill:col, rx:2}, svg); hov(r, `${l}: ${v} mm<br>${b["detector occ fraction"]} · ${b.frames} frames`); });
    txt(svg, gx + bw / 2, H - B + 16, b["detector occ fraction"], {"text-anchor":"middle"}); txt(svg, gx + bw / 2, H - B + 30, b.frames.toLocaleString() + " frames", {"text-anchor":"middle", opacity:.7}); });
  let lx = L; ms.forEach(([, l, col]) => { el("rect", {x:lx, y:2, width:9, height:9, fill:col, rx:2}, svg); txt(svg, lx + 13, 10, l); lx += 13 + l.length * 7 + 14; });
}

/* ================= clips ================= */
function buildClips(D){
  const C2 = D.clips, main = C2.filter(r => r.jit_hmr2 != null && r.jit_dgs != null);
  const better = main.filter(r => r.jit_dgs < r.jit_hmr2).length, lowDet = C2.filter(r => r.det < 0.7).length;
  const st = $("#clipStats"); [[C2.length, "clips (22 stock + 2 team)", C.ink],[`${better}/${main.length}`, "clips where DGS is smoother", C.dgs],[lowDet, "clips with person tracked < 70% of frames", C.bad],["4.3×", "DGS runtime vs HMR2.0", C.ink]].forEach(([v, l, col]) => { const d = document.createElement("div"); d.className = "stat"; d.innerHTML = `<b style="color:${col}">${v}</b><span>${l}</span>`; st.appendChild(d); });
  const groups = ["All", ...new Set(C2.map(r => r.group))]; let cur = "All"; const seg = $("#groupSeg");
  groups.forEach(g => { const b = document.createElement("button"); b.className = "btn" + (g === cur ? " on" : ""); b.textContent = g; b.onclick = () => { cur = g; [...seg.children].forEach(c => c.classList.toggle("on", c === b)); render(); }; seg.appendChild(b); });
  const mx = Math.max(...main.map(r => Math.max(r.jit_hmr2, r.jit_dgs)));
  function bar(label, v, col, max, fmt){ return `<div class="bar"><span class="muted">${label}</span><div class="t"><i style="width:${Math.min(100, v / max * 100)}%;background:${col}"></i></div><span class="v">${fmt(v)}</span></div>`; }
  function render(){
    const g = $("#gallery"); g.innerHTML = "";
    C2.filter(r => cur === "All" || r.group === cur).forEach(r => {
      const card = document.createElement("article"); card.className = "card";
      const d = r.jit_dgs - r.jit_hmr2, chips = [];
      if (r.jit_dgs != null) chips.push(d < -0.5 ? `<span class="chip good">DGS ${d.toFixed(1)} mm</span>` : d > 0.5 ? `<span class="chip bad">DGS +${d.toFixed(1)} mm</span>` : `<span class="chip">about equal</span>`);
      if (r.det < 0.7) chips.push(`<span class="chip bad">tracker loses person</span>`);
      if (r.edit) chips.push(`<span class="chip warn">edited clip</span>`); if (r.weak) chips.push(`<span class="chip warn">weak fit</span>`);
      if (!r.img) chips.push(`<span class="chip acc">team clip</span>`);
      const name = r.clip.replace(/^row\d+_/, "").replace(/_\d+$/, "").replace(/_/g, " ");
      card.innerHTML = (r.img ? `<button class="thumb" style="background-image:url(img/${r.img}.jpg)" aria-label="Enlarge contact sheet for ${esc(r.desc)}"></button>` : `<div class="thumb none">team recording: ${esc(r.clip)}</div>`) +
        `<div class="body"><div class="title"><h3>${esc(r.desc)}</h3><span class="row">ROW ${String(r.row).padStart(2,"0")}</span></div>
         <div class="muted" style="font-size:12px;margin-top:-4px">${esc(name)} · ${r.frames} frames</div>
         <div style="display:flex;gap:6px;flex-wrap:wrap">${chips.join("")}</div>
         <div class="bars">${bar("HMR2.0", r.jit_hmr2, C.hmr, mx, v => v.toFixed(1))}${r.jit_dgs != null ? bar("DGS", r.jit_dgs, C.dgs, mx, v => v.toFixed(1)) : ""}${bar("tracked", r.det, r.det < 0.7 ? C.bad : C.good, 1, v => Math.round(v * 100) + "%")}</div></div>`;
      const th = card.querySelector("button.thumb"); if (th) th.onclick = () => { $("#lbImg").src = `img/${r.img}.jpg`; $("#lbImg").alt = r.desc; $("#lbCap").textContent = `Row ${r.row} · ${r.desc} — boxes are the person detector's output`; $("#lb").classList.add("on"); };
      g.appendChild(card); });
  }
  render();
  $("#lb").onclick = () => $("#lb").classList.remove("on"); addEventListener("keydown", e => { if (e.key === "Escape") $("#lb").classList.remove("on"); });
}

/* ================= synthetic charts ================= */
function groupedBars(svg, cats, series, {y0 = 0, y1, unit = "mm", H = 260, fmt = v => v.toFixed(1)} = {}){
  clear(svg); const W = widthOf(svg), L = 44, R = 12, T = 30, B = 34; svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  const sy = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B), gw = (W - L - R) / cats.length, step = (y1 - y0) / 4;
  for (let v = y0; v <= y1 + 1e-9; v += step){ el("line", {x1:L, x2:W - R, y1:sy(v), y2:sy(v), stroke:C.line}, svg); txt(svg, L - 6, sy(v) + 4, fmt(v).replace(/\.0+$/, ""), {"text-anchor":"end"}); }
  cats.forEach((c, i) => { const gx = L + i * gw, w = (gw - 20) / series.length;
    series.forEach((s, j) => { const v = s.v[i]; if (!isFinite(v)) return; const x = gx + 10 + j * w, r = el("rect", {x, y:sy(v), width:Math.max(2, w - 4), height:H - B - sy(v), fill:s.col, rx:2}, svg);
      if (s.err){ const e = s.err[i]; el("line", {x1:x + (w - 4) / 2, x2:x + (w - 4) / 2, y1:sy(v - e), y2:sy(v + e), stroke:C.ink, "stroke-width":1.3}, svg); }
      hov(r, `${esc(s.name)} · ${esc(c)}<br>${fmt(v)}${s.err ? " ± " + s.err[i] : ""} ${unit}`); });
    txt(svg, gx + gw / 2, H - B + 18, c, {"text-anchor":"middle", class:"lab"}); });
  let lx = L; series.forEach(s => { el("rect", {x:lx, y:6, width:10, height:10, fill:s.col, rx:2}, svg); txt(svg, lx + 14, 15, s.name); lx += 14 + s.name.length * 7 + 16; });
}
function drawSelfOcc(){ const cols = ["#6d7480", C.hmr, "#b8bec9", C.dgs]; groupedBars($("#selfOcc"), ["Visible","Hidden by object","Hidden by body"], SYN.selfocc.map((s, i) => ({name:s.m, v:s.v, col:cols[i]})), {y0:0, y1:100}); }
function drawGaps(){
  const svg = $("#rq2gaps"); clear(svg); const W = widthOf(svg), rows = SYN.rq2gaps, L = 190, R = 50, T = 10, rh = 34, H = T + rows.length * rh + 24;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  const sx = v => L + (v + 6) / 12 * (W - L - R); el("line", {x1:sx(0), x2:sx(0), y1:4, y2:H - 18, stroke:C.muted, "stroke-dasharray":"3 3"}, svg);
  [-6, -3, 0, 3, 6].forEach(v => txt(svg, sx(v), H - 4, (v > 0 ? "+" : "") + v, {"text-anchor":"middle"}));
  rows.forEach((r, i) => { const y = T + i * rh + 10, col = r.d < 0 ? C.good : C.bad; txt(svg, L - 10, y + 4, r.l, {"text-anchor":"end", class:"lab"});
    const b = el("rect", {x:Math.min(sx(0), sx(r.d)), y:y - 7, width:Math.abs(sx(r.d) - sx(0)), height:14, fill:col, rx:3, opacity:.85}, svg);
    if (r.lo != null) el("line", {x1:sx(r.lo), x2:sx(r.hi), y1:y, y2:y, stroke:C.ink, "stroke-width":1.5}, svg);
    txt(svg, W - R + 8, y + 4, (r.d > 0 ? "+" : "") + r.d.toFixed(1)); hov(b, `${esc(r.l)}: ${r.d > 0 ? "+" : ""}${r.d} mm${r.lo != null ? `<br>95% CI [${r.lo}, ${r.hi}]` : ""}<br>same sign in all 3 seeds`); });
}
function drawHeat(){
  const svg = $("#heat"); clear(svg); const W = Math.max(widthOf(svg), 640), rows = SYN.heat, cols = SYN.conds, L = 120, T = 30, rh = 34, H = T + rows.length * rh + 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H); const cw = (W - L - 6) / cols.length;
  cols.forEach((c, j) => { txt(svg, L + j * cw + cw / 2, 18, c, {"text-anchor":"middle", class:"lab"});
    const vals = rows.map(r => r[1][j]), lo = Math.min(...vals), hi = Math.max(...vals);
    rows.forEach((r, i) => { const v = r[1][j], t = (v - lo) / (hi - lo || 1), x = L + j * cw, y = T + i * rh;
      const isGen = r[0] === "GEN", base = isGen ? [195,166,255] : r[0].startsWith("DGS") ? [255,138,61] : [127,176,232];
      const a = 0.85 - t * 0.75; const rect = el("rect", {x:x + 2, y:y + 2, width:cw - 4, height:rh - 4, rx:5, fill:`rgba(${base.join(",")},${a.toFixed(2)})`}, svg);
      if (v === lo) el("rect", {x:x + 2, y:y + 2, width:cw - 4, height:rh - 4, rx:5, fill:"none", stroke:C.ink, "stroke-width":2}, svg);
      txt(svg, x + cw / 2, y + rh / 2 + 4, v.toFixed(1), {"text-anchor":"middle", style:`fill:${a > .5 ? "#0f1115" : C.ink};font-size:12px`});
      hov(rect, `${esc(r[0])} on ${esc(c)}: ${v} mm${v === lo ? "<br>best in column" : ""}`); }); });
  rows.forEach((r, i) => txt(svg, L - 10, T + i * rh + rh / 2 + 4, r[0], {"text-anchor":"end", class:"lab"}));
}
function hbars(svg, rows, {x0, x1, unit = "mm", colFn, L = 130}){
  clear(svg); const W = widthOf(svg), R = 54, T = 6, rh = 30, H = T + rows.length * rh + 20; svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  const sx = v => L + (v - x0) / (x1 - x0) * (W - L - R);
  rows.forEach(([l, v, col], i) => { const y = T + i * rh; txt(svg, L - 10, y + 18, l, {"text-anchor":"end", class:"lab"}); const r = el("rect", {x:L, y:y + 6, width:Math.max(2, sx(v) - L), height:16, rx:3, fill:col || colFn(l)}, svg); txt(svg, sx(v) + 6, y + 18, v.toFixed(1)); hov(r, `${esc(l)}: ${v} ${unit}`); });
  txt(svg, L, H - 4, `axis starts at ${x0} ${unit}`, {opacity:.7});
}
function drawFusion(){ hbars($("#fusion"), SYN.fusion.map(([l, v]) => [l, v, l === "Learned" ? C.dgs : "#8f96a3"]), {x0:48, x1:52}); }
function drawRq4Real(D){ const s = Object.fromEntries(D.tdpw.summary.map(r => [r.model, r.pa])); hbars($("#rq4real"), [["Released HMR2.0", s.hmr2, C.hmr],["Best expert (E3)", s.E3, "#b8bec9"],["DGS learned", s.dgs_learned, C.dgs],["GEN", s.GEN, C.gen]], {x0:40, x1:50, L:150}); }
function drawRq3(){ const cols = ["#8f96a3", C.dgs, C.bad]; groupedBars($("#rq3chart"), SYN.rq3.tests.map(t => t + " test"), SYN.rq3.s.map(([n, v, e], i) => ({name:n, v, err:e, col:cols[i]})), {y0:30, y1:70}); }
function drawAuc(D){
  const svg = $("#aucChart"); const r = D.rq6real;
  groupedBars(svg, ["Occlusion","Quality","Framing"], [
    {name:"Detector, Blender test", v:[SYN.aucSyn.occlusion, SYN.aucSyn.quality, SYN.aucSyn.framing], col:"#8f96a3"},
    {name:"Detector, real clips", v:[r.occlusion, r.quality, r.framing], col:C.dgs},
    {name:"Simple image checks, real clips", v:[NaN, r.cheap_quality, r.cheap_framing], col:C.sg}], {y0:0, y1:1, unit:"AUC", fmt:v => isNaN(v) ? "n/a" : v.toFixed(2)});
  const H = +svg.getAttribute("height"), W = +svg.getAttribute("width"), y = 30 + 0.5 * (H - 30 - 34);
  el("line", {x1:44, x2:W - 12, y1:y, y2:y, stroke:C.warn, "stroke-dasharray":"5 4"}, svg); txt(svg, 50, y - 6, "chance = 0.5", {style:`fill:${C.warn}`});
}

/* nav highlight */
const links = [...document.querySelectorAll("#nav a")];
const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id)); }), {rootMargin:"-40% 0px -55% 0px"});
document.querySelectorAll("main section").forEach(s => io.observe(s));
})();
