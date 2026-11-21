/* Animated architecture graph: packets flow from the video, through person finding, the problem
   identifier and the router, split across the specialists on HMR2.0, merge in fusion, and end as characters. */
(() => {
  const root = document.getElementById("flow"); if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PCOL = ["#ff8a3d", "#5ab0ff", "#c38bff"];               // person A, B, C
  const ECOL = ["#9aa1ad", "#ff6b6b", "#f2c14e", "#5fd38a"];    // E0 clean, E1 occ, E2 quality, E3 framing

  const SCEN = [
    {k:"clean",  l:"Clean shot",          s:[0.06, 0.05, 0.04], d:"Nothing is wrong, so the clean expert E0 gets almost all the weight."},
    {k:"occ",    l:"Behind a chair",      s:[0.86, 0.10, 0.12], d:"The identifier sees occlusion, so most of the work goes to E1, the occlusion expert."},
    {k:"dark",   l:"Low light, blurry",   s:[0.10, 0.82, 0.08], d:"Image quality is the problem, so E2 leads."},
    {k:"cut",    l:"Cut off at the edge", s:[0.08, 0.10, 0.84], d:"The body leaves the frame, so E3, the framing expert, leads."},
    {k:"mix",    l:"Occluded and dark",   s:[0.70, 0.62, 0.10], d:"Two problems at once: E1 and E2 share the work and fusion blends them."},
    {k:"real",   l:"Real clip (honest)",  s:[0.48, 0.52, 0.47], d:"On real phone video our identifier is near chance (AUC 0.33–0.54), so the weights end up spread out. This is the main weakness we found."}
  ];
  // same formula as pose_lib/detector.router_weights
  const weights = p => { const w0 = (1 - p[0]) * (1 - p[1]) * (1 - p[2]); const w = [w0, ...p]; const t = w.reduce((a, b) => a + b, 0) || 1; return w.map(x => x / t); };

  root.innerHTML = `
  <div class="flowbar">
    <div class="seg" role="group" aria-label="Pipeline">
      <button class="btn on" data-mode="route">Specialists + router</button>
      <button class="btn" data-mode="gen">One generalist (GEN)</button></div>
    <div class="seg" role="group" aria-label="People"><span class="muted">People</span>
      ${[1, 2, 3].map(n => `<button class="btn ${n === 2 ? "on" : ""}" data-n="${n}">${n}</button>`).join("")}</div>
    <div class="seg"><button class="btn" id="flPlay">Pause</button><button class="btn" id="flStep">Next step</button></div>
  </div>
  <div class="flowgrid">
    <div class="flowbox"><svg id="flSvg" viewBox="0 0 1400 610" role="img" aria-label="Animated pipeline diagram"></svg>
      <div class="flowcap" aria-live="polite"><span class="num" id="flN">01</span><span id="flCap"></span></div></div>
    <div class="flowside">
     <div class="panel"><div class="eyebrow">Scenario</div>
      <div class="scen" id="flScen">${SCEN.map((c, i) => `<button class="btn ${i === 1 ? "on" : ""}" data-i="${i}">${c.l}</button>`).join("")}</div>
      <p class="note" id="flScenD"></p></div>
     <div class="panel"><div class="eyebrow">Problem scores (drag them)</div>
      ${["Occlusion", "Image quality", "Framing"].map((l, i) => `<label class="sl"><span>${l}</span><input type="range" min="0" max="1" step="0.01" data-s="${i}"><b class="num" id="flSv${i}"></b></label>`).join("")}
      <div class="eyebrow" style="margin-top:6px">Router weights</div>
      <div id="flW" class="wbars"></div></div>
     <div class="panel flinfo" id="flInfo"><div class="eyebrow">Details</div><p class="note">Click any box in the diagram for what it does and the evidence.</p></div>
    </div>
  </div>`;

  const svg = root.querySelector("#flSvg");
  const mk = (tag, at, par = svg) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); par.appendChild(e); return e; };
  const defs = mk("defs", {});
  defs.innerHTML = `<filter id="flGlow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <marker id="flArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4a5160"/></marker>`;

  // ---------- nodes ----------
  const N = {
    input:  {x:20,   y:245, w:135, h:100, t:"Video", sub:"photo · webcam", s:"ok", info:"in_video"},
    detect: {x:195,  y:245, w:150, h:100, t:"Find every person", sub:"YOLO11 + tracker", s:"weak", info:"iou"},
    ident:  {x:385,  y:215, w:175, h:160, t:"Problem identifier", sub:"ResNet-18", s:"weak", info:"detector"},
    router: {x:600,  y:265, w:105, h:60,  t:"Router", sub:"scores → weights", s:"ok", info:"router"},
    e0:     {x:762,  y:72,  w:185, h:54,  t:"E0 · clean", sub:"LoRA expert", s:"ok", info:"e0", e:0},
    e1:     {x:762,  y:138, w:185, h:54,  t:"E1 · occlusion", sub:"LoRA expert", s:"ok", info:"e1", e:1},
    e2:     {x:762,  y:204, w:185, h:54,  t:"E2 · quality", sub:"LoRA expert", s:"ok", info:"e2", e:2},
    e3:     {x:762,  y:270, w:185, h:54,  t:"E3 · framing", sub:"LoRA expert", s:"ok", info:"e3", e:3},
    gen:    {x:762,  y:345, w:185, h:54,  t:"GEN · generalist", sub:"one adapter for all", s:"ok", info:"gen"},
    p0:     {x:762,  y:425, w:185, h:40,  t:"Fast-motion expert", sub:"", s:"plan", info:"e4"},
    p1:     {x:762,  y:473, w:185, h:40,  t:"Contact expert", sub:"", s:"plan", info:"e5"},
    p2:     {x:762,  y:521, w:185, h:40,  t:"World-space expert", sub:"", s:"plan", info:"e6"},
    fuse:   {x:1000, y:255, w:120, h:80,  t:"Fusion", sub:"combine poses", s:"ok", info:"fl"},
    smooth: {x:1150, y:255, w:105, h:80,  t:"Smooth", sub:"over time", s:"ok", info:"sg"},
    out:    {x:1268, y:225, w:124, h:140, t:"Characters", sub:"web · Blender", s:"ok", info:"web"}
  };
  // base model band behind the experts
  const band = mk("g", {class:"fl-band", tabindex:0, role:"button", "aria-label":"HMR2.0 base model"});
  mk("rect", {x:745, y:22, width:219, height:552, rx:16}, band);
  mk("text", {x:854, y:46, "text-anchor":"middle", class:"fl-bandt"}, band).textContent = "HMR2.0 base model · frozen";
  mk("text", {x:854, y:590, "text-anchor":"middle", class:"fl-small"}, svg).textContent = "dashed = not built yet";
  band.addEventListener("click", () => info("hmr2"));

  const edgeG = mk("g", {}), nodeG = mk("g", {}), partG = mk("g", {});
  const R = n => [n.x + n.w, n.y + n.h / 2], L = n => [n.x, n.y + n.h / 2];
  const curve = (a, b) => { const dx = Math.max(40, (b[0] - a[0]) * 0.5); return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`; };
  const E = {};
  const edge = (id, d, cls = "") => { const p = mk("path", {d, class:"fl-edge " + cls, "marker-end":"url(#flArr)"}, edgeG); E[id] = {p, len:p.getTotalLength()}; };
  edge("in-det", curve(R(N.input), L(N.detect)));
  edge("det-id", curve(R(N.detect), L(N.ident)), "route");
  edge("id-rt", curve(R(N.ident), L(N.router)), "route");
  [0, 1, 2, 3].forEach(k => { edge("rt-e" + k, curve(R(N.router), L(N["e" + k])), "route"); edge("e" + k + "-fu", curve(R(N["e" + k]), L(N.fuse)), "route"); });
  edge("fu-sm", curve(R(N.fuse), L(N.smooth)), "route");
  edge("det-gen", `M${N.detect.x + N.detect.w / 2},${N.detect.y + N.detect.h} C${N.detect.x + 75},470 640,${N.gen.y + 27} ${N.gen.x},${N.gen.y + 27}`, "gen");
  edge("gen-sm", curve(R(N.gen), [N.smooth.x, N.smooth.y + N.smooth.h - 14]), "gen");
  edge("sm-out", curve(R(N.smooth), L(N.out)));

  const box = {};
  for (const id in N) {
    const n = N[id];
    const g = mk("g", {class:`fl-node ${n.s}`, tabindex:0, role:"button", "aria-label":n.t}, nodeG);
    mk("rect", {x:n.x, y:n.y, width:n.w, height:n.h, rx:10, class:"fl-bg"}, g);
    mk("rect", {x:n.x, y:n.y + 8, width:4, height:n.h - 16, rx:2, class:"fl-tick"}, g);
    const ty = id === "ident" ? n.y + 24 : id === "out" ? n.y + 22 : n.y + n.h / 2 - (n.sub ? 4 : -4);
    mk("text", {x:n.x + 14, y:ty, class:"fl-t"}, g).textContent = n.t;
    if (n.sub) mk("text", {x:n.x + 14, y:ty + 16, class:"fl-small"}, g).textContent = n.sub;
    g.addEventListener("click", () => info(n.info, id));
    g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); info(n.info, id); } });
    box[id] = g;
  }
  // identifier score bars
  const bars = ["occ", "qual", "frame"].map((l, i) => {
    const y = N.ident.y + 62 + i * 30;
    mk("text", {x:N.ident.x + 14, y:y + 9, class:"fl-small"}, box.ident).textContent = l;
    mk("rect", {x:N.ident.x + 58, y, width:80, height:10, rx:5, class:"fl-track"}, box.ident);
    const b = mk("rect", {x:N.ident.x + 58, y, width:0, height:10, rx:5, fill:ECOL[i + 1]}, box.ident);
    const v = mk("text", {x:N.ident.x + 144, y:y + 9, class:"fl-small num"}, box.ident);
    return {b, v};
  });
  // expert weight labels + fill
  const wlab = [0, 1, 2, 3].map(k => { const n = N["e" + k];
    const f = mk("rect", {x:n.x + 1, y:n.y + n.h - 5, width:0, height:4, rx:2, fill:ECOL[k]}, box["e" + k]);
    const t = mk("text", {x:n.x + n.w - 12, y:n.y + n.h / 2 + 4, "text-anchor":"end", class:"fl-w num"}, box["e" + k]);
    return {f, t}; });
  // output: little figures, one per person
  const figs = mk("g", {}, box.out);
  const drawFigs = () => { figs.innerHTML = ""; const n = N.out, cnt = people, sp = n.w / (cnt + 1);
    for (let i = 0; i < cnt; i++) { const cx = n.x + sp * (i + 1), cy = n.y + 70, c = PCOL[i];
      const g = mk("g", {stroke:c, "stroke-width":3, "stroke-linecap":"round", fill:"none", class:"fl-fig"}, figs);
      g.style.animationDelay = (i * 0.25) + "s";
      mk("circle", {cx, cy, r:6, fill:c, stroke:"none"}, g);
      mk("path", {d:`M${cx},${cy + 7} L${cx},${cy + 30} M${cx - 12},${cy + 12} L${cx},${cy + 16} L${cx + 12},${cy + 10} M${cx},${cy + 30} L${cx - 9},${cy + 48} M${cx},${cy + 30} L${cx + 9},${cy + 48}`}, g); } };
  // input: frame with people dots
  const inDots = mk("g", {}, box.input);
  const drawIn = () => { inDots.innerHTML = ""; const n = N.input;
    mk("rect", {x:n.x + 14, y:n.y + 62, width:n.w - 28, height:28, rx:4, class:"fl-track"}, inDots);
    for (let i = 0; i < people; i++) mk("circle", {cx:n.x + 30 + i * 22, cy:n.y + 76, r:6, fill:PCOL[i]}, inDots); };

  // ---------- state ----------
  let mode = "route", people = 2, scen = 1, sc = SCEN[1].s.slice(), W = weights(sc), playing = !reduce;
  const sliders = [...root.querySelectorAll("[data-s]")];
  function setScores(s, label) {
    sc = s.slice(); W = weights(sc);
    sliders.forEach((r, i) => { r.value = sc[i]; root.querySelector("#flSv" + i).textContent = sc[i].toFixed(2); });
    bars.forEach((b, i) => { b.b.setAttribute("width", 80 * sc[i]); b.v.textContent = sc[i].toFixed(2); });
    wlab.forEach((l, k) => { l.t.textContent = Math.round(W[k] * 100) + "%"; l.f.setAttribute("width", (N.e0.w - 2) * W[k]); });
    [0, 1, 2, 3].forEach(k => { const a = mode === "route" ? 0.25 + 0.75 * W[k] / Math.max(...W) : 0.2;
      ["rt-e" + k, "e" + k + "-fu"].forEach(id => { E[id].p.style.strokeWidth = (1.5 + 9 * W[k]).toFixed(1); E[id].p.style.opacity = a; });
      box["e" + k].style.opacity = mode === "route" ? 0.45 + 0.55 * W[k] / Math.max(...W) : 0.35; });
    root.querySelector("#flW").innerHTML = ["E0 clean", "E1 occlusion", "E2 quality", "E3 framing"].map((l, k) =>
      `<div class="wrow"><span>${l}</span><i><b style="width:${(W[k] * 100).toFixed(0)}%;background:${ECOL[k]}"></b></i><em class="num">${Math.round(W[k] * 100)}%</em></div>`).join("");
    if (label !== undefined) root.querySelector("#flScenD").textContent = label;
  }
  function setMode(m) {
    mode = m; svg.classList.toggle("genmode", m === "gen");
    root.querySelectorAll("[data-mode]").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
    setScores(sc); restart();
  }
  root.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => setMode(b.dataset.mode));
  root.querySelectorAll("[data-n]").forEach(b => b.onclick = () => { people = +b.dataset.n; root.querySelectorAll("[data-n]").forEach(x => x.classList.toggle("on", x === b)); drawIn(); drawFigs(); restart(); });
  root.querySelectorAll("#flScen [data-i]").forEach(b => b.onclick = () => { scen = +b.dataset.i; root.querySelectorAll("#flScen [data-i]").forEach(x => x.classList.toggle("on", x === b)); setScores(SCEN[scen].s, SCEN[scen].d); restart(); });
  sliders.forEach(r => r.oninput = () => { const s = sc.slice(); s[+r.dataset.s] = +r.value; root.querySelectorAll("#flScen [data-i]").forEach(x => x.classList.remove("on")); setScores(s, "Your own scores. The router uses the same formula as our code: the clean expert gets (1−occ)(1−quality)(1−framing), each other expert gets its problem score, then everything is normalised."); });

  // ---------- steps ----------
  const top = () => W.indexOf(Math.max(...W));
  const pname = () => people === 1 ? "the person" : `each of the ${people} people`;
  const STEPS = {
    route: [
      {n:["input"], e:["in-det"], c:() => `A frame comes in with ${people === 1 ? "one person" : people + " people"} in it.`},
      {n:["detect"], e:["det-id"], c:() => `The detector boxes every person and the tracker gives each one an id. From here ${pname()} goes through separately (one colour per person).`},
      {n:["ident"], e:[], c:() => `The problem identifier scores the crop: occlusion ${sc[0].toFixed(2)}, quality ${sc[1].toFixed(2)}, framing ${sc[2].toFixed(2)}.`, hold:1300},
      {n:["ident"], e:["id-rt"], c:() => "The scores go to the router."},
      {n:["router"], e:[0, 1, 2, 3].map(k => "rt-e" + k), split:true, c:() => `The router turns scores into trust: ${["E0", "E1", "E2", "E3"].map((l, k) => l + " " + Math.round(W[k] * 100) + "%").join(", ")}. Thicker line, more weight.`},
      {n:["e0", "e1", "e2", "e3"], e:[0, 1, 2, 3].map(k => "e" + k + "-fu"), split:true, band:true, c:() => `Each specialist is a small LoRA adapter on the frozen HMR2.0 base model, and each predicts a full 3D body pose. ${["E0", "E1", "E2", "E3"][top()]} counts most here.`},
      {n:["fuse"], e:["fu-sm"], c:() => "Fusion combines the specialists' poses into one, weighted by the router (per joint in the per-joint version)."},
      {n:["smooth"], e:["sm-out"], c:() => "Smoothing over time removes frame-to-frame shake (about 14 mm of jitter down to about 2 mm)."},
      {n:["out"], e:[], c:() => `${people === 1 ? "The person drives their" : "Each person drives their own"} character, in the browser or as a Blender rig.`, hold:1600}
    ],
    gen: [
      {n:["input"], e:["in-det"], c:() => `A frame comes in with ${people === 1 ? "one person" : people + " people"} in it.`},
      {n:["detect"], e:["det-gen"], c:() => `Every person is found and tracked. No identifier or router: ${pname()} goes straight to one model.`},
      {n:["gen"], e:["gen-sm"], band:true, c:() => "GEN is one adapter trained on all the problem types with the same budget. It was our most accurate model: 42.9 mm PA-MPJPE on 3DPW, against 44.3 for the best fusion and 47.9 for plain HMR2.0."},
      {n:["smooth"], e:["sm-out"], c:() => "Smoothing over time removes frame-to-frame shake."},
      {n:["out"], e:[], c:() => `${people === 1 ? "The person drives their" : "Each person drives their own"} character.`, hold:1600}
    ]
  };
  let step = 0, parts = [], timer = 0, stepT = 0;
  const DUR = 1050;
  function enter(i) {
    const st = STEPS[mode][i]; step = i; stepT = performance.now();
    Object.values(box).forEach(g => g.classList.remove("hot")); band.classList.toggle("hot", !!st.band);
    st.n.forEach(id => box[id].classList.add("hot"));
    Object.values(E).forEach(x => x.p.classList.remove("live")); st.e.forEach(id => E[id].p.classList.add("live"));
    root.querySelector("#flN").textContent = String(i + 1).padStart(2, "0");
    root.querySelector("#flCap").textContent = st.c();
    for (let p = 0; p < people; p++) st.e.forEach(id => {
      const k = /e(\d)/.exec(id), w = st.split && k ? W[+k[1]] : 1;
      if (st.split && w < 0.03) return;
      const c = mk("circle", {r:(st.split ? 3 + 7 * Math.sqrt(w) : 6), fill:PCOL[p], filter:"url(#flGlow)"}, partG);
      parts.push({c, e:E[id], t0:performance.now() + p * 180});
    });
  }
  function restart() { parts.forEach(p => p.c.remove()); parts = []; enter(0); if (reduce) draw(performance.now()); }
  function next() { parts.forEach(p => p.c.remove()); parts = []; enter((step + 1) % STEPS[mode].length); }
  function draw(now) {
    parts = parts.filter(p => { const u = Math.min(1, Math.max(0, (now - p.t0) / DUR));
      const q = p.e.p.getPointAtLength(p.e.len * (reduce ? 1 : u)); p.c.setAttribute("cx", q.x); p.c.setAttribute("cy", q.y);
      p.c.style.opacity = u < 1 || reduce ? 1 : 0; return true; });
  }
  // ambient trickle so idle edges still show direction
  let amb = [], lastAmb = 0;
  function ambient(now) {
    if (now - lastAmb > 260) { lastAmb = now;
      const ids = Object.keys(E).filter(id => mode === "route" ? !E[id].p.classList.contains("gen") : !E[id].p.classList.contains("route"));
      const id = ids[Math.floor(Math.random() * ids.length)], k = /e(\d)/.exec(id);
      if (!(mode === "route" && k && W[+k[1]] < 0.08)) amb.push({c:mk("circle", {r:2.2, class:"fl-amb"}, partG), e:E[id], t0:now}); }
    amb = amb.filter(a => { const u = (now - a.t0) / 1800; if (u >= 1) { a.c.remove(); return false; }
      const q = a.e.p.getPointAtLength(a.e.len * u); a.c.setAttribute("cx", q.x); a.c.setAttribute("cy", q.y); return true; });
  }
  function loop(now) {
    const visible = root.offsetParent !== null && !document.hidden;
    if (visible && playing) {
      draw(now); ambient(now);
      const st = STEPS[mode][step], need = (st.e.length ? DUR + (people - 1) * 180 : 0) + (st.hold || 450);
      if (now - stepT > need) next();
    }
    requestAnimationFrame(loop);
  }
  const playBtn = root.querySelector("#flPlay");
  playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; stepT = performance.now(); };
  root.querySelector("#flStep").onclick = () => { playing = false; playBtn.textContent = "Play"; next(); draw(performance.now() + DUR); };
  if (!playing) playBtn.textContent = "Play";

  // ---------- details ----------
  const CUSTOM = {
    fuse: {name:"Fusion", s:"ok", what:"Combines the specialists' SMPL poses using the router weights. Three versions: weighted rotation average, per-joint (hidden joints lean on E1), and a small learned network.", io:"4 expert poses + weights → 1 pose", ev:"3DPW 44.3 mm for all three; learned is best on synthetic (50.6 mm). None beats GEN (42.9 mm)."},
    detect: {name:"Find every person", s:"weak", what:"YOLO11 boxes every person; an IoU tracker links boxes over time so each person keeps one id. The browser studio uses MediaPipe for up to 5 people instead.", io:"frame → one track per person", ev:"Weakest link on real video: the main person is kept in only 34–61% of frames on far, low-light, crowd and hug clips. A re-identifying tracker is not built yet."}
  };
  function info(key, nid) {
    let nd = CUSTOM[nid];
    if (!nd) for (const stg of (window.ARCH || [])) for (const x of stg.nodes) if (x.id === key) nd = x;
    if (!nd) return;
    const ST = {ok:["Built", "good"], weak:["Built · weak on real video", "warn"], used:["Used, not trained by us", "acc"], plan:["Not built yet", ""]};
    Object.values(box).forEach(g => g.classList.remove("sel")); if (nid && box[nid]) box[nid].classList.add("sel");
    root.querySelector("#flInfo").innerHTML = `<div class="eyebrow">Details</div><h3>${nd.name}</h3><span class="chip ${ST[nd.s][1]}">${ST[nd.s][0]}</span><p>${nd.what}</p><div class="io"><span class="muted">Input → output</span><b class="num">${nd.io}</b></div><div class="io"><span class="muted">Evidence</span><p style="margin:0">${nd.ev}</p></div>`;
  }

  drawIn(); drawFigs(); setScores(SCEN[1].s, SCEN[1].d); enter(0);
  requestAnimationFrame(loop);
})();
