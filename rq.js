/* Research questions as separate pages: one section visible at a time, with tabs, a
   question / test / answer card, jump links to the section's parts, and previous / next. */
window.RQ = (() => {
  const P = [
    {id:"findings",  k:"◎",     l:"Overview"},
    {id:"viewer",    k:"▣",     l:"Model output"},
    {id:"rq1-3dpw",  k:"RQ1",   l:"3DPW accuracy", v:"good",
      q:"Does training on synthetic Blender renders make the model better on real video?",
      t:"Released HMR2.0 against every model we fine-tuned, on 3DPW: 17,764 real frames in 24 outdoor sequences with ground-truth 3D joints.",
      a:"Yes for body articulation: PA-MPJPE drops from 47.9 to 42.9 mm (GEN). Facing direction gets worse, though: MPJPE rises from 73.2 to 83.7 mm for GEN."},
    {id:"rq1-clips", k:"RQ1",   l:"Real clips", v:"warn",
      q:"Where does the system break on everyday footage?",
      t:"24 phone-style clips covering the 29-row test plan (crowds, occlusion, low light, camera motion and more). No ground truth, so we measure stability and tracking.",
      a:"Mostly at finding and following people: the main person is kept in only 34–61% of frames on far, low-light, crowded, long and hugging clips. Our model is smoother on 11 of 24 clips."},
    {id:"rq2",       k:"RQ2",   l:"2D vs 3D occlusion", v:"good",
      q:"Is it better to train with occluders rendered in 3D, or with 2D cut-outs pasted onto the image?",
      t:"The same model fine-tuned three ways (B clean, C 2D-pasted, D 3D-rendered), three seeds each, tested on synthetic occlusion and on 3DPW.",
      a:"3D-rendered wins on occluded images by 2.1 mm in every seed. It costs 3.1 mm on clean images and trails 2D-pasted by 2.1 mm on 3DPW."},
    {id:"rq3",       k:"RQ3",   l:"Hard-data picking", v:"bad",
      q:"Does choosing hard training examples beat picking them at random?",
      t:"The same 2,000-image budget chosen three ways: random, hard-example mining and a Hedge bandit, three seeds each.",
      a:"No. Random sampling wins on all three test sets in every seed."},
    {id:"rq4",       k:"RQ4·5", l:"Experts vs one", v:"bad",
      q:"RQ4: do specialists for each problem beat one generalist? RQ5: which way of fusing the specialists works best?",
      t:"Four LoRA experts with three fusion methods, against one GEN adapter with the same total size and compute, on synthetic tests and on 3DPW.",
      a:"GEN wins every condition by 3.3–4.8 mm, and by 1.4 mm on 3DPW. The fusion method barely matters (50.6 vs 51.1 mm)."},
    {id:"rq6",       k:"RQ6",   l:"Problem detector", v:"bad",
      q:"Can a trained model recognise what is wrong with a frame?",
      t:"A ResNet-18 problem detector trained on Blender renders, scored on held-out renders and on labelled real clips.",
      a:"Near-perfect on renders (AUC 0.9998), chance on real clips (0.50 occlusion, 0.33 quality, 0.54 framing). Simple image checks do better (up to 0.68)."}
  ];
  const VL = {good:"Held up", warn:"Mixed", bad:"Did not hold up"};
  const sec = id => document.getElementById(id);

  P.forEach((p, i) => {
    const s = sec(p.id); if (!s) return;
    const tabs = document.createElement("nav"); tabs.className = "rqtabs"; tabs.setAttribute("aria-label", "Research questions");
    tabs.innerHTML = P.map(x => `<a href="#${x.id}" class="${x.id === p.id ? "on" : ""}"${x.id === p.id ? ' aria-current="page"' : ""}><b>${x.k}</b>${x.l}</a>`).join("");
    s.prepend(tabs);
    const head = s.querySelector(".sec-head");
    if (p.q && head) {
      const c = document.createElement("div"); c.className = "rqcard";
      c.innerHTML = `<div><h4>Question</h4><p>${p.q}</p></div><div><h4>What we tested</h4><p>${p.t}</p></div><div class="ans ${p.v}"><h4>Answer · <span class="chip ${p.v}">${VL[p.v]}</span></h4><p>${p.a}</p></div>`;
      head.after(c);
      const parts = [...s.querySelectorAll(".panel h3")].filter(h => h.textContent.trim());
      if (parts.length > 1) {
        const sub = document.createElement("div"); sub.className = "rqsub";
        sub.innerHTML = `<span class="muted">On this page:</span>` + parts.map((h, j) => `<a href="#" data-j="${j}">${h.textContent.replace(/\s*\(.*\)$/, "")}</a>`).join("");
        sub.addEventListener("click", e => { const a = e.target.closest("a"); if (!a) return; e.preventDefault(); parts[+a.dataset.j].closest(".panel").scrollIntoView({behavior:"smooth", block:"start"}); });
        c.after(sub);
      }
    }
    const nav = document.createElement("div"); nav.className = "rqnext";
    const pr = P[i - 1], nx = P[i + 1];
    nav.innerHTML = (pr ? `<a href="#${pr.id}"><span>← Previous</span>${pr.k.length > 1 && pr.k.startsWith("RQ") ? pr.k + " · " : ""}${pr.l}</a>` : "") +
                    (nx ? `<a class="nx" href="#${nx.id}"><span>Next →</span>${nx.k.startsWith("RQ") ? nx.k + " · " : ""}${nx.l}</a>` : "");
    s.appendChild(nav);
  });

  // verdict cards on the overview open their own page
  const TAG = {"RQ1 · 3DPW":"rq1-3dpw", "RQ1 · Real clips":"rq1-clips", "RQ2":"rq2", "RQ3":"rq3", "RQ4":"rq4", "RQ5":"rq4", "RQ6":"rq6"};
  const fg = document.getElementById("findGrid");
  if (fg) {
    fg.addEventListener("click", e => { const c = e.target.closest(".find"); if (!c) return; const id = TAG[c.querySelector(".chip").textContent.trim()]; if (id) location.hash = id; });
    new MutationObserver(() => fg.querySelectorAll(".find:not([data-go])").forEach(c => { c.dataset.go = 1; c.tabIndex = 0; c.setAttribute("role", "link");
      c.addEventListener("keydown", e => { if (e.key === "Enter") c.click(); });
      const g = document.createElement("span"); g.className = "go"; g.textContent = "Open this question →"; c.appendChild(g); })).observe(fg, {childList:true});
  }

  function show(h) {
    let id = h;
    const el = document.getElementById(h);
    if (!P.some(p => p.id === h)) { const s = el && el.closest("#page-results > section"); id = s ? s.id : "findings"; }
    document.querySelectorAll("#page-results > section").forEach(s => s.classList.toggle("on", s.id === id));
    document.querySelectorAll("#nav a").forEach(a => { if (a.dataset.page === "results") a.classList.toggle("on", a.getAttribute("href") === "#" + id); });
    const p = P.find(x => x.id === id);
    document.title = (p ? (p.k.startsWith("RQ") ? p.k + " · " : "") + p.l + " · " : "") + "Real World to Blender Studio";
  }
  return {show};
})();
if (location.hash && document.getElementById("page-results").classList.contains("on")) RQ.show(location.hash.slice(1));
