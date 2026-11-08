import { json, loadReplay } from "./data.js";
import { fmtSI, logPlot, timePlot } from "./plots.js";
import { Viewer } from "./viewer.js";

const COLORS = {
  mujoco311: "#9aa5b1", mujoco4: "#4fb3ff", mjwarp4: "#3fcf8e", newton: "#b48cff",
  genesis: "#ff9f43", "article:mujoco_cpu": "#c9d1d9", "article:newton_mjwarp": "#d6b8ff",
  "article:genesis": "#ffd29b", "article:physx": "#ff7b9c",
};
const PHASES = [["Phase 1", ["slide", "drop", "hinge"]], ["Phase 2", ["collision-chain"]],
                ["Phase 3", ["panda-effort", "panda-tracking", "grasp", "stack", "push"]]];
const DEFAULT_SIGNAL = {
  slide: "cube|fz", drop: "cube|fz", hinge: "q|0", "collision-chain": "delta",
  "panda-effort": "delta", "panda-tracking": "delta", grasp: "cube|z", stack: "upper|z",
  push: "cube|x",
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) e.append(k instanceof Node ? k : document.createTextNode(k ?? ""));
  return e;
};
const get = (o, ...path) => path.reduce((a, k) => (a == null ? undefined : a[k]), o);
const mm = (m, digits = 3) => (m == null ? "—" : `${(m * 1e3).toPrecision(digits)} mm`);
const num = (v) => {
  if (v == null) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.map(num).join(", ");
  if (typeof v === "object") return "…";
  if (typeof v !== "number") return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e6) ? v.toExponential(2) : String(+v.toPrecision(5));
};

let INDEX, SUMMARY, viewer, current, pending = null, playing = false, playT = 0, plot;
const label = (r) => INDEX.routes[r] ?? r;

// --- tabs ----------------------------------------------------------------------------

function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === name));
  if (name === "replay" && !current && !pending) openReplay(INDEX.replays[0].file);
  if (name === "runtime") runtime(document.querySelector("#workload button.active")?.dataset.w
                                  ?? "panda");
  const hash = name === "replay" && (pending ?? current?.file) ? `replay:${pending ?? current.file}`
    : name;
  if (location.hash.slice(1) !== hash) history.replaceState(null, "", `#${hash}`);
}

// --- overview ------------------------------------------------------------------------

function findings() {
  const S = SUMMARY;
  const cards = [];
  const card = (labelText, big, text, replay) => cards.push(el("div", { class: "card" },
    el("div", { class: "label" }, labelText), el("div", { class: "big" }, big),
    el("p", {}, text, replay ? [" ", el("a", { onclick: () => openReplay(replay) },
                                         "Open replay →")] : [])));

  const slide = get(S, "phase1", "slide", "dt2ms");
  const effortArticle = get(S, "phase3", "panda-effort", "dt1ms", "article:mujoco_cpu");
  if (slide) {
    const ours = slide.mujoco4?.metrics, art = slide["article:mujoco_cpu"]?.metrics;
    card("Article reproduced", mm(effortArticle?.path_diff_vs_mujoco4_m, 2),
      `Largest Panda link distance between the article's MuJoCo trace and MuJoCo 4.0 over the `
      + `0.5 s effort replay. Sliding block: ${num(ours?.travel_x_m * 1e3)} mm travel and `
      + `${num(ours?.peak_contact_force_N)} N peak here, against `
      + `${num(art?.travel_x_m * 1e3)} mm and ${num(art?.peak_contact_force_N)} N in the article.`,
      "panda-effort--dt1ms.json");
  }

  const versions = S["mujoco4-vs-311"] ?? {};
  const rows = Object.entries(versions);
  const same = rows.filter(([, r]) => (r.path_diff_m ?? 0) < 1e-6).length;
  const chain = versions["collision-chain/dt0.5ms"];
  const pairs = (route) => Object.fromEntries(chain?.contacts_per_frame[route] ?? []);
  card("MuJoCo 4.0 vs 3.11", `${same} / ${rows.length} identical`,
    `Variants that agree within 1 µm on identical MJCF. The rest all have a box face `
    + `resting on another surface, where 4.0 generates a different contact set. In the `
    + `collision chain the prism averages ${num(pairs("mujoco311")["prism-world"])} ground `
    + `contacts per step on 3.11 and ${num(pairs("mujoco4")["prism-world"])} on 4.0, and the `
    + `paths end ${mm(chain?.path_diff_m)} apart.`, "collision-chain--dt0.5ms.json");

  const b4 = get(S, "bench/mujoco4", "workloads", "panda", "runs");
  const b3 = get(S, "bench/mujoco311", "workloads", "panda", "runs");
  if (b4 && b3) {
    const pick = (runs, mode, threads) => runs.find((r) => r.mode === mode && r.threads === threads);
    const s1 = pick(b4, "mj_step", 1).world_steps_per_s / pick(b3, "mj_step", 1).world_steps_per_s;
    const s32 = pick(b4, "rollout", 32)?.world_steps_per_s / pick(b3, "rollout", 32)?.world_steps_per_s;
    card("MuJoCo 4.0 CPU speed", `${((s1 - 1) * 100).toFixed(0)}% faster`,
      `Contact-free Panda: ${fmtSI(pick(b4, "mj_step", 1).world_steps_per_s)} steps/s on one `
      + `thread vs ${fmtSI(pick(b3, "mj_step", 1).world_steps_per_s)} for 3.11, and `
      + `${((s32 - 1) * 100).toFixed(0)}% faster with mujoco.rollout on 32 threads.`);
  }

  const gpu = ["mjwarp4", "newton", "genesis"].map((r) => {
    const runs = (get(S, `bench/${r}`, "workloads", "panda", "runs") ?? []).filter((x) => !x.error);
    const best = runs.reduce((a, x) => (x.world_steps_per_s > (a?.world_steps_per_s ?? 0) ? x : a),
                             null);
    return [r, best];
  }).filter(([, b]) => b);
  if (gpu.length) {
    const top = gpu.reduce((a, b) => (b[1].world_steps_per_s > a[1].world_steps_per_s ? b : a));
    card("Peak GPU throughput", `${fmtSI(top[1].world_steps_per_s)} world-steps/s`,
      `${label(top[0])} on the contact-free Panda at ${top[1].worlds.toLocaleString()} worlds. `
      + gpu.map(([r, b]) => `${label(r)}: ${fmtSI(b.world_steps_per_s)}`).join(", ") + ".");
  }

  const det = get(S, "phase4", "push/dt1ms-w90mm-det/mjwarp4");
  const newtonDet = get(S, "phase4", "collision-chain/dt0.5ms-det/newton");
  if (det) {
    card("GPU determinism", `${mm(det.base_seed_spread_m)} → ${mm(det.seed_spread_m)}`,
      `MuJoCo-Warp 4.0 seed-to-seed spread in the push scene, with and without `
      + `DeterminismType.ALL. ${newtonDet?.status === "error"
        ? "Newton 1.7's deterministic mode fails to compile its own contact-conversion kernel."
        : ""}`);
  }

  const stack = get(S, "phase3", "stack");
  if (stack) {
    const routes = ["mujoco4", "mjwarp4", "newton", "genesis"];
    const offsets = Object.keys(stack).filter((v) => /^dt1ms-offset\d+mm$/.test(v))
      .sort((a, b) => parseFloat(a.match(/\d+/g)[1]) - parseFloat(b.match(/\d+/g)[1]));
    const survive = offsets.filter((v) => routes.every((r) => stack[v][r]?.metrics?.stack_survives));
    const last = survive.at(-1)?.match(/offset(\d+)mm/)[1];
    card("Stacking edge", `survives to ${last} mm`,
      `All engines agree on every nominal offset. Under load the hand lags its reference by about `
      + `4 mm, so 22 mm lands near 18 mm and stands, and 24 mm passes the 20 mm static `
      + `edge and topples.`, "stack--dt1ms-offset24mm.json");
  }

  const sparse = Object.entries(S.phase4 ?? {}).filter(([k]) => k.includes("jac-sparse"));
  if (sparse.length) {
    const worst = Math.max(...sparse.map(([, r]) => r.path_diff_vs_base_m ?? 0));
    card("mujoco#3767 check", worst < 1e-9 ? "not triggered" : mm(worst),
      `A forced sparse Jacobian vs dense under the Newton solver, on 3.11 and 4.0.0, for `
      + `${sparse.length / 2} scenes. The largest difference is ${worst.toExponential(1)} m.`);
  }

  const disc = get(S, "phase4", "slide/dt2ms-discrete/mujoco4");
  if (disc) {
    const a = disc.base_metrics.peak_contact_force_N, b = disc.metrics.peak_contact_force_N;
    card("New in 4.0: discrete integrator", `${num(a)} → ${num(b)} N`,
      `Peak sliding-block force with Euler and with integrator="discrete", a `
      + `${((1 - b / a) * 100).toFixed(0)}% smaller start-of-slide spike. IPC contacts give the `
      + `same result in these scenes, and speculative contacts change nothing at default margins.`);
  }
  $("#findings").replaceChildren(...cards);
}

// --- replay --------------------------------------------------------------------------

function buildSceneList() {
  const box = $("#scene-list");
  box.replaceChildren();
  for (const [phase, scenes] of PHASES) {
    const group = el("div", { class: "scene-group" }, el("div", { class: "muted" }, phase));
    for (const r of INDEX.replays.filter((x) => scenes.includes(x.scene))) {
      const v = r.variant.replace(/^dt/, "").replace("-", " · ");
      group.append(el("button", { class: "scene-btn", "data-file": r.file,
                                  onclick: () => openReplay(r.file) },
                      r.title, " ", el("small", {}, v)));
    }
    box.append(group);
  }
}

let visible = [];

function buildRouteList() {
  const box = $("#route-list");
  box.replaceChildren(...Object.keys(current.routes).map((name) => {
    const row = el("div", { class: `route ${visible.includes(name) ? "" : "off"}`,
                            onclick: () => {
                              visible = visible.includes(name) ? visible.filter((n) => n !== name)
                                : Object.keys(current.routes).filter((n) => visible.includes(n)
                                                                       || n === name);
                              buildRouteList();
                              viewer.setVisible(visible);
                              drawPlot();
                            } },
                   el("span", { class: "chip", style: `background:${COLORS[name]}` }),
                   label(name));
    return row;
  }));
}

async function openReplay(file, routes) {
  pending = file;
  showTab("replay");
  document.querySelectorAll(".scene-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.file === file));
  const replay = await loadReplay(file);
  if (pending !== file) return;
  pending = null;
  current = { ...replay, file };
  const names = Object.keys(current.routes);
  visible = routes?.filter((r) => names.includes(r))
    ?? names.filter((n) => !n.startsWith("article:") || names.length < 3);
  viewer.load(current, COLORS);
  viewer.setVisible(visible);
  buildRouteList();
  buildSignals();
  playT = 0;
  seek(0);
  sceneMetrics();
}

function frameAt(t) {
  const d = Object.values(current.routes)[0];
  let lo = 0, hi = d.frames - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (d.t[mid] <= t + 1e-9) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function seek(t) {
  const d = Object.values(current.routes)[0];
  playT = Math.max(0, Math.min(t, d.t[d.frames - 1]));
  const f = frameAt(playT);
  viewer.frame(f);
  $("#scrub").value = String(Math.round((1000 * f) / Math.max(1, d.frames - 1)));
  $("#time").textContent = `${playT.toFixed(3)} s`;
  plot?.redraw(false, false);
}

function buildSignals() {
  const sel = $("#signal");
  const opts = [["delta", "Largest COM distance from MuJoCo 4.0, all bodies"]];
  const d0 = Object.values(current.routes)[0];
  current.bodies.forEach((b, i) => {
    let moved = 0;
    for (let f = 1; f < d0.frames; f++) {
      const o = (f * d0.nb + i) * 3;
      moved = Math.max(moved, Math.abs(d0.com[o] - d0.com[i * 3]) + Math.abs(d0.com[o + 2]
        - d0.com[i * 3 + 2]));
    }
    if (moved < 1e-4 && !["cube", "upper", "support"].includes(b)) return;
    for (const [k, txt] of [["x", "COM x"], ["y", "COM y"], ["z", "COM z"],
                            ["fz", "contact force z"], ["fn", "|contact force|"]]) {
      opts.push([`${b}|${k}`, `${b}: ${txt}`]);
    }
  });
  (current.joints ?? []).forEach((j, i) => opts.push([`q|${i}`, `joint ${j}`]));
  sel.replaceChildren(...opts.map(([v, t]) => el("option", { value: v }, t)));
  const want = DEFAULT_SIGNAL[current.scene];
  sel.value = opts.some(([v]) => v === want) ? want : opts[0][0];
  drawPlot();
}

function signal(name, key) {
  const d = current.routes[name];
  const out = new Array(d.frames);
  if (key === "delta") {
    const base = current.routes.mujoco4;
    if (!base || name === "mujoco4") return null;
    for (let f = 0; f < d.frames; f++) {
      let m = 0;
      for (let b = 0; b < d.nb; b++) {
        const o = (f * d.nb + b) * 3;
        m = Math.max(m, Math.hypot(d.com[o] - base.com[o], d.com[o + 1] - base.com[o + 1],
                                   d.com[o + 2] - base.com[o + 2]));
      }
      out[f] = m * 1e3;
    }
    return out;
  }
  const [what, k] = key.split("|");
  if (what === "q") {
    if (!d.q) return null;
    for (let f = 0; f < d.frames; f++) out[f] = d.q[f * d.nq + Number(k)];
    return out;
  }
  const b = current.bodies.indexOf(what);
  for (let f = 0; f < d.frames; f++) {
    const o = (f * d.nb + b) * 3;
    if (k === "fz") out[f] = d.force[o + 2];
    else if (k === "fn") out[f] = Math.hypot(d.force[o], d.force[o + 1], d.force[o + 2]);
    else out[f] = d.com[o + "xyz".indexOf(k)] * 1e3;
  }
  return out;
}

function drawPlot() {
  const key = $("#signal").value;
  const unit = key === "delta" ? "distance [mm]" : key.startsWith("q|") ? "q [rad or m]"
    : key.endsWith("|fz") || key.endsWith("|fn") ? "force [N]" : "position [mm]";
  const t = Array.from(Object.values(current.routes)[0].t);
  const series = visible.map((n) => ({ label: label(n), color: COLORS[n], values: signal(n, key) }))
    .filter((s) => s.values).map((s) => ({ ...s, values: s.values.slice(0, t.length) }));
  $("#signal-note").textContent = key === "delta" && !current.routes.mujoco4
    ? "no MuJoCo 4.0 trace for this replay" : "";
  plot = timePlot($("#plot"), { t, series, unit, getT: () => playT, onSeek: seek });
}

function table(headers, rows, cls = "") {
  return el("div", { class: "tbl-wrap" }, el("table", { class: cls },
    el("thead", {}, el("tr", {}, headers.map((h) => el("th", {}, h)))),
    el("tbody", {}, rows)));
}

function sceneMetrics() {
  const phase = PHASES.findIndex(([, s]) => s.includes(current.scene)) + 1;
  const rows = get(SUMMARY, `phase${phase}`, current.scene, current.variant) ?? {};
  const keys = INDEX.headline[current.scene] ?? [];
  $("#scene-metrics").replaceChildren(el("h3", {}, "Metrics for this variant"), table(
    ["engine", ...keys, "COM path vs MuJoCo 4.0", "seed spread"],
    Object.entries(rows).map(([r, row]) => el("tr", {},
      el("td", {}, el("span", { class: "chip-inline", style: `color:${COLORS[r]}` }, "■ "),
         label(r)),
      keys.map((k) => el("td", {}, num(row.metrics?.[k]))),
      el("td", {}, mm(row.path_diff_vs_mujoco4_m)), el("td", {}, mm(row.seed_spread_m)))),
  ));
}

// --- versions, sensitivity -----------------------------------------------------------

function replayFor(scene, variant) {
  return INDEX.replays.find((r) => r.scene === scene && r.variant === variant)?.file;
}

function versionsTable() {
  const rows = Object.entries(SUMMARY["mujoco4-vs-311"] ?? {}).map(([key, r]) => {
    const [scene, variant] = key.split("/");
    const old = Object.fromEntries(r.contacts_per_frame.mujoco311);
    const neu = Object.fromEntries(r.contacts_per_frame.mujoco4);
    const changed = [...new Set([...Object.keys(old), ...Object.keys(neu)])]
      .filter((p) => Math.abs((old[p] ?? 0) - (neu[p] ?? 0)) > 0.005)
      .map((p) => `${p} ${(old[p] ?? 0).toFixed(2)}→${(neu[p] ?? 0).toFixed(2)}`);
    const file = replayFor(scene, variant);
    const diff = r.path_diff_m ?? 0;
    return el("tr", {},
      el("td", {}, key),
      el("td", { class: diff < 1e-6 ? "pass" : diff < 1e-3 ? "warn" : "fail" }, mm(diff)),
      el("td", {}, r.first_divergence_1um_s == null ? "—" : `${r.first_divergence_1um_s} s`),
      el("td", { class: "l" }, changed.join("; ") || "—"),
      el("td", {}, file ? el("button", { class: "link",
                                          onclick: () => openReplay(file, ["mujoco311", "mujoco4"]) },
                             "overlay") : ""));
  });
  $("#versions-table").replaceChildren(table(
    ["scene / variant", "path difference", "diverges at", "contacts per frame, 3.11 → 4.0", ""],
    rows));
}

function sensitivityTable() {
  const groups = {};
  for (const [key, r] of Object.entries(SUMMARY.phase4 ?? {})) {
    const [scene, variant, route] = key.split("/");
    (groups[scene] ??= []).push({ variant, route, ...r });
  }
  const rows = [];
  for (const [scene, items] of Object.entries(groups)) {
    rows.push(el("tr", { class: "group" }, el("td", { colspan: 7 }, scene)));
    for (const r of items) {
      const changed = r.metrics ? Object.entries(r.metrics)
        .filter(([k, v]) => JSON.stringify(v) !== JSON.stringify(r.base_metrics[k]))
        .map(([k, v]) => `${k}: ${num(r.base_metrics[k])} → ${num(v)}`) : [r.error ?? ""];
      rows.push(el("tr", {},
        el("td", {}, r.variant), el("td", {}, label(r.route)),
        el("td", { class: r.status === "ok" ? "" : "fail" }, r.status),
        el("td", {}, mm(r.path_diff_vs_base_m)), el("td", {}, mm(r.seed_spread_m)),
        el("td", {}, mm(r.base_seed_spread_m)),
        el("td", { class: "l" }, changed.join("; ").slice(0, 400) || "—")));
    }
  }
  $("#sensitivity-table").replaceChildren(table(
    ["variant", "engine", "status", "path vs baseline", "seed spread", "baseline spread",
     "changed metrics"], rows));
}

// --- runtime -------------------------------------------------------------------------

function runtime(workload) {
  document.querySelectorAll("#workload button").forEach((b) =>
    b.classList.toggle("active", b.dataset.w === workload));
  const gpu = ["mjwarp4", "newton", "genesis"];
  const worlds = [...new Set(gpu.flatMap((r) =>
    (get(SUMMARY, `bench/${r}`, "workloads", workload, "runs") ?? []).map((x) => x.worlds)))]
    .sort((a, b) => a - b);
  const series = (field) => gpu.map((r) => {
    const runs = get(SUMMARY, `bench/${r}`, "workloads", workload, "runs") ?? [];
    return { label: label(r), color: COLORS[r],
             values: worlds.map((n) => runs.find((x) => x.worlds === n && !x.error)?.[field] ?? null) };
  });
  logPlot($("#bench-throughput"), { x: worlds, series: series("world_steps_per_s"),
                                    xlabel: "worlds", ylabel: "world-steps/s" });
  logPlot($("#bench-rtf"), { x: worlds, series: series("rtf_per_world"), xlabel: "worlds",
                             ylabel: "× real time per world" });

  const cpuRows = [];
  const r3 = get(SUMMARY, "bench/mujoco311", "workloads", workload, "runs") ?? [];
  const r4 = get(SUMMARY, "bench/mujoco4", "workloads", workload, "runs") ?? [];
  for (const a of r4) {
    const b = r3.find((x) => x.mode === a.mode && x.threads === a.threads);
    const speed = b ? a.world_steps_per_s / b.world_steps_per_s : null;
    cpuRows.push(el("tr", {},
      el("td", {}, a.mode === "rollout" ? `rollout, ${a.worlds} worlds` : "mj_step, 1 world"),
      el("td", {}, a.threads), el("td", {}, b ? fmtSI(b.world_steps_per_s) : "—"),
      el("td", {}, fmtSI(a.world_steps_per_s)),
      el("td", { class: speed > 1.05 ? "pass" : speed < 0.95 ? "fail" : "" },
         speed ? `${speed.toFixed(2)}×` : "—")));
  }
  $("#bench-cpu").replaceChildren(table(
    ["mode", "threads", "3.11 world-steps/s", "4.0 world-steps/s", "4.0 / 3.11"], cpuRows));

  const gpuRows = [];
  for (const r of gpu) {
    const w = get(SUMMARY, `bench/${r}`, "workloads", workload);
    if (!w) continue;
    for (const x of w.runs) {
      gpuRows.push(el("tr", {}, el("td", {}, label(r)), el("td", {}, x.worlds.toLocaleString()),
        el("td", {}, x.error ? "—" : fmtSI(x.world_steps_per_s)),
        el("td", {}, x.error ? "—" : x.rtf_per_world.toFixed(3)),
        el("td", {}, x.gpu_memory_mib ? `${x.gpu_memory_mib.toLocaleString()} MiB` : "—"),
        el("td", {}, x.setup_s != null ? `${x.setup_s.toFixed(1)} s` : "—"),
        el("td", { class: x.error ? "fail l" : "l" }, x.error ? x.error.slice(0, 90) : "ok")));
    }
    const e = w.equivalence ?? {};
    gpuRows.push(el("tr", {}, el("td", {}, ""), el("td", { class: "l", colspan: 6 },
      `batch equivalence (max |Δq| over 1,000 steps): repeat n=1 ${num(e.repeat_n1_max_dq)}, `
      + `n=128 vs n=1 ${num(e.batch_n128_vs_n1_max_dq)}`)));
  }
  $("#bench-gpu").replaceChildren(table(
    ["engine", "worlds", "world-steps/s", "rtf/world", "GPU memory", "setup", "status"], gpuRows));
}

// --- audit ---------------------------------------------------------------------------

function audit() {
  const out = [];
  for (const model of ["panda", "panda_article"]) {
    const c = SUMMARY[`audit/${model}/compare`];
    if (!c) continue;
    const keys = Object.keys(c.tolerances);
    out.push(el("h3", {}, `${model} vs MuJoCo 4.0`), table(
      ["route", ...keys, "status"],
      [el("tr", {}, el("td", { class: "muted" }, "tolerance"),
          keys.map((k) => el("td", { class: "muted" }, num(c.tolerances[k]))), el("td", {}, "")),
       ...c.routes.map((r) => el("tr", {}, el("td", {}, r.route),
         keys.map((k) => el("td", { class: r.max_abs_diff[k] > c.tolerances[k] ? "fail" : "" },
                            num(r.max_abs_diff[k]))),
         el("td", { class: r.fail.length ? "fail" : "pass" },
            r.fail.length ? r.fail.join(", ") : "pass")))]));
  }
  $("#audit-tables").replaceChildren(...out);
  const envs = Object.entries(SUMMARY).filter(([k]) => k.startsWith("env/"));
  $("#env-table").replaceChildren(table(["route", "packages", "GPU"], envs.map(([k, e]) =>
    el("tr", {}, el("td", {}, k.slice(4)),
       el("td", { class: "l" }, Object.entries(e.packages ?? {})
         .filter(([p]) => !["simbench", "numpy"].includes(p))
         .map(([p, v]) => `${p} ${v}`).join(", ")),
       el("td", { class: "l" }, (e.gpus ?? []).map((g) => g.name ?? JSON.stringify(g)).join(", "))))));
}

// --- boot ----------------------------------------------------------------------------

async function main() {
  [INDEX, SUMMARY] = await Promise.all([json("data/index.json"), json("data/summary.json")]);
  viewer = new Viewer($("#viewer"));
  viewer.routeLabel = label;
  document.querySelectorAll("nav button").forEach((b) =>
    b.addEventListener("click", () => showTab(b.dataset.tab)));
  buildSceneList();
  findings();
  versionsTable();
  sensitivityTable();
  audit();

  const workloads = ["panda", "push", "slide", "drop"];
  $("#workload").replaceChildren(...workloads.map((w) =>
    el("button", { "data-w": w, onclick: () => runtime(w) }, w)));
  $("#view-mode").addEventListener("click", (e) => {
    const mode = e.target.dataset.mode;
    if (!mode) return;
    document.querySelectorAll("#view-mode button").forEach((b) =>
      b.classList.toggle("active", b === e.target));
    viewer.mode = mode;
    viewer.setVisible(visible);
  });
  $("#show-forces").addEventListener("change", (e) => { viewer.showForces = e.target.checked;
                                                        seek(playT); });
  $("#show-trails").addEventListener("change", (e) => { viewer.showTrails = e.target.checked; });
  $("#force-scale").addEventListener("change", (e) => { viewer.forceScale = +e.target.value;
                                                        seek(playT); });
  $("#signal").addEventListener("change", drawPlot);
  $("#scrub").addEventListener("input", (e) => {
    const d = Object.values(current.routes)[0];
    seek(d.t[Math.round((e.target.value / 1000) * (d.frames - 1))]);
  });
  const toggle = () => { playing = !playing; $("#play").textContent = playing ? "❚❚" : "▶"; };
  $("#play").addEventListener("click", toggle);
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && $("#replay").classList.contains("active")) {
      e.preventDefault();
      toggle();
    }
  });
  let last = performance.now();
  const tick = (now) => {
    const dt = (now - last) / 1000;
    last = now;
    if (playing && current) {
      const d = Object.values(current.routes)[0];
      let t = playT + dt * Number($("#speed").value);
      if (t > d.t[d.frames - 1]) t = 0;
      seek(t);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  const [tab, file] = location.hash.slice(1).split(":");
  if (file) openReplay(file);
  else showTab(tab || "overview");
}

main();
