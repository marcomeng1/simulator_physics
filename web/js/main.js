import { json, loadReplay, setCacheBust } from "./data.js";
import { bars, dots, fmt, legend, lines, matrix } from "./charts.js";
import { fmtSI, logPlot, timePlot } from "./plots.js";
import { frameAt, Viewer } from "./viewer.js";

const COLORS = {
  mujoco311: "#9aa5b1", mujoco4: "#4fb3ff", mjwarp4: "#3fcf8e", newton: "#b48cff",
  "newton-hydro": "#f472b6", genesis: "#ff9f43", "article:mujoco_cpu": "#c9d1d9",
  "article:newton_mjwarp": "#d6b8ff", "article:genesis": "#ffd29b", "article:physx": "#ff7b9c",
  "article:mujoco_warp": "#a8ecc8",
};
for (const k of Object.keys(COLORS)) if (k.startsWith("article:")) COLORS[`${k}#2`] = COLORS[k];
const GPU = ["mjwarp4", "newton", "genesis"];
const ROUTES = ["mujoco311", "mujoco4", ...GPU];
// Side-by-side / engine list order: Newton hydro sits beside Newton point contacts.
const ROUTE_DISPLAY = ["mujoco311", "mujoco4", "mjwarp4", "newton", "newton-hydro", "genesis"];

function orderRoutes(names) {
  const set = new Set(names);
  const main = ROUTE_DISPLAY.filter((n) => set.has(n));
  const rest = names.filter((n) => !ROUTE_DISPLAY.includes(n) && !n.startsWith("article:"));
  const articles = names.filter((n) => n.startsWith("article:"));
  return [...main, ...rest, ...articles];
}

function defaultVisible(names) {
  // Show the five engines plus Newton hydro when present; hide article traces by default.
  return orderRoutes(names).filter((n) => !n.startsWith("article:"));
}
// Plan summaries holding per-scene variant tables, searched in order.
const PLANS = ["phase1", "phase2", "phase3", "phase6", "plastic", "mj4new", "patch", "terrain"];
const DEFAULT_SIGNAL = {
  "hinge-plastic": "q|0",
  slide: "cube|fz", drop: "cube|fz", hinge: "q|0", "collision-chain": "delta",
  "panda-effort": "delta", "panda-tracking": "delta", grasp: "cube|z", stack: "upper|z",
  push: "cube|x", sweep: "delta", insertion: "peg|z", "ball-compress": "plate_hi|fz",
  "ball-grasp": "ball|z", "card-catch": "card|z", tumble: "handle|spin", sheet: "ball|z",
  twist: "q|2", "g1-walk": "pelvis|z",
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
// Per-route rows of one scene variant, from whichever plan summary ran it.
const variantRows = (scene, variant) => {
  for (const p of PLANS) {
    const rows = get(SUMMARY, p, scene, variant);
    if (rows) return rows;
  }
  return {};
};
const metric = (scene, variant, route, key) => get(variantRows(scene, variant), route, "metrics", key);
const fx = (v, d = 1) => (v == null ? "—" : String(+(+v).toFixed(d)));

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

const benchRun = (route, mode, threads, workload = "panda") =>
  (get(SUMMARY, `bench/${route}`, "workloads", workload, "runs") ?? [])
    .find((r) => r.mode === mode && r.threads === threads);

function gpuPeaks() {
  return GPU.map((r) => {
    const runs = (get(SUMMARY, `bench/${r}`, "workloads", "panda", "runs") ?? [])
      .filter((x) => !x.error);
    return [r, runs.reduce((a, x) => (x.world_steps_per_s > (a?.world_steps_per_s ?? 0) ? x : a),
                           null)];
  }).filter(([, b]) => b).sort((a, b) => b[1].world_steps_per_s - a[1].world_steps_per_s);
}

function insertionTally(route) {
  let p = 0, n = 0;
  for (const [v, rows] of Object.entries(get(SUMMARY, "phase6", "insertion") ?? {})) {
    if (v.endsWith("guided")) continue;
    for (const s of rows[route]?.seed_metrics ?? (rows[route] ? [rows[route].metrics] : [])) {
      n += 1;
      p += s.depth_reached ? 1 : 0;
    }
  }
  return [p, n];
}

const sweepCounts = (r) => {
  const row = get(SUMMARY, "phase6", "sweep", "dt1ms", r);
  return (row?.seed_metrics ?? (row ? [row.metrics] : [])).map((s) => s.collected);
};
const span = (xs) => (!xs.length ? "—" : Math.min(...xs) === Math.max(...xs) ? `${xs[0]}`
  : `${Math.min(...xs)}–${Math.max(...xs)}`);
const pct = (a, b) => (a && b ? Math.abs(1 - b / a) * 100 : null);
const plateLoad = (r, v = "dt0.05ms") => metric("ball-compress", v, r, "hold_plate_load_N");

// `tab#anchor`: switch tab, then scroll the anchor into view.
function goto(target) {
  const [tab, anchor] = target.split("#");
  showTab(tab);
  if (anchor) requestAnimationFrame(() => document.getElementById(anchor)
    ?.scrollIntoView({ behavior: "smooth", block: "start" }));
}
const link = (target, text = "See the test →") =>
  el("a", { class: "go", onclick: () => goto(target) }, text);
// "▶ replay" link for a scene variant, or nothing if it has no exported replay.
const replayButton = (scene, variant, text = "▶ Watch the replay") => {
  const file = INDEX.replays.find((r) => r.scene === scene && r.variant === variant)?.file;
  return file ? el("a", { class: "go replay-go", onclick: () => openReplay(file) }, text) : "";
};

// Card outcome on the thin plate: [text, tone].
const CARD_ROUTES = [["mujoco311", ""], ["mujoco4", ""], ["mujoco4", "-speculative"],
                     ["mjwarp4", ""], ["newton", ""], ["genesis", ""]];
const cardCols = () => CARD_ROUTES.map(([r, v]) => ({
  label: v ? "MuJoCo 4.0 + speculative" : label(r), color: COLORS[r] }));
const CARD_CASES = [["dt1ms-v0.3", "0.3 m/s"], ["dt1ms-v1", "1 m/s"], ["dt1ms-v3", "3 m/s"],
                    ["dt1ms-v10", "10 m/s"], ["dt2ms-v1", "1 m/s, 2 ms step"],
                    ["dt1ms-v1-w30", "1 m/s + 30 rad/s spin"],
                    ["dt1ms-v1-soft", "1 m/s, article's 10 ms contact"]];
function cardCell(variant, route, suffix) {
  const row = get(variantRows("card-catch", variant + suffix), route);
  if (!row) return suffix && route === "mujoco4" ? null : { text: "—", tone: "muted" };
  if (row.status === "error") return { text: "error", tone: "muted", title: row.error };
  const m = row.metrics;
  const tone = { caught: "good", tunneled: "bad", "off plate": "warn" }[m.outcome];
  const text = m.outcome === "tunneled" ? `through at ${fx(m.tunnel_at_s * 1e3, 0)} ms`
    : m.outcome;
  const file = replayFor("card-catch", variant + suffix);
  return { text, tone, onclick: file ? () => openReplay(file) : null };
}

function tumbleSeries(integ, routes = ROUTES) {
  const dts = integ === "rk4" ? [10, 5, 2, 1] : [2, 1];
  return routes.map((r) => ({
    label: `${label(r)}`, color: COLORS[r],
    points: dts.map((dt) => [dt, metric("tumble", `dt${dt}ms-${integ}`, r, "max_error_deg")])
      .filter(([, e]) => e != null),
  })).filter((s) => s.points.length);
}

function supportRows() {
  const st = (row) => (row ? row.status : null);
  const flexEl = (r) => {
    const row = get(variantRows("ball-compress", "dt0.05ms"), r);
    return !row ? ["not run", "muted"] : row.metrics?.completed ? ["works", "good"] : ["fails", "bad"];
  };
  const flexPl = (r) => {
    const row = get(variantRows("ball-compress", "dt0.05ms-y0.03"), r);
    if (!row) return ["not run", "muted"];
    if (st(row) === "error") return ["rejected", "muted"];
    return row.metrics?.completed ? ["works", "good"] : ["non-finite", "bad"];
  };
  const shell = (r) => {
    const row = get(variantRows("sheet", "dt0.1ms-yb0.002"), r);
    if (!row) return ["not run", "muted"];
    if (st(row) === "error") return ["rejected", "muted"];
    const m = row.metrics ?? {};
    if (m.invalid_at_s != null) return ["non-finite", "bad"];
    if (m.permanent_set_mm > 2) return ["works", "good"];
    return m.permanent_set_mm > 0.1 ? [`weak (${fx(m.permanent_set_mm, 1)} mm)`, "warn"]
      : ["no set", "bad"];
  };
  const joint = (r) => {
    const row = get(variantRows("hinge-plastic", "dt1ms-y5deg"), r);
    if (!row) return ["not run", "muted"];
    if (st(row) === "error") return ["rejected", "muted"];
    return Math.abs(row.metrics.final_error_deg) < 0.05 ? ["works", "good"]
      : ["silently ignored", "bad"];
  };
  const spec = (r) => {
    const row = get(variantRows("card-catch", "dt1ms-v1-speculative"), r);
    if (r === "mujoco4") return row?.metrics?.caught ? ["works", "good"] : ["fails", "bad"];
    if (r === "mujoco311") return ["absent", "muted"];
    return r === "mjwarp4" ? ["not implemented", "muted"] : ["no equivalent", "muted"];
  };
  const rk4 = (r) => {
    const e = metric("tumble", "dt10ms-rk4", r, "max_error_deg");
    const row = get(variantRows("tumble", "dt10ms-rk4"), r);
    if (row?.status === "error") return ["no RK4", "muted"];
    if (e == null) return ["not run", "muted"];
    if (e < 1e-3) return ["4th order", "good"];
    return r === "mujoco311" ? ["2nd order", "warn"] : [`${fx(e, 1)}° at 10 ms`, "bad"];
  };
  return [["Speculative contacts", spec, "mujoco4#card"], ["RK4 free-body rotation", rk4,
    "mujoco4#tumble"], ["Flex elasticity (3D)", flexEl, "scenarios#compress"],
          ["Joint plasticity", joint, "scenarios#plastic-hinge"],
          ["Solid plasticity (3D flex)", flexPl, "scenarios#plastic-ball"],
          ["Shell plasticity (2D flex)", shell, "scenarios#plastic-sheet"]];
}

const SHORT = { mujoco311: "3.11", mujoco4: "4.0", mjwarp4: "MJW 4.0", newton: "Newton",
                genesis: "Genesis" };

function supportMatrix(compact = false) {
  const rows = supportRows();
  return matrix(rows.map(([name, , target]) => (compact ? name
    : el("a", { class: "go", onclick: () => goto(target) }, name))),
  ROUTES.map((r) => ({ label: compact ? SHORT[r] : label(r), color: COLORS[r] })), (i, j) => {
    const [t, tone] = rows[i][1](ROUTES[j]);
    if (!compact) return { text: t, tone };
    return { text: { good: "✓", warn: "~", bad: "✕" }[tone] ?? "—", tone, title: t };
  });
}

const SYMBOL_KEY = "✓ works · ~ partial · ✕ fails or ignored · — not available. Hover for detail.";

function overview() {
  const S = SUMMARY;
  const versions = Object.values(S["mujoco4-vs-311"] ?? {});
  const same = versions.filter((r) => (r.path_diff_m ?? 0) < 1e-6).length;
  const st1 = (r) => benchRun(r, "mj_step", 1)?.world_steps_per_s;
  const s1 = st1("mujoco4") / st1("mujoco311");
  const peak = gpuPeaks()[0];
  const [p4, n4] = insertionTally("mujoco4");
  const sameIns = ROUTES.every((r) => insertionTally(r).join() === [p4, n4].join());
  const sweep = ROUTES.map((r) => [r, sweepCounts(r)]).filter(([, c]) => c.length);
  const rk = (r, dt) => metric("tumble", `dt${dt}ms-rk4`, r, "max_error_deg");
  const eul = metric("tumble", "dt2ms-euler", "mujoco4", "max_error_deg");

  const W = 520;
  // Outcome per engine on the contact scenes, with no engine as the reference.
  const outcomes = () => {
    const L = (INDEX.ladder ?? []).filter((x) => ["grasp", "stack", "push", "insertion", "sweep",
      "card-catch", "twist", "g1-walk"].includes(x.scene));
    return matrix(L.map((x) => x.title), ROUTES.map((r) => ({ label: SHORT[r], color: COLORS[r] })),
      (i, j) => {
        const row = variantRows(L[i].scene, L[i].variant)[ROUTES[j]];
        if (!row) return { text: "—", tone: "empty" };
        const [key, f] = OUTCOME[L[i].scene];
        const m = row.metrics ?? {};
        const text = row.status === "error" ? "rejected" : f(m[key]);
        return { text: text.replace(" analytic", "").replace(" speed", "").replace(
          "falls through", "through"), tone: outcomeTone(L[i].scene, m, row) };
      });
  };

  const items = [
    {
      topic: "Contact is where engines differ",
      claim: "Without contact the engines agree; with contact, compare outcomes, not paths.",
      text: "Without contact, all five engines agree to micrometres. Add contact and trajectories "
        + "drift apart, by up to metres once objects topple. Task outcomes agree far more often.",
      chart: outcomes, go: "scenarios#ladder", replay: ["sweep", "dt1ms"],
    },
    {
      topic: "Upgrading 3.11 → 4.0",
      claim: "MuJoCo 3.11 → 4.0 is a safe upgrade.",
      replay: ["collision-chain", "dt0.5ms"],
      text: `A safe upgrade: ${same} of ${versions.length} scene variants are identical within `
        + `1 µm, and a contact-free Panda steps ${fx(s1, 2)}× faster on one CPU thread. The rest `
        + "differ in contacts on resting box faces.",
      chart: () => bars([{ label: "MuJoCo 3.11", bars: [{ value: st1("mujoco311"),
        color: COLORS.mujoco311, text: `${fmtSI(st1("mujoco311"))} steps/s` }] },
      { label: "MuJoCo 4.0", bars: [{ value: st1("mujoco4"), color: COLORS.mujoco4,
        text: `${fmtSI(st1("mujoco4"))} steps/s` }] }], { width: W, labelW: 110 }),
      go: "mujoco4#upgrade",
    },
    {
      topic: "New in 4.0: speculative contacts",
      claim: "A stiff contact catches a thin card on every engine; only speculative contacts "
        + "catch a fast one.",
      replay: ["card-catch", "dt1ms-v3-speculative"],
      text: "A 0.3 mm card on a 1 mm plate is caught at 0.3–1 m/s on all five engines when the "
        + "contact time constant is two timesteps; with the article's 10 ms it sinks past the "
        + "plate's mid-plane and falls through even at 0.3 m/s. From 2 mm of travel per step "
        + "(3 m/s at 1 ms) only MuJoCo 4.0 CPU's speculative contacts catch it.",
      chart: () => matrix(CARD_CASES.slice(0, 4).concat([CARD_CASES[5]]).map(([, t]) => t),
        CARD_ROUTES.map(([r, v]) => ({ label: v ? "4.0 + spec." : SHORT[r], color: COLORS[r] })),
        (i, j) => {
          const c = cardCell([...CARD_CASES.slice(0, 4), CARD_CASES[5]][i][0], ...CARD_ROUTES[j]);
          return c && { ...c, text: c.text.startsWith("through") ? "✕" : c.text === "caught" ? "✓"
            : "~", onclick: null };
        }),
      go: "mujoco4#card",
    },
    {
      topic: "New in 4.0: fourth-order RK4",
      claim: "For tumbling bodies, use RK4 on MuJoCo 4.0 CPU; GPU and Euler drift.",
      replay: ["tumble", "dt2ms-euler"],
      text: `RK4 rotation error at a 10 ms step drops ${fx(rk("mujoco311", 10) / rk("mujoco4", 10),
        0)}× (3.11 → 4.0), on the CPU only: MuJoCo-Warp stalls near `
        + `${fx(rk("mjwarp4", 2), 1)}°. Euler, the usual RL choice, is off by ${fx(eul, 1)}° `
        + "after 4 s at 2 ms.",
      chart: () => lines([...tumbleSeries("rk4", ["mujoco311", "mujoco4"]).map((s) => ({
        ...s, label: `${s.label} RK4` })),
      ...tumbleSeries("euler", ["mujoco4"]).map((s) => ({ ...s, label: "4.0 Euler",
        dash: "5 4", color: "#ff6b6b" }))],
      { logX: true, logY: true, xlabel: "timestep [ms]", ylabel: "max error [°]", width: W,
        height: 200 }),
      go: "mujoco4#tumble",
    },
    {
      topic: "New in 4.0: elasto-plasticity",
      claim: "Plasticity works on MuJoCo 4.0 CPU; other engines ignore it or fail.",
      replay: ["hinge-plastic", "dt1ms-y5deg"],
      text: "Joint, solid and shell plasticity all work on MuJoCo 4.0 CPU. MuJoCo-Warp 4.0 "
        + "handles plastic joints, keeps a much smaller set in a shell, and goes non-finite "
        + "when a solid yields. Newton and Genesis accept springyield and silently ignore it.",
      chart: () => el("div", {}, supportMatrix(true), el("p", { class: "muted" }, SYMBOL_KEY)),
      go: "mujoco4#support",
    },
    {
      topic: "Contact-rich tasks",
      claim: "Insertion results transfer between engines; crowded sweeps do not.",
      replay: ["sweep", "dt1ms"],
      text: `Tight insertion is decided by geometry: ${sameIns ? `every engine passes ${p4} of `
        + `${n4}` : "results differ"}. The crowded sweep is not: engines collect `
        + `${sweep.map(([r, c]) => span(c)).join(", ")} of 15 objects.`,
      chart: () => bars(sweep.map(([r, c]) => ({ label: label(r), bars: [{
        value: Math.max(...c), color: COLORS[r], text: `${span(c)} / 15` }] })),
      { width: W, labelW: 140, max: 15 }),
      go: "scenarios#sweep",
    },
    twistRatio("mujoco4", "dt1ms-disc") != null && {
      topic: "Twisting grips: point vs patch contact",
      claim: "Point contacts grip twisting parts too hard; hydroelastic patches come closer.",
      replay: ["twist", "dt1ms-disc-hydro-iw"],
      text: `Point contacts resist a pinch-and-twist ${fx(twistRatio("mujoco4", "dt1ms-disc"), 1)}–`
        + `${fx(twistRatio("mujoco4", "dt1ms-square"), 1)}× harder than uniform pressure allows, on `
        + "every engine. Newton's hydroelastic patches come closest, once a Newton contact-softness "
        + "bug is worked around.",
      chart: () => twistBars(TWIST_OVERVIEW, { width: W, labelW: 190 }),
      go: "scenarios#twist",
    },
    g1Rows().length && {
      topic: "Legged locomotion on terrain",
      claim: "A G1 walking policy transfers on flat ground, falls on glass, and fails on Genesis "
        + "soft ground.",
      replay: ["g1-walk", "dt2ms-sand-v0.6"],
      text: g1Summary(),
      chart: () => g1Matrix(0.6, true),
      go: "scenarios#legged",
    },
    peak && {
      topic: "GPU throughput",
      claim: "GPUs win only with thousands of parallel worlds.",
      text: `${label(peak[0])} reaches ${fmtSI(peak[1].world_steps_per_s)} world-steps/s on a `
        + "contact-free Panda. In contact, GPU runs are not bit-repeatable by default.",
      chart: () => bars(gpuPeaks().map(([r, b]) => ({ label: label(r), bars: [{
        value: b.world_steps_per_s, color: COLORS[r],
        text: `${fmtSI(b.world_steps_per_s)} @ ${b.worlds.toLocaleString()}` }] })),
      { width: W, labelW: 140 }),
      go: "runtime",
    },
  ].filter(Boolean);

  $("#takeaways").replaceChildren(...items.map((it, i) => el("details", { class: "takeaway" },
    el("summary", {}, el("span", { class: "topic" }, `${i + 1}. ${it.topic}`),
      el("span", { class: "claim" }, it.claim)),
    el("p", {}, it.text),
    el("div", { class: "mini" }, it.chart()),
    el("div", { class: "links" }, link(it.go), it.replay ? replayButton(...it.replay) : ""))));
  $("#expand-all").onclick = () => {
    const all = [...document.querySelectorAll("#takeaways details")];
    const open = !all.every((d) => d.open);
    all.forEach((d) => { d.open = open; });
    $("#expand-all").textContent = open ? "Collapse all" : "Expand all";
  };

  const advice = [
    ["Legged locomotion", "Flat and soft terrains transfer between the MuJoCo engines and "
      + "Newton, not to Genesis. Randomise friction well below 0.5 and contact softness.",
    "scenarios#legged", ["g1-walk", "dt2ms-glass-v0.6"]],
    ["Grasping and pick-and-place", "Lift, stack and push outcomes agree on all five engines. "
      + "Train on GPU (MuJoCo-Warp or Newton), spot-check on MuJoCo CPU.", "scenarios#ladder", ["grasp", "dt1ms"]],
    ["Insertion and assembly", "Geometry decides, so any engine works; model the chamfer and "
      + "clearance faithfully rather than tuning contact.", "scenarios#insertion", ["insertion", "dt1ms-c100-tilted-guided"]],
    ["Clutter, bins and sweeping", "Outcomes differ by engine and GPU runs are not repeatable. "
      + "Evaluate on two engines and several seeds before trusting a success rate.",
    "scenarios#sweep", ["sweep", "dt1ms"]],
    ["Screwing, knobs, in-hand turning", "Point contacts over-resist twist. Randomise friction "
      + "downwards, or use Newton's hydroelastic contact with MuJoCo's invweight0.",
    "scenarios#twist", ["twist", "dt1ms-disc-hydro-iw"]],
    ["Thin parts (cards, sheet metal)", "Set the contact time constant to two timesteps and "
      + "keep v·dt below half the combined thickness; beyond that, MuJoCo 4.0 CPU with "
      + "speculative contacts.", "mujoco4#card", ["card-catch", "dt1ms-v3-speculative"]],
    ["Throwing, flying, tumbling", "Use RK4 on MuJoCo 4.0 CPU or a smaller step; Euler and "
      + "implicitfast drift degrees within seconds.", "mujoco4#tumble", ["tumble", "dt10ms-rk4"]],
    ["Soft and plastic objects", "MuJoCo 4.0 (CPU or MuJoCo-Warp) with a stiff pad contact "
      + "(time constant 2 ms), and check tetrahedron validity; Newton and Genesis need their "
      + "own soft-body solvers.", "scenarios#deform", ["ball-grasp", "dt0.05ms-e2N-stiff"]],
  ];
  $("#advice").replaceChildren(...advice.map(([h, t, go, rp]) => el("li", {},
    el("strong", {}, h), el("span", {}, t), " ", link(go, "test →"),
    rp ? replayButton(...rp) : "")));
  $("#beyond").replaceChildren(...beyond().map(([t, go, rp]) => el("li", {}, t, " ",
    link(go, "test →"), rp ? replayButton(...rp) : "")));
}

function beyond() {
  const out = [];
  const [l3, l4, legacy] = [plateLoad("mujoco311"), plateLoad("mujoco4"),
                            plateLoad("mujoco4", "dt0.05ms-legacy")];
  if (legacy) {
    out.push([`The 4.0 ball is ${fx((1 - l4 / l3) * 100, 0)}% softer than on 3.11. The `
      + `changelog's legacy constants recover ${fx(((legacy - l4) / (l3 - l4)) * 100, 0)}% of `
      + "the gap, so it is the elastic-constant fix.", "scenarios#compress",
      ["ball-compress", "dt0.05ms"]]);
  }
  const cpuGpu = pct(plateLoad("mujoco4"), plateLoad("mjwarp4"));
  if (cpuGpu != null) {
    out.push([`MuJoCo CPU and MuJoCo-Warp now differ by ${fx(cpuGpu, 0)}% in soft-ball plate `
      + `load (3.x: ${fx(pct(plateLoad("article:mujoco_cpu"), plateLoad("article:mujoco_warp")),
        0)}%).`, "scenarios#compress", ["ball-compress", "dt0.05ms"]]);
  }
  const g = (r, k) => metric("ball-grasp", "dt0.05ms-e2N", r, k);
  if (g("mujoco4", "invalid_frames")) {
    const ok = ["mujoco311", "mujoco4", "mjwarp4"].every((r) =>
      metric("ball-grasp", "dt0.05ms-e2N-stiff", r, "completed"));
    out.push(["With the article's 10 ms pad contact, MuJoCo 4.0 CPU's softer (corrected) ball "
      + `lets a pad push a surface vertex through its tetrahedron for ${g("mujoco4",
        "invalid_frames")} frames at first touch, failing the validity rule. `
      + (ok ? "A 2 ms contact keeps all three MuJoCo builds valid at 2 N." : ""),
    "scenarios#grasp", ["ball-grasp", "dt0.05ms-e2N"]]);
  }
  out.push(["At yield strain 0.03 the plastic ball slumps under its own weight before it is "
    + "grasped: its yield stress (600 Pa) is about its weight pressure.",
  "scenarios#plastic-ball", ["ball-compress", "dt0.05ms-y0.03"]]);
  const sh = metric("sheet", "dt0.1ms-yb0.002", "mujoco4", "permanent_set_mm");
  const shw = metric("sheet", "dt0.1ms-yb0.002", "mjwarp4", "permanent_set_mm");
  if (sh != null && shw != null) {
    out.push([`A 1 mm shell with bending yield 0.002 keeps a ${fx(sh, 1)} mm tip set after a `
      + `15 mm press on 4.0 CPU and ${fx(shw, 1)} mm on MuJoCo-Warp; at 0.004 it springs back.`,
    "scenarios#plastic-sheet", ["sheet", "dt0.1ms-yb0.002"]]);
  }
  const spin = get(variantRows("card-catch", "dt1ms-v1-w30-speculative"), "mujoco4", "metrics");
  if (spin && !spin.caught) {
    out.push(["Speculative contacts catch a flat card at up to 10 m/s, but not one spinning at "
      + "30 rad/s.", "mujoco4#card", ["card-catch", "dt1ms-v1-w30-speculative"]]);
  }
  const gi = metric("tumble", "dt2ms-implicitfast", "genesis", "max_error_deg");
  const mi = metric("tumble", "dt2ms-implicitfast", "mujoco4", "max_error_deg");
  if (gi != null && mi != null) {
    out.push([`Genesis's implicitfast tumbles accurately (${fx(gi, 4)}° at 2 ms) where MuJoCo's `
      + `implicitfast drifts ${fx(mi, 1)}°: the integrators share a name, not the result.`,
    "mujoco4#tumble", ["tumble", "dt2ms-implicitfast"]]);
  }
  const nw = twistRatio("newton", "dt1ms-disc");
  if (nw != null) {
    out.push([`Newton 1.7 compiles a hinged light body's body_invweight0 unlike MuJoCo, which `
      + `softens its contacts about 1,000×: the pinched puck slips at ${fx(nw * 100, 0)}% of the `
      + "analytic torque. Copying MuJoCo's values restores normal contact.", "scenarios#twist",
      ["twist", "dt1ms-disc"]]);
  }
  out.push(["Newton 1.7's add_mjcf ignores <option gravity>; scenes with non-default gravity "
    + "must pass it to the ModelBuilder.", "scenarios#plastic-hinge",
    ["hinge-plastic", "dt1ms-y5deg"]]);
  out.push(["MuJoCo-Warp's DeterminismType.ALL does not make the crowded sweep repeatable "
    + "(first difference at 0.111 s, 4 nm).", "scenarios#sweep",
    ["sweep", "dt1ms"]]);
  return out;
}

// --- replay --------------------------------------------------------------------------

function buildSceneList() {
  const box = $("#scene-list");
  box.replaceChildren();
  for (const [phase, scenes] of INDEX.ladder_groups ?? []) {
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
  const names = orderRoutes(Object.keys(current.routes));
  box.replaceChildren(...names.map((name) => {
    const row = el("div", { class: `route ${visible.includes(name) ? "" : "off"}`,
                            onclick: () => {
                              visible = visible.includes(name) ? visible.filter((n) => n !== name)
                                : orderRoutes(Object.keys(current.routes)).filter((n) =>
                                  visible.includes(n) || n === name);
                              buildRouteList();
                              viewer.setVisible(visible);
                              drawPlot();
                            } },
                   el("span", { class: "chip",
                                style: `background:${COLORS[name] ?? "#888"}` }),
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
  visible = routes?.length
    ? orderRoutes(routes.filter((r) => names.includes(r)))
    : defaultVisible(names);
  // Keep article traces available if the caller asked for a tiny compare set only.
  if (!routes?.length && names.length < 3) visible = orderRoutes(names);
  viewer.load(current, COLORS);
  viewer.setVisible(visible);
  buildRouteList();
  buildSignals();
  playT = 0;
  seek(0);
  sceneMethod();
  sceneMetrics();
}

function seek(t) {
  const d = Object.values(current.routes)[0];
  playT = Math.max(0, Math.min(t, d.t[d.frames - 1]));
  const f = frameAt(d.t, playT);
  viewer.seek(playT);
  $("#scrub").value = String(Math.round((1000 * f) / Math.max(1, d.frames - 1)));
  $("#time").textContent = `${playT.toFixed(3)} s`;
  plot?.redraw(false, false);
}

function buildSignals() {
  const sel = $("#signal");
  const opts = [["delta", "Spread: largest COM distance from the five-engine mean"]];
  const d0 = Object.values(current.routes)[0];
  const flexBodies = new Set(current.flex?.bodies ?? []);
  if (current.flex) opts.push(["ball|z", "deformable: vertex centroid z"]);
  current.bodies.forEach((b, i) => {
    if (flexBodies.has(i)) return;
    let moved = 0;
    for (let f = 1; f < d0.frames; f++) {
      const o = (f * d0.nb + i) * 3;
      moved = Math.max(moved, Math.abs(d0.com[o] - d0.com[i * 3]) + Math.abs(d0.com[o + 2]
        - d0.com[i * 3 + 2]));
    }
    if (moved < 1e-4 && !["cube", "upper", "support"].includes(b)) return;
    if (current.scene === "tumble") opts.push([`${b}|spin`, `${b}: spin axis · world y`]);
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
    const peers = ROUTES.map((r) => current.routes[r]).filter((p) => p?.nb === d.nb);
    if (peers.length < 2) return null;
    for (let f = 0; f < d.frames; f++) {
      let m = 0;
      const fp = peers.map((p) => frameAt(p.t, d.t[f]));
      for (let b = 0; b < d.nb; b++) {
        const o = (f * d.nb + b) * 3;
        const mean = [0, 1, 2].map((i) => peers.reduce((s, p, j) =>
          s + p.com[(fp[j] * d.nb + b) * 3 + i], 0) / peers.length);
        m = Math.max(m, Math.hypot(d.com[o] - mean[0], d.com[o + 1] - mean[1],
                                   d.com[o + 2] - mean[2]));
      }
      out[f] = m * 1e3;
    }
    return out;
  }
  const [what, k] = key.split("|");
  if (what === "ball") {
    const idx = current.flex.bodies;
    for (let f = 0; f < d.frames; f++) {
      let z = 0;
      for (const b of idx) z += d.com[(f * d.nb + b) * 3 + 2];
      out[f] = (z / idx.length) * 1e3;
    }
    return out;
  }
  if (what === "q") {
    if (!d.q) return null;
    for (let f = 0; f < d.frames; f++) out[f] = d.q[f * d.nq + Number(k)];
    return out;
  }
  const b = current.bodies.indexOf(what);
  for (let f = 0; f < d.frames; f++) {
    const o = (f * d.nb + b) * 3;
    if (k === "spin") {
      const q = (f * d.nb + b) * 4;
      const [x, z] = [d.quat[q + 1], d.quat[q + 3]];
      out[f] = 1 - 2 * (x * x + z * z);
    } else if (k === "fz") out[f] = d.force[o + 2];
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
  // Routes are sampled differently (the article's bundles at 10 ms): resample onto `t`.
  const series = visible.map((n) => ({ n, label: label(n), color: COLORS[n],
                                       values: signal(n, key) }))
    .filter((s) => s.values).map((s) => {
      const own = current.routes[s.n].t;
      return { ...s, values: t.map((x) => (x > own[own.length - 1] + 1e-6 ? null
                                           : s.values[frameAt(own, x)])) };
    });
  $("#signal-note").textContent = key === "delta" && !current.routes.mujoco4
    ? "no MuJoCo 4.0 trace for this replay" : "";
  plot = timePlot($("#plot"), { t, series, unit, getT: () => playT, onSeek: seek });
}

function table(headers, rows, cls = "") {
  return el("div", { class: "tbl-wrap" }, el("table", { class: cls },
    el("thead", {}, el("tr", {}, headers.map((h) => el("th", {}, h)))),
    el("tbody", {}, rows)));
}

function sceneMethod() {
  const box = $("#scene-method");
  if (!box) return;
  const method = current.params?.method;
  if (!method || current.scene !== "cloth-fold") {
    box.replaceChildren();
    return;
  }
  const card = method.cloth_card ?? {};
  const engines = method.engines ?? {};
  const fallback = {
    mujoco4: { model: "2D flex", solver: "MuJoCo flex elasticity" },
    mjwarp4: { model: "2D flex", solver: "MuJoCo-Warp flex elasticity" },
    newton: { model: "SolverVBD", solver: "newton.solvers.SolverVBD" },
    genesis: { model: "PBD.Cloth", solver: "genesis PBDSolver" },
  };
  const routeOrder = orderRoutes(Object.keys(current.routes));
  const engineRows = routeOrder.map((r) => {
    const s = current.routes[r]?.settings ?? {};
    const key = r.replace(/-hydro$/, "");
    const blurb = engines[r] ?? engines[key] ?? "—";
    const fb = fallback[key] ?? {};
    const model = s.cloth_model || s.cloth || fb.model || "—";
    const solver = s.cloth_solver || fb.solver || s.engine || "—";
    const drive = s.robot_drive || current.params?.robot_drive || "kinematic-tape";
    return el("tr", {},
      el("td", {}, el("span", { class: "chip-inline", style: `color:${COLORS[r] ?? "#aaa"}` },
         "■ "), label(r)),
      el("td", {}, model),
      el("td", {}, solver),
      el("td", {}, drive),
      el("td", { class: "wrap" }, blurb));
  });
  const size = card.size_x != null
    ? `${card.size_x}×${card.size_y ?? "?"} m`
    : `${card.size_m ?? "?"} m`;
  box.replaceChildren(
    el("h3", {}, "Fair comparison — controller & cloth stacks"),
    el("div", { class: "method-box" },
      el("p", {}, el("b", {}, "Controller: "), method.controller || ""),
      el("p", {}, el("b", {}, "Grasp: "), method.grasp || ""),
      el("p", {}, el("b", {}, "Shared cloth card: "),
         `ρ=${card.areal_density ?? "?"} kg/m², thickness=${card.thickness_m ?? "?"} m, `,
         `E=${card.young ?? "?"} Pa, damping=${card.damping ?? "?"}, `,
         `grid ${card.nx ?? "?"}×${card.ny ?? "?"} on ${size}`),
      table(["engine", "cloth model", "solver", "robot drive", "notes"], engineRows),
    ),
  );
}

function sceneMetrics() {
  const rows = { ...variantRows(current.scene, current.variant) };
  // Hydroelastic Newton is merged into the replay as "newton-hydro" from <variant>-hydro.
  if (current.routes["newton-hydro"]) {
    const hydro = variantRows(current.scene, `${current.variant}-hydro`).newton;
    if (hydro) rows["newton-hydro"] = hydro;
  }
  const keys = INDEX.headline[current.scene] ?? [];
  const order = [...ROUTES, "newton-hydro"].filter((r) => rows[r]);
  const extra = Object.keys(rows).filter((r) => !order.includes(r));
  $("#scene-metrics").replaceChildren(el("h3", {}, "Metrics for this variant"), table(
    ["engine", ...keys, "seed spread"],
    [...order, ...extra].map((r) => {
      const row = rows[r];
      return el("tr", {},
        el("td", {}, el("span", { class: "chip-inline", style: `color:${COLORS[r] ?? "#aaa"}` },
           "■ "), label(r)),
        keys.map((k) => el("td", {}, num(row.metrics?.[k]))),
        el("td", {}, mm(row.seed_spread_m)));
    }),
  ));
  contactNote();
}

// Yellow dots carry force; cool-blue dots are proximity reports with ~0 N (common on
// Newton's MuJoCo-Warp 3.15 mesh contact buffer before first touch).
function contactNote() {
  const note = $("#contact-note");
  if (!current) { note.textContent = ""; return; }
  let idle = 0, active = 0, newtonIdle = false, hydroIdle = false;
  for (const [name, r] of Object.entries(current.routes)) {
    const c = r.contacts;
    if (!c) continue;
    let routeIdle = 0, routeActive = 0;
    const n = c.start[c.start.length - 1];
    for (let j = 0; j < n; j++) {
      const f = Math.hypot(c.force[3 * j], c.force[3 * j + 1], c.force[3 * j + 2]);
      if (f > 1e-4) { active += 1; routeActive += 1; }
      else { idle += 1; routeIdle += 1; }
    }
    if (name === "newton" && routeIdle > routeActive) newtonIdle = true;
    if (name === "newton-hydro" && routeIdle > routeActive) hydroIdle = true;
  }
  if (!idle && !active) { note.replaceChildren(); return; }
  const parts = [
    el("div", {}, el("b", {}, "Contact points. "),
       "Yellow = active (nonzero force), drawn on top. ",
       "Blue at 50% opacity = inactive proximity (~0 N), kept so the buffer is visible without hiding yellow points."),
  ];
  if (newtonIdle) {
    parts.push(el("div", {}, el("b", {}, "Newton (point contacts). "),
      "MuJoCo-Warp often lists every peg–bore pair from t = 0 with zero force; those stay blue. ",
      "First yellow force matches the other engines."));
  }
  if (hydroIdle) {
    parts.push(el("div", {}, el("b", {}, "Newton hydroelastic. "),
      "SDF hydroelastic emits many nearby samples at 0 N (blue, translucent). ",
      "Only samples that carry force are yellow — that is the load-bearing patch."));
  }
  note.replaceChildren(...parts);
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

// --- scenarios -----------------------------------------------------------------------

const routeCell = (r) => el("td", {}, el("span", { class: "chip-inline",
                                                   style: `color:${COLORS[r]}` }, "■ "), label(r));
const openBtn = (scene, variant, routes) => {
  const file = replayFor(scene, variant);
  return el("td", {}, file ? el("button", { class: "link", onclick: () => openReplay(file, routes) },
                                "replay") : "");
};
const replayLink = (scene, variant, text = "Open replay →") => {
  const file = replayFor(scene, variant);
  return file ? el("p", {}, el("button", { class: "link", onclick: () => openReplay(file) }, text))
    : "";
};
const routeLegend = (routes) => legend(routes.map((r) => ({ label: label(r), color: COLORS[r] })));

function sweepTable() {
  const v = get(SUMMARY, "phase6", "sweep", "dt1ms") ?? {};
  const det = get(SUMMARY, "phase6", "sweep", "dt1ms-det", "mjwarp4");
  const order = [...ROUTES, "article:mujoco_cpu", "article:newton_mjwarp", "article:genesis",
                 "article:physx"];
  const runs = (row) => (row?.seed_metrics ?? (row ? [row.metrics] : [])).map((s) => s.collected);
  const rows = order.filter((r) => v[r]).map((r) => {
    const c = [...runs(v[r]), ...runs(v[`${r}#2`])];
    return { label: label(r), sub: r.startsWith("article") ? "article" : `${c.length} runs`,
             points: c.map((n, i) => ({ value: n, color: COLORS[r], title: `run ${i + 1}: ${n}` })) };
  });
  if (det) {
    rows.push({ label: label("mjwarp4"), sub: "DeterminismType.ALL",
                points: runs(det).map((n) => ({ value: n, color: COLORS.mjwarp4, shape: "square",
                                                title: `${n}` })) });
  }
  const peak = (r) => v[r]?.metrics?.peak_paddle_contact_N;
  $("#p6-sweep").replaceChildren(
    dots(rows.map((r) => ({ ...r, label: r.sub ? `${r.label} · ${r.sub}` : r.label })),
      { log: false, min: 0, max: 15, unit: "objects collected (of 15)", labelW: 300 }),
    bars(ROUTES.filter((r) => peak(r) != null).map((r) => ({ label: label(r),
      bars: [{ value: peak(r), color: COLORS[r], text: `${fx(peak(r), 0)} N` }] })),
    { unit: "peak paddle contact force [N]", labelW: 300, barH: 12 }),
    replayLink("sweep", "dt1ms", "Open the sweep replay →"));
}

function insertionTable() {
  const ins = get(SUMMARY, "phase6", "insertion") ?? {};
  // Hydro companions are folded into the parent replay as "newton-hydro", not matrix rows.
  const variants = Object.keys(ins).filter((v) => !v.endsWith("-hydro") && !v.endsWith("-fem"))
    .sort((a, b) => a.localeCompare(b));
  const cols = [...ROUTES, "newton-hydro", "article:mujoco_cpu"];
  $("#p6-insertion").replaceChildren(matrix(variants.map((v) => {
    const [, c, kind, guided] = v.match(/c(\d+)-(\w+)(-guided)?/);
    return `${+c / 1000} mm · ${kind}${guided ? " · guided" : ""}`;
  }), cols.map((r) => ({ label: label(r), color: COLORS[r] ?? "#aaa" })), (i, j) => {
    const route = cols[j];
    const row = route === "newton-hydro" ? ins[`${variants[i]}-hydro`]?.newton
      : ins[variants[i]]?.[route];
    if (!row) return null;
    const seeds = row.seed_metrics ?? [row.metrics];
    const pass = seeds.filter((s) => s.depth_reached).length;
    const file = replayFor("insertion", variants[i]);
    return { text: `${pass}/${seeds.length}`, tone: pass === seeds.length ? "good" : pass ? "warn"
      : "bad", title: `final depth ${fx(row.metrics.final_depth_mm, 2)} mm, peak contact `
      + `${fx(row.metrics.peak_contact_N, 1)} N`, onclick: file ? () => openReplay(file) : null };
  }, { corner: "clearance · approach" }),
  el("p", { class: "muted" }, "Cells: runs that reached 19 mm. Point-contact engines all jam on "
    + "the tilted 0.1 mm case and pass it with the guided controller. Newton hydro (when "
    + "present) can pass the same open-impedance tilt with ~0.3 N — see the discussion "
    + "above. Continuum soft peg/bore cases live under Deformable → Soft peg / bore."));
}

function softInsertionTable() {
  const rows = get(SUMMARY, "phase6", "soft-insertion") ?? {};
  const order = ["dt1ms-fem-peg", "dt1ms-fem-bore", "dt0.05ms-flex-peg",
                 "dt0.05ms-flex-peg-y0.05"];
  const variants = order.filter((v) => rows[v]).concat(
    Object.keys(rows).filter((v) => !order.includes(v)).sort());
  const cols = ["genesis", "mujoco4", "mjwarp4"];
  const titles = {
    "dt1ms-fem-peg": "A1 Genesis FEM peg / rigid bore",
    "dt1ms-fem-bore": "A2 rigid peg / Genesis FEM bore",
    "dt0.05ms-flex-peg": "M1 MuJoCo flex peg elastic (E=20 kPa)",
    "dt0.05ms-flex-peg-y0.05": "M2 MuJoCo flex peg yield 0.05",
  };
  const elBox = $("#def-soft-insertion");
  if (!elBox) return;
  elBox.replaceChildren(matrix(variants.map((v) => titles[v] ?? v),
    cols.map((r) => ({ label: label(r), color: COLORS[r] ?? "#aaa" })), (i, j) => {
      const row = rows[variants[i]]?.[cols[j]];
      if (!row) return null;
      const m = row.metrics ?? {};
      const file = replayFor("soft-insertion", variants[i]);
      const pass = m.depth_reached;
      return {
        text: pass ? `${fx(m.final_depth_mm, 1)} mm` : `${fx(m.final_depth_mm, 1)} mm`,
        tone: pass ? "good" : "bad",
        title: `hand |F| ${fx(m.peak_hand_force_N, 2)} N, |M| ${fx(m.peak_hand_moment_Nm, 3)} N·m`
          + (m.peak_vertex_disp_mm != null ? `, disp ${fx(m.peak_vertex_disp_mm, 2)} mm` : "")
          + (m.permanent_set_mm != null ? `, set ${fx(m.permanent_set_mm, 2)} mm` : ""),
        onclick: file ? () => openReplay(file) : null,
      };
    }, { corner: "material case" }),
  el("p", { class: "muted" }, "Cells: final depth (pass ≥ 19 mm). Title shows peak hand wrench "
    + "(contact force / moment about the Franka hand). Continuum cases are not merged into "
    + "the rigid peg-insertion matrix."));
}

function clothFoldTable() {
  const rows = get(SUMMARY, "phase6", "cloth-fold") ?? {};
  const variants = Object.keys(rows).sort();
  const cols = ["mujoco4", "mjwarp4", "newton", "genesis"];
  const elBox = $("#def-cloth-fold");
  if (!elBox) return;
  if (!variants.length) {
    elBox.replaceChildren(el("p", { class: "muted" }, "No cloth-fold runs in phase6 yet."));
    return;
  }
  elBox.replaceChildren(matrix(variants.map((v) => v.replace(/^dt[\d.]+ms-?/, "") || v),
    cols.map((r) => ({ label: label(r), color: COLORS[r] ?? "#aaa" })), (i, j) => {
      const row = rows[variants[i]]?.[cols[j]];
      if (!row) return null;
      const m = row.metrics ?? {};
      const file = replayFor("cloth-fold", variants[i]);
      const pass = m.passed;
      return {
        text: m.coverage == null ? "—" : `${fx(m.coverage * 100, 0)}%`,
        tone: pass ? "good" : "bad",
        title: `coverage ${fx((m.coverage ?? 0) * 100, 1)}%, slip ${fx(m.grasp_slip_mm, 1)} mm`,
        onclick: file ? () => openReplay(file) : null,
      };
    }, { corner: "variant" }),
  el("p", { class: "muted" }, "Cells: fraction of cloth vertices with y ≤ 0 after the fold "
    + "(pass ≥ 80%). Title shows grasp slip of the pinched patches. Open the replay for the "
    + "shared kinematic controller and per-engine cloth model (2D flex / VBD / PBD)."));
}

function tuningTable() {
  const rows = ROUTES.map((r) => [r, SUMMARY[`tuning/${r}`]]).filter(([, t]) => t).map(([r, t]) => {
    const ok = (trials) => (trials ?? []).filter((x) => x.all_passed).length;
    const base = "p1-d1-t3";
    return el("tr", {}, routeCell(r), el("td", {}, t.selected), el("td", {}, t.runner_up),
      el("td", {}, `${(t.margin_vs_runner_up * 100).toFixed(2)}%`),
      el("td", {}, `${ok(t.held_out[base])}/${t.held_out[base].length}`),
      el("td", {}, `${ok(t.held_out[t.selected])}/${t.held_out[t.selected].length}`));
  });
  rows.push(el("tr", {}, routeCell("article:mujoco_cpu"), el("td", {}, "p0.5-d0.7-t4"),
    el("td", { class: "muted" }, "—"), el("td", {}, "< 0.6%"), el("td", {}, "4/6"),
    el("td", {}, "4/6")));
  $("#p6-tuning").replaceChildren(table(
    ["engine", "selected", "runner-up", "effort margin", "held-out, baseline", "held-out, selected"],
    rows),
  el("p", { class: "muted" }, "p<stiffness>-d<damping>-t<push seconds>; baseline p1-d1-t3. "
    + "MuJoCo-Warp was not tuned (about 1.5 h). Every candidate passes all selection layouts, "
    + "so the pick is decided by effort."));
}

const OUTCOME = {
  hinge: ["max_reference_error_deg", (v) => `${fx(v, 4)}°`],
  tumble: ["max_error_deg", (v) => `${fx(v, 4)}°`],
  "panda-effort": ["hand_excursion_cm", (v) => `${fx(v, 1)} cm`],
  "panda-tracking": ["max_tracking_error_deg", (v) => `${fx(v, 3)}°`],
  slide: ["peak_contact_force_N", (v) => `${fx(v, 1)} N`],
  drop: ["peak_contact_force_N", (v) => `${fx(v, 1)} N`],
  "card-catch": ["outcome", (v) => (v === "tunneled" ? "falls through" : v)],
  "collision-chain": ["peak_horizontal_force_N", (v) => `${fx(v, 1)} N`],
  grasp: ["hold_lift_cm", (v) => `${fx(v, 2)} cm`],
  stack: ["stack_survives", (v) => (v ? "stands" : "falls")],
  push: ["passed_count", (v) => `${v}/3`],
  insertion: ["final_depth_mm", (v) => `${fx(v, 1)} mm`],
  "soft-insertion": ["final_depth_mm", (v) => `${fx(v, 1)} mm`],
  "cloth-fold": ["coverage", (v) => `${fx((v ?? 0) * 100, 0)}%`],
  sweep: ["collected", (v) => `${v}/15`],
  "ball-compress": ["hold_plate_load_N", (v) => `${fx(v, 2)} N`],
  "ball-grasp": ["hold_lift_mm", (v) => (v == null ? "invalid" : `${fx(v, 1)} mm`)],
  "hinge-plastic": ["final_angle_deg", (v) => `${fx(v, 2)}°`],
  sheet: ["permanent_set_mm", (v) => `${fx(v, 1)} mm set`],
  twist: ["slip_ratio", (v) => `${fx(v, 2)}× analytic`],
  "g1-walk": ["speed_ratio", (v) => `${fx(v, 2)}× speed`],
};
const OUTCOME_LABEL = {
  hinge: "error vs analytic", tumble: "RK4 error, 2 ms", "panda-effort": "hand excursion",
  "panda-tracking": "tracking error", slide: "peak contact force", drop: "peak contact force",
  "card-catch": "card at 1 m/s", "collision-chain": "peak horizontal force",
  grasp: "hold lift", stack: "20 mm offset stack", push: "objects through channel",
  insertion: "tilted 0.1 mm depth", "soft-insertion": "soft peg depth",
  "cloth-fold": "fold coverage",
  sweep: "objects collected", "ball-compress": "plate load",
  "ball-grasp": "2 N hold lift", "hinge-plastic": "rest angle", sheet: "tip set",
  twist: "disc slip torque", "g1-walk": "flat, 0.6 m/s",
};
const outcomeTone = (scene, m, row) => {
  if (row.status === "error") return "muted";
  if (scene === "card-catch") return m.outcome === "caught" ? "good" : "bad";
  if (scene === "stack") return m.stack_survives ? "good" : "bad";
  if (scene === "push") return m.passed_count === 3 ? "good" : "warn";
  if (scene === "ball-grasp") return m.hold_lift_mm == null ? "bad" : "good";
  if (scene === "soft-insertion") return m.depth_reached ? "good" : "bad";
  if (scene === "cloth-fold") return m.passed ? "good" : "bad";
  if (scene === "hinge-plastic") return Math.abs(m.final_error_deg) < 0.05 ? "good" : "bad";
  if (scene === "g1-walk") return m.fell ? "bad" : "good";
  if (scene === "twist") return Math.abs(m.slip_ratio - 1) < 0.25 ? "good" : "warn";
  return "none";
};

function ladderTable() {
  const L = INDEX.ladder ?? [];
  const open = (s) => { const f = replayFor(s.scene, s.variant); if (f) openReplay(f); };
  const cols = ROUTES.map((r) => ({ label: label(r), color: COLORS[r] }));
  const groups = [...new Set(L.map((s) => s.group))];
  const section = (i) => {
    if (L[i].group === L[i - 1]?.group) return null;
    const g = L[i].group;
    return el("span", {}, el("span", { class: "level" }, `${groups.indexOf(g) + 1}`),
      el("span", { class: "group-name" }, g),
      el("span", { class: "muted small" }, ` · ${LADDER_NOTE[g.split(" (")[0]] ?? ""}`));
  };
  $("#ladder-matrix").replaceChildren(matrix(L.map((s) => el("span", {}, s.title,
    el("span", { class: "muted small" }, ` · ${OUTCOME_LABEL[s.scene]} · `
      + `${fx(s.mean_contacts, 1)} contacts/step`))), cols, (i, j) => {
    const s = L[i];
    const row = variantRows(s.scene, s.variant)[ROUTES[j]];
    if (!row) return { text: "not run", tone: "empty" };
    const [key, f] = OUTCOME[s.scene];
    const m = row.metrics ?? {};
    if (s.scene === "g1-walk" && m.fell) {
      return { text: `falls ${fx(m.fell_at_s, 1)} s`, tone: "bad", onclick: () => open(s) };
    }
    return { text: row.status === "error" ? "rejected" : f(m[key]),
             tone: outcomeTone(s.scene, m, row), title: row.error ?? "",
             onclick: () => open(s) };
  }, { corner: "scene · outcome", section }));
}

const LADDER_NOTE = {
  "No contact": "free motion and actuation: integrator and dynamics only",
  "One body on a surface": "one object against the ground: friction, impact, tunnelling",
  "Multi-body impacts": "momentum passed along a row of resting bodies",
  "Grasping and stacking": "objects held up by friction from fingers or each other",
  "Multi-object manipulation": "a tool pushing or inserting parts that touch each other",
  "Patch contact": "torsional friction over a contact area, not a point",
  "Crowded": "many objects touching at once",
  "Deformable": "soft bodies with many distributed contacts",
  "Elasto-plastic": "materials that yield and keep a permanent set",
  "Legged locomotion": "repeated foot impacts under a closed-loop policy",
};

// --- MuJoCo 4.0 ----------------------------------------------------------------------

function mj4Cards() {
  const S = SUMMARY;
  const cards = [];
  const card = (labelText, big, text) => cards.push(el("div", { class: "card" },
    el("div", { class: "label" }, labelText), el("div", { class: "big" }, big), el("p", {}, text)));
  const one = benchRun("mujoco4", "mj_step", 1)?.world_steps_per_s
    / benchRun("mujoco311", "mj_step", 1)?.world_steps_per_s;
  const many = benchRun("mujoco4", "rollout", 32)?.world_steps_per_s
    / benchRun("mujoco311", "rollout", 32)?.world_steps_per_s;
  card("CPU speed", `${fx(one, 2)}× / ${fx(many, 2)}×`, "4.0 over 3.11, contact-free Panda: "
    + "one thread, and 32 threads with mujoco.rollout.");
  const versions = Object.values(S["mujoco4-vs-311"] ?? {});
  card("Identical trajectories", `${versions.filter((r) => (r.path_diff_m ?? 0) < 1e-6).length}`
    + ` / ${versions.length}`, "Variants within 1 µm over the whole run.");
  const chain = S["mujoco4-vs-311"]?.["collision-chain/dt0.5ms"];
  const pairs = (route) => Object.fromEntries(chain?.contacts_per_frame[route] ?? []);
  if (chain) {
    card("Resting box faces", `${fx(pairs("mujoco311")["prism-world"], 2)} → `
      + `${fx(pairs("mujoco4")["prism-world"], 2)}`, "Prism–ground contacts per step in the "
      + "collision chain: where the versions differ.");
  }
  const sparse = Object.entries(S.phase4 ?? {}).filter(([k]) => k.includes("jac-sparse"));
  if (sparse.length) {
    const worst = Math.max(...sparse.map(([, r]) => r.path_diff_vs_base_m ?? 0));
    card("mujoco#3767 sparse bug", worst < 1e-9 ? "not triggered" : mm(worst),
      `Sparse vs dense Jacobian on ${sparse.length / 2} scenes.`);
  }
  $("#mj4-cards").replaceChildren(...cards);
}

function versionsChart() {
  const rows = Object.entries(SUMMARY["mujoco4-vs-311"] ?? {})
    .map(([key, r]) => ({ key, d: Math.max((r.path_diff_m ?? 0) * 1e3, 1e-7) }))
    .sort((a, b) => a.d - b.d);
  $("#versions-chart").replaceChildren(bars(rows.map((r) => ({ label: r.key, bars: [{
    value: r.d, color: r.d < 1e-3 ? "#3fcf8e" : r.d < 1 ? "#f2b84b" : "#ff6b6b",
    text: r.d <= 1e-7 ? "identical" : mm(r.d / 1e3, 2) }] })),
  { log: true, min: 1e-7, max: 3e3, unit: "3.11 vs 4.0 path difference [mm]", labelW: 260,
    barH: 11 }));
}

function cardSection() {
  const cols = cardCols();
  $("#card-matrix").replaceChildren(matrix(CARD_CASES.map(([, t]) => t), cols, (i, j) =>
    cardCell(CARD_CASES[i][0], ...CARD_ROUTES[j]), { corner: "impact speed" }),
  el("p", { class: "muted" }, "Click a cell for its replay. 'Through at N ms' is when the card's "
    + "centre passed below the plate inside its footprint."));
  const z = (v, r) => metric("card-catch", v, r, "height_mm");
  const series = [...ROUTES.map((r) => ({ label: label(r), color: COLORS[r], r, v: "dt1ms-v3" })),
                  { label: "4.0 + speculative", color: "#ffe066", r: "mujoco4",
                    v: "dt1ms-v3-speculative" }]
    .map((s) => ({ ...s, points: (z(s.v, s.r) ?? []).slice(0, 21).map((h, k) => [k, h]),
                   marker: true })).filter((s) => s.points.length);
  $("#card-chart").replaceChildren(lines(series, { xlabel: "time [ms], 3 m/s release, 1 ms step",
    ylabel: "card centre above plate top [mm]", ymin: -8, ymax: 6, xmin: 0, xmax: 20,
    hline: { value: -1, label: "plate bottom (1 mm plate)", color: "#ff6b6b" }, height: 280 }),
  el("p", { class: "muted" }, "Without speculative contacts the five engines trace the same "
    + "falling line, so they overlap as one."),
  replayLink("card-catch", "dt1ms-v3", "Open the 3 m/s card replay →"));
}

function tumbleSection() {
  const series = [
    ...tumbleSeries("rk4").map((s) => ({ ...s, label: `${s.label} · RK4` })),
    ...tumbleSeries("euler").map((s) => ({ ...s, label: `${s.label} · Euler`, dash: "5 4" })),
  ];
  $("#tumble-chart").replaceChildren(lines(series, { logX: true, logY: true,
    xlabel: "timestep [ms]", ylabel: "max orientation error over 4 s [°]", height: 340,
    ymin: 1e-6, ymax: 100 }),
  el("p", { class: "muted" }, "3.11's RK4 converges at second order (slope 2); 4.0's at fourth "
    + "order and reaches the reference's resolution at 5 ms. MuJoCo-Warp 4.0 has the same RK4 "
    + "but stalls at a 0.2–0.6° floor that grows with step count, consistent with float32 "
    + "round-off. Newton 1.7 (bundled MuJoCo-Warp 3.15) is off by 1–13° with no convergence; "
    + "the cause is not isolated. Genesis has no RK4."));
  const errs = (v, r) => metric("tumble", v, r, "error_deg");
  const t = (arr) => arr.map((e, k) => [(k * 4) / (arr.length - 1), e]);
  const ts = [["dt2ms-rk4", "mujoco311", "3.11 RK4"], ["dt2ms-rk4", "mujoco4", "4.0 RK4"],
              ["dt2ms-euler", "mujoco4", "4.0 Euler"],
              ["dt2ms-implicitfast", "mujoco4", "4.0 implicitfast"],
              ["dt2ms-implicitfast", "genesis", "Genesis implicitfast"],
              ["dt2ms-rk4", "newton", "Newton RK4"],
              ["dt2ms-rk4", "mjwarp4", "MuJoCo-Warp RK4"]]
    .map(([v, r, name], i) => ({ label: name, color: COLORS[r],
      dash: v.includes("rk4") ? null : i === 3 ? "2 3" : "5 4",
      points: t(errs(v, r) ?? []).filter(([, e]) => e > 0), marker: false }))
    .filter((s) => s.points.length);
  $("#tumble-time").replaceChildren(lines(ts, { logY: true, xlabel: "time [s], 2 ms step",
    ylabel: "orientation error [°]", height: 340, ymin: 1e-6, ymax: 100 }),
  replayLink("tumble", "dt2ms-euler", "Open the tumble replay (Euler) →"));
}

// --- deformable and plastic ----------------------------------------------------------

function defCompress() {
  const rows = [["mujoco311", "dt0.05ms", "as authored"], ["mujoco4", "dt0.05ms", "as authored"],
                ["mjwarp4", "dt0.05ms", "as authored"],
                ["mujoco4", "dt0.05ms-legacy", "legacy E′, ν′"],
                ["mjwarp4", "dt0.05ms-legacy", "legacy E′, ν′"],
                ["article:mujoco_cpu", "dt0.05ms", "article, 3.x"],
                ["article:mujoco_warp", "dt0.05ms", "article, 3.x"]]
    .map(([r, v, what]) => ({ label: label(r), sub: what,
      bars: [{ value: plateLoad(r, v), color: COLORS[r], dim: what !== "as authored",
               text: `${fx(plateLoad(r, v), 3)} N` }] }))
    .filter((r) => r.bars[0].value != null);
  $("#def-compress").replaceChildren(bars(rows, { unit: "hold plate load [N]", labelW: 200 }),
    replayLink("ball-compress", "dt0.05ms", "Open the compression replay →"));
}

function defGrasp() {
  const P = get(SUMMARY, "phase6", "ball-grasp") ?? {};
  const effort = (v) => parseFloat(v.match(/e([\d.]+)N/)[1]);
  const stiff = (v) => v.endsWith("-stiff");
  const vs = Object.keys(P).filter((v) => /^dt0\.05ms-e[\d.]+N(-stiff)?$/.test(v))
    .sort((a, b) => stiff(b) - stiff(a) || effort(b) - effort(a));
  const cols = ["mujoco311", "mujoco4", "mjwarp4", "article:mujoco_cpu", "article:mujoco_warp"];
  $("#def-grasp").replaceChildren(matrix(vs.map((v) => `${effort(v)} N · `
    + (stiff(v) ? "2 ms contact" : "article's 10 ms")),
    cols.map((r) => ({ label: label(r), color: COLORS[r] })), (i, j) => {
      const row = P[vs[i]][cols[j]];
      if (!row) return null;
      const m = row.metrics;
      const file = replayFor("ball-grasp", vs[i]);
      return m.completed ? { text: `${fx(m.hold_lift_mm)} mm`, tone: "good",
                             onclick: file ? () => openReplay(file) : null }
        : { text: `invalid ${fx(m.invalid_at_s, 2)} s`, tone: "bad",
            title: `${m.invalid_frames ?? "?"} invalid frames; lift ignoring validity `
              + `${fx(m.ungated_hold_lift_mm)} mm`, onclick: file ? () => openReplay(file) : null };
    }, { corner: "finger effort" }));
}

function defHinge() {
  const v = "dt1ms-y5deg";
  const rows = ROUTES.map((r) => {
    const row = get(variantRows("hinge-plastic", v), r);
    const m = row?.metrics ?? {};
    return { label: label(r), sub: row?.status === "error" ? "rejects springyield"
      : `peak ${fx(m.peak_angle_deg, 2)}°, ideal rest ${fx(m.expected_final_deg, 2)}°`,
    bars: [{ value: row?.status === "error" ? null : Math.max(m.final_angle_deg, 0.001),
             color: COLORS[r], text: row?.status === "error" ? "compile error"
        : `rests at ${fx(m.final_angle_deg, 3)}°` }] };
  });
  const ideal = metric("hinge-plastic", v, "mujoco4", "expected_final_deg");
  $("#def-hinge").replaceChildren(bars(rows, { unit: "rest angle after release [°]", min: 0,
    max: 6, ref: { value: ideal, label: `ideal ${fx(ideal, 2)}°` }, labelW: 260 }),
  replayLink("hinge-plastic", v, "Open the plastic hinge replay →"));
}

function defPlastic() {
  const ys = [0.01, 0.03, 0.1];
  const set = (y) => metric("ball-compress", `dt0.05ms-y${y}`, "mujoco4", "permanent_set_mm");
  const load = (y) => metric("ball-compress", `dt0.05ms-y${y}`, "mujoco4", "hold_plate_load_N");
  $("#def-plastic-chart").replaceChildren(lines([
    { label: "permanent set [mm]", color: COLORS.mujoco4, points: ys.map((y) => [y, set(y)]) },
    { label: "hold load ×5 [N]", color: "#f2b84b", dash: "5 4",
      points: ys.map((y) => [y, load(y) * 5]) }],
  { logX: true, xlabel: "yield strain (MuJoCo 4.0 CPU)", ylabel: "mm  /  N×5", height: 280,
    ymin: 0 }),
  el("p", { class: "muted" }, `Elastic ball: set ${fx(metric("ball-compress", "dt0.05ms",
    "mujoco4", "permanent_set_mm"), 2)} mm, load ${fx(plateLoad("mujoco4"), 2)} N.`));
  const cases = [["ball-compress", "dt0.05ms-y0.1", "squeeze, εy 0.1"],
                 ["ball-compress", "dt0.05ms-y0.03", "squeeze, εy 0.03"],
                 ["ball-compress", "dt0.05ms-y0.01", "squeeze, εy 0.01"],
                 ["ball-grasp", "dt0.05ms-e2N-y0.1", "2 N grasp, εy 0.1"],
                 ["ball-grasp", "dt0.05ms-e2N-y0.03", "2 N grasp, εy 0.03"]];
  const routes = ["mujoco311", "mujoco4", "mjwarp4"];
  $("#def-plastic").replaceChildren(matrix(cases.map(([, , t]) => t),
    routes.map((r) => ({ label: label(r), color: COLORS[r] })), (i, j) => {
      const [scene, v] = cases[i];
      const row = get(variantRows(scene, v), routes[j]);
      if (!row) return null;
      const file = replayFor(scene, v);
      const onclick = file ? () => openReplay(file) : null;
      if (row.status === "error") return { text: "rejected", tone: "muted" };
      const m = row.metrics;
      if (scene === "ball-compress") {
        return m.completed ? { text: `set ${fx(m.permanent_set_mm, 1)} mm`, tone: "good", onclick }
          : { text: `non-finite ${fx(m.invalid_at_s, 2)} s`, tone: "bad", onclick };
      }
      return m.completed ? { text: `dent ${fx(m.dent_mm, 1)} mm`, tone: "good", onclick }
        : { text: `invalid ${fx(m.invalid_at_s, 3)} s`, tone: "bad", onclick };
    }, { corner: "case" }),
  el("p", { class: "muted" }, "At εy 0.03 the yield stress (600 Pa) is about the ball's weight "
    + "pressure, so it slumps on the table before the fingers close."));
}

function defSheet() {
  const vs = [["dt0.1ms-elastic", "elastic"], ["dt0.1ms-yb0.004", "bending yield 0.004"],
              ["dt0.1ms-yb0.003", "bending yield 0.003"], ["dt0.1ms-yb0.002", "bending yield 0.002"]];
  const routes = ["mujoco311", "mujoco4", "mjwarp4"];
  $("#def-sheet").replaceChildren(matrix(vs.map(([, t]) => t),
    routes.map((r) => ({ label: label(r), color: COLORS[r] })), (i, j) => {
      const row = get(variantRows("sheet", vs[i][0]), routes[j]);
      if (!row) return { text: "not run", tone: "empty" };
      if (row.status === "error") {
        return { text: /discrete/.test(row.error ?? "") ? "no discrete integrator"
          : "rejects yield", tone: "muted", title: row.error };
      }
      const m = row.metrics ?? {};
      if (m.invalid_at_s != null) return { text: `non-finite ${fx(m.invalid_at_s, 3)} s`, tone: "bad" };
      const file = replayFor("sheet", vs[i][0]);
      return { text: `set ${fx(m.permanent_set_mm, 2)} mm · held ${fx(m.held_tip_mm, 1)} mm`,
               tone: m.permanent_set_mm > 2 ? "good" : m.permanent_set_mm > 0.1 ? "warn" : "none",
               onclick: file ? () => openReplay(file) : null };
    }, { corner: "shell" }),
  lines([["mujoco4", "dt0.1ms-elastic", "4.0 CPU · elastic", "5 4"],
         ["mujoco4", "dt0.1ms-yb0.002", "4.0 CPU · yield 0.002", null],
         ["mjwarp4", "dt0.1ms-elastic", "MJW 4.0 · elastic", "5 4"],
         ["mjwarp4", "dt0.1ms-yb0.002", "MJW 4.0 · yield 0.002", null]]
    .map(([r, v, name, dash]) => {
      const tip = metric("sheet", v, r, "tip_mm") ?? [];
      return { label: name, color: COLORS[r], dash, marker: false,
               points: tip.map((d, k) => [(k * 2) / (tip.length - 1), d]) };
    }).filter((s) => s.points.length),
  { xlabel: "time [s]: press 0.1–0.6 s, hold, release 0.8–1.3 s", ylabel: "tip deflection [mm]",
    height: 260, hline: { value: 0, label: "undeformed", color: "#8d96a3" } }),
  el("p", { class: "muted" }, "The press puts about 0.0046 outer-fibre strain at the clamp, so "
    + "yield strains above that stay elastic. 3.11 has neither the discrete integrator "
    + "needed for shell elasticity nor yield."),
  replayLink("sheet", "dt0.1ms-yb0.002", "Open the plastic sheet replay →"));
}

function deformable() {
  $("#support-matrix").replaceChildren(supportMatrix(),
    el("p", { class: "muted" }, "Newton and Genesis are not run on MJCF flex: the article "
      + "compares their own soft-body solvers (VBD, FEM), which do not read it."));
  softInsertionTable();
  clothFoldTable();
  defCompress();
  defGrasp();
  defHinge();
  defPlastic();
  defSheet();
}

// --- patch contact -------------------------------------------------------------------

const twistRatio = (route, variant) => metric("twist", variant, route, "slip_ratio");
// [route, variant suffix, label]; the suffix follows the shape (dt1ms-disc<suffix>).
const TWIST_CASES = [
  ["mujoco311", "", "MuJoCo 3.11"], ["mujoco4", "", "MuJoCo 4.0"],
  ["mujoco4", "-condim6", "4.0 + torsional friction"], ["mujoco4", "-multiccd", "4.0 + multiccd"],
  ["mjwarp4", "", "MuJoCo-Warp 4.0"], ["genesis", "", "Genesis"],
  ["newton", "", "Newton, as shipped"], ["newton", "-iw", "Newton + invweight fix"],
  ["newton", "-point-iw", "Newton point pipeline + fix"],
  ["newton", "-hydro", "Newton hydroelastic"], ["newton", "-hydro-iw", "Newton hydroelastic + fix"],
  ["newton", "-hydro-full-iw", "Newton full patch + fix"],
];
const TWIST_OVERVIEW = [0, 1, 6, 7, 10].map((i) => TWIST_CASES[i]);

function twistBars(cases, opts = {}) {
  const ref = (shape) => metric("twist", `dt1ms-${shape}`, "mujoco4", "slip_torque_ref_Nm");
  const rows = cases.map(([r, suffix, name]) => ({ label: name, bars: ["disc", "square"]
    .map((shape) => {
      const m = get(variantRows("twist", `dt1ms-${shape}${suffix}`), r, "metrics");
      return m?.slip_ratio == null ? null : { value: m.slip_ratio, color: COLORS[r],
        dim: shape === "square", title: `${shape}: ${fx(m.slip_torque_Nm, 3)} N·m (analytic `
          + `${fx(ref(shape), 3)}), ${fx(m.contacts_per_step, 0)} contacts/step`,
        text: `${shape === "disc" ? "disc" : "sq."} ${fx(m.slip_ratio, 2)}×` };
    }).filter(Boolean) })).filter((r) => r.bars.length);
  return bars(rows, { log: true, min: 0.03, max: 3, unit: "slip torque / analytic (log)",
    ref: { value: 1, label: "uniform pressure" }, barH: 11, ...opts });
}

function twistSection() {
  const file = replayFor("twist", "dt1ms-disc-hydro-iw");
  $("#twist-chart").replaceChildren(twistBars(TWIST_CASES, { labelW: 230 }),
    el("p", { class: "muted" }, "Bright bars: disc, dim bars: square; hover for torque and "
      + "contacts per step. The point contacts sit at the rim or corners, which gives the "
      + "largest lever arm, and torsional friction adds on top. As shipped, Newton 1.7 sets "
      + "body_invweight0 on the hinged puck about 1,000× above MuJoCo's, so the pads sink "
      + "millimetres into it and grip far too weakly. The full hydroelastic patch "
      + "(about 8,500 contacts) hits the solver's iteration limit."),
    file ? el("p", {}, el("button", { class: "link", onclick: () => openReplay(file) },
      "Open the hydroelastic replay (X-ray shows the patch) →")) : "");
}

// --- legged locomotion ---------------------------------------------------------------

const TERRAINS = ["flat", "glass", "sand", "mud", "sponge"];
const G1_SPEEDS = [0.3, 0.6, 0.9];
const g1Metrics = (terrain, speed, route) =>
  metric("g1-walk", `dt2ms-${terrain}-v${speed}`, route, "fell") == null ? null
    : get(variantRows("g1-walk", `dt2ms-${terrain}-v${speed}`), route, "metrics");
const g1Rows = () => TERRAINS.flatMap((t) => G1_SPEEDS.map((v) => [t, v]))
  .filter(([t, v]) => ROUTES.some((r) => g1Metrics(t, v, r)));

function g1Cell(terrain, speed, route, compact) {
  const m = g1Metrics(terrain, speed, route);
  if (!m) return { text: "—", tone: "empty" };
  const file = replayFor("g1-walk", `dt2ms-${terrain}-v${speed}`);
  const onclick = file && !compact ? () => openReplay(file) : null;
  const title = `${fx(m.distance_m, 2)} m, drift ${fx(m.lateral_drift_m, 2)} m, sink `
    + `${fx(m.sink_mm, 1)} mm, foot slip ${fx(m.foot_slip_mm_s, 0)} mm/s`;
  if (m.fell) return { text: compact ? "✕" : `falls ${fx(m.fell_at_s, 1)} s`, tone: "bad", title, onclick };
  const tone = m.speed_ratio > 0.7 && m.speed_ratio < 1.3 ? "good" : "warn";
  return { text: compact ? "✓" : `${fx(m.speed_ratio, 2)}× speed`, tone, title, onclick };
}

function g1Matrix(speed = null, compact = false) {
  const rows = speed == null ? g1Rows() : TERRAINS.map((t) => [t, speed]);
  return matrix(rows.map(([t, v]) => (speed == null ? `${t} · ${v} m/s` : t)),
    ROUTES.map((r) => ({ label: compact ? SHORT[r] : label(r), color: COLORS[r] })),
    (i, j) => g1Cell(...rows[i], ROUTES[j], compact),
    compact ? {} : { corner: "terrain · command" });
}

function g1Falls() {
  return Object.fromEntries(ROUTES.map((r) => [r, g1Rows().filter(([t, v]) =>
    g1Metrics(t, v, r)?.fell).map(([t, v]) => `${t} ${v}`)]));
}

function g1Summary() {
  const falls = g1Falls();
  const soft = (r) => falls[r].filter((f) => !f.startsWith("glass")).length;
  const softRuns = g1Rows().filter(([t]) => !["flat", "glass"].includes(t)).length;
  const flatOk = ROUTES.every((r) => !falls[r].some((f) => f.startsWith("flat")));
  const steady = ROUTES.filter((r) => !soft(r)).map((r) => SHORT[r]);
  const shaky = ROUTES.filter((r) => soft(r)).map((r) => `${label(r)} falls in ${soft(r)} of `
    + `${softRuns}`);
  return `${flatOk ? "Every engine walks flat ground alike. " : ""}Glass (μ 0.1) topples the `
    + "policy at 0.6 m/s and above almost everywhere. On sand, mud and sponge "
    + `${steady.join(", ")} keep walking${shaky.length ? `; ${shaky.join(", ")} runs with the `
      + "same contact parameters, because soft contact maps to different sinkage per engine"
      : ""}.`;
}

function g1Section() {
  if (!g1Rows().length) return;
  $("#g1-matrix").replaceChildren(g1Matrix(), el("p", { class: "muted" }, "Cells: forward "
    + "speed from 2 s to the end over the command, or the time the pelvis dropped below 0.5 m. "
    + "Hover for distance, drift, sinkage and foot slip; 0.6 m/s cells open the replay."));
  const grouped = (key, unit, digits = 1) => bars(TERRAINS.map((t) => ({ label: t,
    bars: ROUTES.map((r) => {
      const v = g1Metrics(t, 0.6, r)?.[key];
      return v == null ? null : { value: v, color: COLORS[r], text: `${fx(v, digits)}` };
    }).filter(Boolean) })), { unit, labelW: 70, barH: 8 });
  $("#g1-sink").replaceChildren(el("h3", {}, "Foot sinkage at 0.6 m/s"),
    grouped("sink_mm", "median deepest foot penetration [mm]"));
  $("#g1-track").replaceChildren(el("h3", {}, "Foot slip at 0.6 m/s"),
    grouped("foot_slip_mm_s", "mean loaded-foot speed [mm/s]", 0), routeLegend(ROUTES));
  $("#g1-short").replaceChildren(el("b", {}, "In short: "), g1Summary());
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
  const best = gpu.map((r) => [r, (get(SUMMARY, `bench/${r}`, "workloads", workload, "runs") ?? [])
    .filter((x) => !x.error).sort((a, b) => b.world_steps_per_s - a.world_steps_per_s)[0]])
    .filter(([, b]) => b).sort((a, b) => b[1].world_steps_per_s - a[1].world_steps_per_s);
  const cpu1 = r4.find((x) => x.mode === "mj_step" && x.threads === 1)?.world_steps_per_s;
  $("#runtime-short").replaceChildren(el("b", {}, "In short: "), best.length
    ? `on ${workload}, ${label(best[0][0])} peaks at ${fmtSI(best[0][1].world_steps_per_s)} `
      + `world-steps/s with ${best[0][1].worlds.toLocaleString()} worlds`
      + (cpu1 ? `, against ${fmtSI(cpu1)} on one MuJoCo 4.0 CPU thread` : "")
      + ". GPUs pay off only with thousands of worlds." : "no GPU runs for this workload.");
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
  // Replay fetches append ?v=<build> so a fresh `make web` is picked up without fighting
  // a stale browser cache of the previous export.
  setCacheBust(INDEX.build || INDEX.replays?.length || Date.now());
  viewer = new Viewer($("#viewer"));
  viewer.routeLabel = label;
  document.querySelectorAll("nav button").forEach((b) =>
    b.addEventListener("click", () => showTab(b.dataset.tab)));
  buildSceneList();
  for (const f of [overview, ladderTable, sweepTable, insertionTable, tuningTable, mj4Cards,
                   versionsChart, versionsTable, sensitivityTable, cardSection, tumbleSection,
                   deformable, twistSection, g1Section]) {
    try {
      f();
    } catch (err) {
      console.error(`${f.name}:`, err);
    }
  }
  document.querySelectorAll("[data-goto]").forEach((a) =>
    a.addEventListener("click", () => goto(a.dataset.goto)));
  document.querySelectorAll("[data-replay]").forEach((p) =>
    p.append(replayButton(...p.dataset.replay.split("|"))));
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
  $("#show-points").addEventListener("change", (e) => {
    viewer.showPoints = e.target.checked;
    const dim = e.target.checked ? "1" : "0.35";
    $("#contact-key").style.opacity = dim;
    $("#contact-legend").style.opacity = dim;
    $("#contact-note").style.opacity = dim;
    seek(playT);
  });
  const setOpacity = (v) => { $("#opacity").value = v; viewer.setOpacity(+v); };
  $("#opacity").addEventListener("input", (e) => setOpacity(e.target.value));
  $("#xray").addEventListener("click", () => setOpacity(viewer.opacity < 1 ? 1 : 0.3));
  $("#studio").addEventListener("change", (e) => viewer.setStudio(e.target.checked));
  if (new URLSearchParams(location.search).get("studio") === "0") {
    $("#studio").checked = false;
    viewer.setStudio(false);
  }
  const urlOpacity = new URLSearchParams(location.search).get("opacity");
  if (urlOpacity) setOpacity(urlOpacity);
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
