// Small dependency-free SVG charts for the result tabs: grouped bars, dot rows on a log
// axis, line charts (lin/log) and an HTML outcome matrix. Dark theme, viewBox-scaled.

const NS = "http://www.w3.org/2000/svg";
const INK = "#c3c9d2", MUTED = "#8d96a3", GRID = "#232a32";

function s(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  for (const k of kids.flat()) {
    if (k == null) continue;
    e.append(k instanceof Node ? k : document.createTextNode(k));
  }
  return e;
}

const tip = (text) => (text ? s("title", {}, text) : null);
let clipId = 0;

function scale(lo, hi, a, b, log) {
  if (log) {
    const [l0, l1] = [Math.log10(lo), Math.log10(hi)];
    return (v) => a + ((Math.log10(Math.max(v, lo)) - l0) / (l1 - l0)) * (b - a);
  }
  return (v) => a + ((v - lo) / (hi - lo)) * (b - a);
}

export function fmt(v) {
  if (v == null || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e6) return `${+(v / 1e6).toPrecision(3)}M`;
  if (a >= 1e3) return `${+(v / 1e3).toPrecision(3)}k`;
  if (a < 1e-3) return v.toExponential(0);
  return String(+v.toPrecision(3));
}

function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) out.push(10 ** e);
  return out;
}

function linTicks(lo, hi, n = 5) {
  const step = 10 ** Math.floor(Math.log10((hi - lo) / n));
  const k = [1, 2, 5, 10].find((m) => (hi - lo) / (step * m) <= n) * step;
  const out = [];
  for (let v = Math.ceil(lo / k) * k; v <= hi + 1e-12; v += k) out.push(+v.toPrecision(10));
  return out;
}

/** Horizontal bars. rows: [{label, sub?, bars: [{value, color, text?, title?}]}]. */
export function bars(rows, { unit = "", log = false, min, max, ref, width = 760, barH = 14,
                             labelW = 190, onRow } = {}) {
  const vals = rows.flatMap((r) => r.bars.map((b) => b.value)).filter((v) => v != null && isFinite(v));
  const hi = max ?? Math.max(...vals, ref?.value ?? 0) * 1.12;
  const lo = min ?? (log ? Math.min(...vals.filter((v) => v > 0)) / 2 : 0);
  const rowH = (r) => Math.max(r.bars.length * (barH + 3) + 10, r.sub ? 36 : 0);
  const plotH = rows.reduce((a, r) => a + rowH(r), 0) + 4;
  const height = plotH + (unit ? 40 : 22);
  const x = scale(lo, hi, labelW, width - 70, log);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart" });
  for (const t of log ? logTicks(lo, hi) : linTicks(lo, hi)) {
    svg.append(s("line", { x1: x(t), x2: x(t), y1: 0, y2: plotH, stroke: GRID }),
               s("text", { x: x(t), y: plotH + 14, fill: MUTED, "font-size": 11,
                           "text-anchor": "middle" }, fmt(t)));
  }
  if (unit) svg.append(s("text", { x: (labelW + width - 70) / 2, y: height - 4, fill: MUTED,
                                   "font-size": 11, "text-anchor": "middle" }, unit));
  let y = 4;
  for (const r of rows) {
    const g = s("g", { class: onRow ? "clickable" : null });
    if (onRow) g.addEventListener("click", () => onRow(r));
    const h = rowH(r);
    g.append(s("text", { x: 0, y: y + h / 2 - (r.sub ? 2 : -4), fill: INK, "font-size": 12.5 },
               r.label));
    if (r.sub) g.append(s("text", { x: 0, y: y + h / 2 + 12, fill: MUTED, "font-size": 11 }, r.sub));
    r.bars.forEach((b, i) => {
      const yy = y + 4 + i * (barH + 3);
      if (b.value == null || !isFinite(b.value)) {
        g.append(s("text", { x: labelW + 4, y: yy + barH - 3, fill: MUTED, "font-size": 11.5 },
                   b.text ?? "—", tip(b.title)));
        return;
      }
      const w = Math.max(1.5, x(b.value) - labelW);
      g.append(s("rect", { x: labelW, y: yy, width: w, height: barH, rx: 2, fill: b.color,
                           opacity: b.dim ? 0.45 : 0.9 }, tip(b.title)),
               s("text", { x: labelW + w + 5, y: yy + barH - 3, fill: INK, "font-size": 11.5 },
                 b.text ?? fmt(b.value)));
    });
    svg.append(g);
    y += h;
  }
  if (ref) {
    svg.append(s("line", { x1: x(ref.value), x2: x(ref.value), y1: 0, y2: plotH,
                           stroke: "#e6e9ee", "stroke-dasharray": "4 3", opacity: 0.7 }),
               s("text", { x: x(ref.value) + 4, y: 11, fill: INK, "font-size": 11 }, ref.label));
  }
  return svg;
}

/** Rows of dots on one shared axis. rows: [{label, group?, points: [{value, color, title}]}]. */
export function dots(rows, { log = true, min, max, unit = "", width = 760, labelW = 200,
                             rowH = 24, bands, onRow, right } = {}) {
  const vals = rows.flatMap((r) => r.points.map((p) => p.value)).filter((v) => v > 0 && isFinite(v));
  const lo = min ?? Math.min(...vals) / 3, hi = max ?? Math.max(...vals) * 3;
  const rightW = right ? 70 : 0;
  const groups = rows.filter((r, i) => r.group && r.group !== rows[i - 1]?.group).length;
  const plotH = rows.length * rowH + groups * 22 + 8;
  const height = plotH + (unit ? 40 : 22);
  const x = scale(lo, hi, labelW, width - 16 - rightW, log);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart" });
  for (const b of bands ?? []) {
    svg.append(s("rect", { x: x(b.from), width: x(b.to) - x(b.from), y: 0, height: plotH,
                           fill: b.color, opacity: 0.07 }),
               s("text", { x: (x(b.from) + x(b.to)) / 2, y: plotH - 4, fill: b.color,
                           "font-size": 10.5, "text-anchor": "middle", opacity: 0.9 }, b.label));
  }
  for (const t of log ? logTicks(lo, hi) : linTicks(lo, hi)) {
    svg.append(s("line", { x1: x(t), x2: x(t), y1: 0, y2: plotH, stroke: GRID }),
               s("text", { x: x(t), y: plotH + 14, fill: MUTED, "font-size": 11,
                           "text-anchor": "middle" }, fmt(t)));
  }
  if (unit) svg.append(s("text", { x: (labelW + width - 16 - rightW) / 2, y: height - 4,
                                   fill: MUTED, "font-size": 11, "text-anchor": "middle" }, unit));
  if (right) svg.append(s("text", { x: width - 4, y: 12, fill: MUTED, "font-size": 10.5,
                                    "text-anchor": "end" }, right.title));
  let y = 0, group = null;
  for (const r of rows) {
    if (r.group && r.group !== group) {
      group = r.group;
      svg.append(s("text", { x: 0, y: y + 16, fill: "#4fb3ff", "font-size": 11.5,
                             "font-weight": 600 }, group));
      y += 22;
    }
    const g = s("g", { class: onRow ? "clickable" : null });
    if (onRow) g.addEventListener("click", () => onRow(r));
    const cy = y + rowH / 2;
    g.append(s("rect", { x: 0, y, width, height: rowH, fill: "transparent" }),
             s("line", { x1: labelW, x2: width - 16 - rightW, y1: cy, y2: cy, stroke: GRID }),
             s("text", { x: 8, y: cy + 4, fill: INK, "font-size": 12.5 }, r.label));
    for (const p of r.points) {
      if (p.value == null || !isFinite(p.value)) continue;
      const v = Math.max(p.value, lo);
      g.append(s(p.shape === "square" ? "rect" : "circle", p.shape === "square"
        ? { x: x(v) - 5, y: cy - 5, width: 10, height: 10, fill: p.color, opacity: 0.9 }
        : { cx: x(v), cy, r: 5.5, fill: p.color, opacity: 0.9, stroke: "#0f1216" },
      tip(p.title)));
    }
    if (right) g.append(s("text", { x: width - 4, y: cy + 4, fill: MUTED, "font-size": 11.5,
                                    "text-anchor": "end" }, right.value(r)));
    svg.append(g);
    y += rowH;
  }
  return svg;
}

/** Line chart. series: [{label, color, points: [[x, y]], dash?, marker?}]. */
export function lines(series, { logX = false, logY = false, xlabel = "", ylabel = "",
                                width = 760, height = 300, xmin, xmax, ymin, ymax, hline,
                                legend = true } = {}) {
  const pts = series.flatMap((d) => d.points).filter(([a, b]) => isFinite(a) && isFinite(b)
    && (!logX || a > 0) && (!logY || b > 0));
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1] = [xmin ?? Math.min(...xs), xmax ?? Math.max(...xs)];
  let [y0, y1] = [ymin ?? Math.min(...ys), ymax ?? Math.max(...ys)];
  if (!logY && ymin == null) y0 = Math.min(0, y0);
  if (logY) { y0 = ymin ?? y0 / 3; y1 = ymax ?? y1 * 3; } else if (ymax == null) y1 += (y1 - y0) * 0.08;
  const L = 62, R = legend ? 170 : 16, T = 10, B = 40;
  const x = scale(x0, x1, L, width - R, logX), y = scale(y0, y1, height - B, T, logY);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart" });
  for (const t of logX ? logTicks(x0, x1) : linTicks(x0, x1)) {
    svg.append(s("line", { x1: x(t), x2: x(t), y1: T, y2: height - B, stroke: GRID }),
               s("text", { x: x(t), y: height - B + 15, fill: MUTED, "font-size": 11,
                           "text-anchor": "middle" }, fmt(t)));
  }
  for (const t of logY ? logTicks(y0, y1) : linTicks(y0, y1)) {
    svg.append(s("line", { x1: L, x2: width - R, y1: y(t), y2: y(t), stroke: GRID }),
               s("text", { x: L - 6, y: y(t) + 4, fill: MUTED, "font-size": 11,
                           "text-anchor": "end" }, fmt(t)));
  }
  svg.append(s("text", { x: (L + width - R) / 2, y: height - 6, fill: MUTED, "font-size": 11.5,
                         "text-anchor": "middle" }, xlabel),
             s("text", { x: 12, y: (T + height - B) / 2, fill: MUTED, "font-size": 11.5,
                         "text-anchor": "middle",
                         transform: `rotate(-90 12 ${(T + height - B) / 2})` }, ylabel));
  if (hline) {
    svg.append(s("line", { x1: L, x2: width - R, y1: y(hline.value), y2: y(hline.value),
                           stroke: hline.color ?? "#e6e9ee", "stroke-dasharray": "5 4",
                           opacity: 0.7 }),
               s("text", { x: width - R - 4, y: y(hline.value) - 5, fill: INK, "font-size": 11,
                           "text-anchor": "end" }, hline.label));
  }
  const clip = `clip${++clipId}`;
  svg.append(s("clipPath", { id: clip }, s("rect", { x: L, y: T, width: width - R - L,
                                                      height: height - B - T })));
  const plot = s("g", { "clip-path": `url(#${clip})` });
  svg.append(plot);
  series.forEach((d, i) => {
    const ok = d.points.filter(([a, b]) => isFinite(a) && isFinite(b) && (!logX || a > 0)
      && (!logY || b > 0));
    if (ok.length > 1) {
      plot.append(s("polyline", { points: ok.map(([a, b]) => `${x(a)},${y(b)}`).join(" "),
                                  fill: "none", stroke: d.color, "stroke-width": 2,
                                  "stroke-dasharray": d.dash ?? null, opacity: 0.95 }));
    }
    if (d.marker !== false) {
      for (const [a, b] of ok) {
        plot.append(s("circle", { cx: x(a), cy: y(b), r: 3.6, fill: d.color },
                     tip(`${d.label}: ${fmt(a)}, ${fmt(b)}`)));
      }
    }
    if (legend) {
      const ly = T + 8 + i * 18;
      svg.append(s("line", { x1: width - R + 12, x2: width - R + 30, y1: ly, y2: ly,
                             stroke: d.color, "stroke-width": 2.5,
                             "stroke-dasharray": d.dash ?? null }),
                 s("text", { x: width - R + 36, y: ly + 4, fill: INK, "font-size": 11.5 },
                   d.label));
    }
  });
  return svg;
}

/** Outcome matrix (HTML grid). cell(row, col) -> {text, tone, title, onclick} | null. */
export function matrix(rowLabels, cols, cell, { corner = "", section = null } = {}) {
  const tbl = document.createElement("table");
  tbl.className = "matrix";
  const head = document.createElement("tr");
  head.append(Object.assign(document.createElement("th"), { textContent: corner }));
  for (const c of cols) {
    const th = document.createElement("th");
    th.innerHTML = `<span style="color:${c.color ?? INK}">■</span> ${c.label}`;
    head.append(th);
  }
  tbl.append(head);
  rowLabels.forEach((r, i) => {
    const head = section?.(i);
    if (head) {
      const sr = document.createElement("tr");
      sr.className = "section";
      const th = Object.assign(document.createElement("th"), { colSpan: cols.length + 1 });
      th.append(head);
      sr.append(th);
      tbl.append(sr);
    }
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.className = "row";
    if (typeof r === "string") th.textContent = r;
    else th.append(r);
    tr.append(th);
    cols.forEach((c, j) => {
      const td = document.createElement("td");
      const v = cell(i, j);
      if (v) {
        td.className = `tone-${v.tone ?? "none"}${v.onclick ? " clickable" : ""}`;
        td.textContent = v.text;
        if (v.title) td.title = v.title;
        if (v.onclick) td.addEventListener("click", v.onclick);
      } else {
        td.className = "tone-empty";
      }
      tr.append(td);
    });
    tbl.append(tr);
  });
  const wrap = document.createElement("div");
  wrap.className = "tbl-wrap";
  wrap.append(tbl);
  return wrap;
}

/** Legend chips. items: [{label, color}] */
export function legend(items) {
  const d = document.createElement("div");
  d.className = "legend";
  for (const it of items) {
    const span = document.createElement("span");
    span.innerHTML = `<i style="background:${it.color}"></i>${it.label}`;
    d.append(span);
  }
  return d;
}
