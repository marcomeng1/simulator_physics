// uPlot helpers: dark theme, a time cursor line, click-to-seek and log-log charts.

import uPlot from "uplot";

const axis = { stroke: "#8d96a3", grid: { stroke: "#232a32" }, ticks: { stroke: "#2a3038" } };

function cursorLine(getT) {
  return {
    hooks: {
      draw: [(u) => {
        const t = getT();
        if (t == null) return;
        const x = u.valToPos(t, "x", true);
        const { ctx } = u;
        ctx.save();
        ctx.strokeStyle = "#e6e9ee";
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, u.bbox.top);
        ctx.lineTo(x, u.bbox.top + u.bbox.height);
        ctx.stroke();
        ctx.restore();
      }],
    },
  };
}

export function timePlot(el, { t, series, unit, getT, onSeek }) {
  el.innerHTML = "";
  const opts = {
    width: el.clientWidth - 12, height: 230,
    scales: { x: { time: false } },
    axes: [{ ...axis, label: "time [s]" }, { ...axis, label: unit, size: 70 }],
    series: [{ label: "t", value: (u, v) => (v == null ? "" : v.toFixed(3)) },
             ...series.map((s) => ({ label: s.label, stroke: s.color, width: 1.6,
                                     value: (u, v) => (v == null ? "" : v.toPrecision(5)) }))],
    plugins: [cursorLine(getT)],
    cursor: { drag: { x: true, y: false } },
  };
  const u = new uPlot(opts, [t, ...series.map((s) => s.values)], el);
  u.over.addEventListener("click", (e) => onSeek(u.posToVal(e.offsetX, "x")));
  new ResizeObserver(() => u.setSize({ width: el.clientWidth - 12, height: 230 })).observe(el);
  return u;
}

export function logPlot(el, { x, series, xlabel, ylabel, logY = true }) {
  el.innerHTML = "";
  const opts = {
    width: el.clientWidth - 12, height: 300,
    scales: { x: { time: false, distr: 3 }, y: { distr: logY ? 3 : 1 } },
    axes: [{ ...axis, label: xlabel }, { ...axis, label: ylabel, size: 80,
             values: (u, vals) => vals.map((v) => (v == null ? "" : fmtSI(v))) }],
    series: [{ label: xlabel, value: (u, v) => (v == null ? "" : v.toLocaleString()) },
             ...series.map((s) => ({ label: s.label, stroke: s.color, width: 2,
                                     points: { show: true, size: 6, fill: s.color },
                                     value: (u, v) => (v == null ? "—" : fmtSI(v)) }))],
  };
  const u = new uPlot(opts, [x, ...series.map((s) => s.values)], el);
  new ResizeObserver(() => u.setSize({ width: el.clientWidth - 12, height: 300 })).observe(el);
  return u;
}

export function fmtSI(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toPrecision(3)}G`;
  if (a >= 1e6) return `${(v / 1e6).toPrecision(3)}M`;
  if (a >= 1e3) return `${(v / 1e3).toPrecision(3)}k`;
  return a >= 1 ? v.toPrecision(3) : v.toPrecision(2);
}
