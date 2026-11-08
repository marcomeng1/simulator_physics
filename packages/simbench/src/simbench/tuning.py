"""Phase 6 controller tuning on the push task (article section 6).

Each route evaluates the same 27 candidates (arm KP x {0.5, 1, 2}, arm KD x {0.7, 1, 1.4},
push duration {2, 3, 4} s) on three 90 mm selection layouts and picks one with the
article's frozen rule: most successful layouts, then most passed objects, then highest
capped minimum-object progress, then lowest mean normalised squared arm effort. The
selected candidate and the baseline (1x, 1x, 3 s) then run on six held-out layouts
(four 90 mm, two 80 mm), and a speed control keeps baseline gains while shortening
the push. The article's layouts are not published, so these are defined here as
(width, rear-cube offset, front-pair offset) in mm.
"""

import json
from itertools import product

from simbench.metrics import METRICS
from simbench.paths import SUMMARY
from simbench.run import simulate, trace_stem
from simbench.scenes import make
from simbench.trace import Trace

KP_SCALES, KD_SCALES, PUSH_S = (0.5, 1.0, 2.0), (0.7, 1.0, 1.4), (2.0, 3.0, 4.0)
BASELINE = (1.0, 1.0, 3.0)
SELECTION = [(90.0, 0.0, 0.0), (90.0, 2.0, 0.0), (90.0, 0.0, 1.0)]
HELD_OUT = [(90.0, -3.0, 0.0), (90.0, 3.0, 0.0), (90.0, 0.0, -1.0), (90.0, -2.0, -1.0),
            (80.0, -1.0, 0.0), (80.0, 1.0, 0.0)]
SPEED_S = (3.0, 2.0, 1.5, 1.0, 0.75)
KEYS = ("all_passed", "passed_count", "min_progress_capped", "mean_norm_sq_effort",
        "max_hand_height_change_mm", "max_pad_push_N")


def name(c: tuple[float, float, float]) -> str:
    return f"p{c[0]:g}-d{c[1]:g}-t{c[2]:g}"


def layout_name(layout: tuple[float, float, float]) -> str:
    w, rear, front = layout
    return f"w{w:g}-r{rear:+g}-f{front:+g}"


def trial(route: str, group: str, cand, layout, dt: float = 0.001) -> dict:
    w, rear, front = layout
    scene = make("push", dt=dt, width_mm=w, rear_mm=rear, layout_mm=front,
                 kp_scale=cand[0], kd_scale=cand[1], push_s=cand[2])
    tag = f"tune-{group}-{name(cand)}-{layout_name(layout)}"
    stem = trace_stem(scene, route, 0, tag)
    if stem.with_suffix(".json").exists():
        tr = Trace.load(stem)
    else:
        tr = simulate(route, scene, tag=tag)
        tr.save(stem)
    m = METRICS["push"](tr) if tr.meta["status"] == "ok" else {}
    return {"status": tr.meta["status"], "variant": stem.parent.name,
            **{k: m.get(k) for k in KEYS}}


def score(trials: list[dict]) -> tuple:
    """Sort key: larger is better on every component."""
    ok = [t for t in trials if t["status"] == "ok"]
    if len(ok) < len(trials):
        return (-1, 0, 0.0, 0.0)
    return (sum(t["all_passed"] for t in ok), sum(t["passed_count"] for t in ok),
            sum(t["min_progress_capped"] for t in ok) / len(ok),
            -sum(t["mean_norm_sq_effort"] for t in ok) / len(ok))


def tune(route: str) -> dict:
    candidates = list(product(KP_SCALES, KD_SCALES, PUSH_S))
    selection = {}
    for c in candidates:
        selection[name(c)] = [trial(route, "sel", c, lay) for lay in SELECTION]
        s = score(selection[name(c)])
        print(f"{route:<10} {name(c):<14} successes {s[0]} passed {s[1]} "
              f"effort {-s[3]:.6f}", flush=True)
    ranked = sorted(candidates, key=lambda c: score(selection[name(c)]), reverse=True)
    best = ranked[0]
    held = {name(c): [trial(route, "test", c, lay) for lay in HELD_OUT]
            for c in dict.fromkeys([BASELINE, best])}
    speed = {f"{s:g}": [trial(route, "speed", (1.0, 1.0, s), lay) for lay in SELECTION]
             for s in SPEED_S}
    effort = {name(c): -score(selection[name(c)])[3] for c in candidates}
    out = {"route": route, "selected": name(best), "runner_up": name(ranked[1]),
           "margin_vs_runner_up": effort[name(ranked[1])] / effort[name(best)] - 1,
           "ranking": [name(c) for c in ranked], "selection_layouts": SELECTION,
           "held_out_layouts": HELD_OUT, "selection": selection, "held_out": held,
           "speed": speed}
    path = SUMMARY / "tuning" / f"{route}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    for c, trials in held.items():
        print(f"{route:<10} held-out {c:<14} {sum(t['all_passed'] for t in trials)}/6, "
              f"peak hand height change "
              f"{max(t['max_hand_height_change_mm'] or 0 for t in trials):.1f} mm")
    print(f"selected {out['selected']} (runner-up {out['runner_up']}, "
          f"{out['margin_vs_runner_up'] * 100:.2f}% more effort)")
    return out
