"""Pinch-and-twist slip torque on one route for a few contact settings.

uv run --locked --project envs/mujoco4 python scripts/probes/twist_probe.py mujoco4
"""

import sys

from simbench.metrics import twist_metrics
from simbench.run import simulate
from simbench.scenes import patch

route = sys.argv[1] if len(sys.argv) > 1 else "mujoco4"
cases = [("disc condim3", {}, {}), ("disc condim6 default", {"condim": 6}, {}),
         ("disc condim6 matched", {"condim": 6, "torsion": 2 / 3 * 0.5 * 0.02}, {}),
         ("disc multiccd", {"multiccd": True}, {}),
         ("square condim3", {"shape": "square"}, {}),
         ("square multiccd", {"shape": "square", "multiccd": True}, {})]
if route == "newton":
    fix = {"mujoco_invweight": True}
    big = {"naconmax": 2048, "njmax": 8192}
    cases = [(f"{n} +iw", kw, sim | fix) for n, kw, sim in cases[:1] + cases[4:5]]
    cases += [("disc newton-point +iw", {}, {"contacts": "newton"} | fix),
              ("disc hydro", {}, {"contacts": "hydroelastic"} | big),
              ("disc hydro +iw", {}, {"contacts": "hydroelastic"} | big | fix),
              ("square hydro +iw", {"shape": "square"}, {"contacts": "hydroelastic"} | big | fix),
              ("disc hydro full +iw", {}, {"contacts": "hydroelastic", "reduce": False}
               | big | fix)]
for name, kw, sim in cases:
    scene = patch.twist(**kw)
    tr = simulate(route, scene, sim_kwargs=sim)
    if tr.meta["status"] != "ok":
        print(f"{name:24s} {tr.meta['status']} {tr.meta.get('error')}")
        continue
    m = twist_metrics(tr)
    print(f"{name:24s} slip {m['slip_torque_Nm']} ref {m['slip_torque_ref_Nm']:.4f} "
          f"ratio {m['slip_ratio']} contacts {m['contacts_per_step']:.1f} "
          f"pad speed {m['pad_speed_max']:.2e}")
