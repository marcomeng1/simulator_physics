"""Normal force per face and angle vs torque for the twist scene on one route.

uv run --locked --project envs/newton python scripts/probes/twist_debug.py newton hydroelastic
"""

import sys

import numpy as np
from simbench.run import simulate
from simbench.scenes import patch

route = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "-"
extra = dict(a.split("=") for a in sys.argv[3:])
shape = extra.pop("shape", "disc")
sim = {k: int(v) if k == "naconmax" else float(v) for k, v in extra.items()}
if mode != "-":
    sim["contacts"] = mode
tr = simulate(route, patch.twist(shape=shape), sim_kwargs=sim)
print(tr.meta["status"], tr.meta.get("error"))
puck = tr.meta["contact_names"].index("puck")
for k in (150, 400, 600, 800, 1000, 1200, 1400):
    sel = tr.c_frame == k
    f = tr.c_force[sel]
    sign = np.where(tr.c_b[sel] == puck, 1.0, -1.0)[:, None]
    fy = (f * sign)[:, 1]
    print(f"t {tr.t[k]:.2f} tau {tr.tau[k, 2]:.3f} angle {np.degrees(tr.q[k, 2]):8.3f} deg "
          f"n {sel.sum():4d} Fy+ {fy[fy > 0].sum():6.2f} Fy- {fy[fy < 0].sum():6.2f} "
          f"|F| {np.linalg.norm(f, axis=1).sum():6.2f} pads {tr.q[k, 0]:.2e} {tr.q[k, 1]:.2e}")
