"""Probe: when and where ball tetrahedra invert in the grasp, with pad contact forces.
Usage: python deform_inv.py <route> '<json overrides>' <duration_s>

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/deform_inv.py ...
"""
import sys, json, numpy as np
from simbench.scenes import make
from simbench.run import simulate
from simbench.metrics import ball_vertices, contact_force_on
route = sys.argv[1]; kw = json.loads(sys.argv[2]); dur = float(sys.argv[3])
sc = make("ball-grasp", **kw); sc.duration = dur
tr = simulate(route, sc)
v = ball_vertices(tr); tets = np.asarray(sc.params["tets"])
a, b, c, d = (v[:, tets[:, i]] for i in range(4))
vol = np.einsum("nti,nti->nt", np.cross(b - a, c - a), d - a) / 6
rel = vol / vol[0]
names = tr.meta["contact_names"]
pad = np.linalg.norm(contact_force_on(tr, "left_finger", ("ball",)), axis=1)
tab = np.linalg.norm(contact_force_on(tr, "ball", ("world",)), axis=1) if "world" in names else None
for k in range(0, len(tr.t), 25):
    i = rel[k].argmin()
    print(f"t={tr.t[k]:.3f} minrel {rel[k].min():+.3f} tet {i} verts {tets[i].tolist()} z {v[k, tets[i], 2].round(4).tolist()} fingers {tr.q[k,7:].round(4).tolist()} padF {pad[k]:.2f}")
