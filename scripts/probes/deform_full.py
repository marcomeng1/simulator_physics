"""Probe: Panda ball grasp on one route; per-frame tet volume ratio and mass-weighted lift.
Usage: python deform_full.py <route> '<json scene overrides>'

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/deform_full.py ...
"""
import sys, json, numpy as np
from simbench.scenes import make
from simbench.run import simulate
from simbench.metrics import ball_vertices
route = sys.argv[1]; kw = json.loads(sys.argv[2])
sc = make("ball-grasp", **kw); tr = simulate(route, sc)
v = ball_vertices(tr); tets = np.asarray(sc.params["tets"]); mass = np.asarray(sc.params["node_mass"])
a, b, c, d = (v[:, tets[:, i]] for i in range(4))
rel = (np.einsum("nti,nti->nt", np.cross(b - a, c - a), d - a) / 6)
rel = rel / rel[0]
bad = (rel <= 0).any(axis=1)
lift = (v[:, :, 2] @ mass - v[0, :, 2] @ mass) / mass.sum() * 1e3
hold = (tr.t >= 1.9) & (tr.t <= 2.2)
print(route, kw, tr.meta["status"], "inverted frames", int(bad.sum()), "times", tr.t[bad][:8].round(3).tolist(),
      f"hold lift (ungated) {lift[hold].mean():.1f} mm  final {lift[-1]:.1f} mm  min rel vol {rel.min():.3f}")
