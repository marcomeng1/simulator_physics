"""Probe: our MuJoCo 3.11 sweep against the article's Repeat 1 MuJoCo trace.

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/sweep_cmp.py ...
"""
import json, numpy as np
from simbench.scenes import make
from simbench.run import simulate
from simbench.metrics import sweep_metrics
from simbench.reference import article_traces, SWEEP_REPEATS
from simbench import tapes
sc = make("sweep")
tr = simulate("mujoco311", sc, seed=0, tag="probe")
print(json.dumps(sweep_metrics(tr)))
art = article_traces("sweep", SWEEP_REPEATS[1])["mujoco_cpu"]
k = np.round(art.t / 0.001).astype(int) - 1  # article frames are post-step
k[0] = 0
objs = [b for b in art.bodies if b.startswith("object_")]
for name in ["hand", "paddle"] + objs[:15]:
    d = np.linalg.norm(tr.pos[k, tr.body(name)] - art.pos[:, art.body(name)], axis=1)
    first = art.t[np.argmax(d > 1e-3)] if (d > 1e-3).any() else None
    print(f"{name:10s} max {d.max()*1e3:8.3f} mm  first>1mm {first}")
tape = tapes.load(sc.params["tape_spec"])
print("---")
pd = np.linalg.norm(tr.pos[k, tr.body("paddle")] - art.pos[:, art.body("paddle")], axis=1)
for T in [0.5, 1, 2, 3, 3.4, 3.8, 4.2, 5, 7, 9, 11, 12.5]:
    i = np.searchsorted(art.t, T); print(f"t={T:5.2f} paddle err {pd[i]*1e3:7.3f} mm")
for o in objs:
    z = art.pos[:, art.body(o)]; mv = np.linalg.norm(z - z[0], axis=1)
    i = np.argmax(mv > 1e-4); d = np.linalg.norm(tr.pos[k, tr.body(o)] - z, axis=1)
    print(o, "article starts moving", art.t[i], "our err then", f"{d[i]*1e3:.3f}", "initial err", f"{d[0]*1e6:.1f}um")
