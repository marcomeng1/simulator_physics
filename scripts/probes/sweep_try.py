"""Probe: one sweep run on a route with metrics (and the article comparison on mujoco4).
Usage: python sweep_try.py <route>

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/sweep_try.py ...
"""
import time, json, numpy as np
from simbench.scenes import make
from simbench.run import simulate
from simbench.metrics import sweep_metrics
from simbench.reference import article_traces, SWEEP_REPEATS
import sys
route = sys.argv[1]
sc = make("sweep")
t0 = time.time(); tr = simulate(route, sc, seed=0, tag="probe"); print(route, tr.meta["status"], tr.meta.get("error"), f"{time.time()-t0:.1f}s")
if tr.meta["status"] == "ok":
    print(json.dumps(sweep_metrics(tr)))
if route == "mujoco4":
    for rep, path in SWEEP_REPEATS.items():
        for eng, a in article_traces("sweep", path).items():
            a.meta["params"] = sc.params
            m = sweep_metrics(a); print("article", rep, eng, m["collected"], m["fallen"], m["collected_ids"])
