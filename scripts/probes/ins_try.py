"""Probe: one insertion run with its metrics and a depth/lateral timeline.
Usage: python ins_try.py <route> ['<json overrides>']

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/ins_try.py ...
"""
import sys, time, json
from simbench.scenes import make
from simbench.run import simulate
from simbench.metrics import METRICS, insertion_depth
route = sys.argv[1]; kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
sc = make("insertion", **kw)
t0 = time.time(); tr = simulate(route, sc)
print(route, kw, tr.meta["status"], tr.meta["error"], f"{time.time()-t0:.1f}s")
print(json.dumps(METRICS["insertion"](tr)))
d, lat = insertion_depth(tr)
for t in (0.41, 0.51, 0.81, 1.01, 1.31, 1.51, 1.61, 1.91, 2.51, 3.01, 3.51, 3.99):
    k = min(int(round(t / sc.dt)), len(d) - 1); print(f"  t={t:.2f} depth {d[k]*1e3:7.3f} mm lat {lat[k]*1e3:.3f}")
