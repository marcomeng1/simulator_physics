"""Probe: run one deformable scene and print its metrics.
Usage: python deform_try.py <route> <ball-compress|ball-grasp> ['<json overrides>']

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/deform_try.py ...
"""
import sys, time, json
from simbench.scenes import make
from simbench.run import simulate
from simbench.metrics import METRICS
route, name = sys.argv[1], sys.argv[2]; kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
sc = make(name, **kw); t0 = time.time(); tr = simulate(route, sc)
m = METRICS[name](tr)
print(route, name, kw, tr.meta["status"], tr.meta["error"], f"{time.time()-t0:.1f}s", tr.meta["frames"], "frames")
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}))
