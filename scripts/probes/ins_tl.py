"""Probe: tilted 0.1 mm insertion timeline against the article's MuJoCo CPU trace.
Usage: python ins_tl.py <route>

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/ins_tl.py ...
"""
import gzip, json, numpy as np, sys
from simbench.scenes import make
from simbench.run import simulate
from simbench.metrics import insertion_depth, contact_force_on
d = json.load(gzip.open('reference/article/assets/insertion/replay-c100-tilted-r0.json.gz'))
art = {x['engine']: x for x in d['traces']}['mujoco_cpu']['frames']
sc = make("insertion", clearance_mm=0.1, case="tilted"); tr = simulate(sys.argv[1], sc)
dep, lat = insertion_depth(tr); h = tr.body("hand")
F = np.linalg.norm(contact_force_on(tr, "peg", ("world",)), axis=1)
for ta in np.arange(1.51, 2.62, 0.05):
    k = int(round((ta - 0.01) / 0.01)); fr = art[k]; b = {x['id']: x for x in fr['bodies']}
    fa = np.linalg.norm(sum(np.array(f['vector_N']) for f in b['paddle']['forces']))
    j = int(round(ta / sc.dt)) - 1
    print(f"{fr['time_s']:.2f} art x {b['hand']['position_m'][0]*1e6-306891:7.1f}um d {fr['insertion_m']*1e3:6.2f} F {fa:6.2f} | ours x {tr.pos[j,h,0]*1e6-306891:7.1f}um d {dep[j]*1e3:6.2f} F {F[j]:6.2f}")
