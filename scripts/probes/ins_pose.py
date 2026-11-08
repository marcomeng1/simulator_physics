"""Probe: insertion hand pose drift and depth with insertion-module constants overridden.
Usage: python ins_pose.py <route> '<json overrides>' ['<json module constants>']

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/ins_pose.py ...
"""
import sys, json, numpy as np
import simbench.scenes.insertion as ins
from simbench.scenes import make
from simbench.run import simulate
from simbench.metrics import METRICS, quat_to_mat
route = sys.argv[1]; kw = json.loads(sys.argv[2]); over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
for k, v in over.items(): setattr(ins, k, v)
sc = make("insertion", **kw); tr = simulate(route, sc)
h = tr.body("hand"); R0, R1 = quat_to_mat(tr.quat[0, h]), quat_to_mat(tr.quat[-1, h])
rel = R0.T @ R1; ang = 0.5 * np.array([rel[2,1]-rel[1,2], rel[0,2]-rel[2,0], rel[1,0]-rel[0,1]])
m = METRICS["insertion"](tr)
print(over, kw["case"], tr.meta["status"], f"depth {m['final_depth_mm']:.2f} reached {m['depth_reached']} peakF {m.get('peak_contact_N',0):.2f} pen {m.get('peak_penetration_um',0):.2f}um",
      "dpos mm", ((tr.pos[-1,h]-tr.pos[0,h])*1e3).round(3), "rot mrad", (ang*1e3).round(3))
