"""Probe: recover the sweep paddle mount (hand frame) from the article bundle.

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/paddle_probe.py ...
"""
import gzip, json, numpy as np, mujoco
d = json.load(gzip.open("reference/article/assets/sweep-comparison/repeat-1.json.gz"))
for tr in d['traces']:
    rels=[]
    for fr in tr['frames'][::100]:
        b = {x['id']: x for x in fr['bodies']}
        r=np.zeros(9); mujoco.mju_quat2Mat(r, np.array(b['hand']['quaternion_wxyz'])); R=r.reshape(3,3)
        rels.append(R.T @ (np.array(b['paddle']['position_m']) - np.array(b['hand']['position_m'])))
    print(tr['engine'], np.array(rels).mean(0).round(7), np.ptp(rels,0).max())
    print('  paddle forces sample', [f['kind'] for f in tr['frames'][300]['bodies'][11]['forces']][:5], tr['frames'][300]['bodies'][11]['id'])
tr={t['engine']:t for t in d['traces']}['mujoco_cpu']; b={x['id']:x for x in tr['frames'][0]['bodies']}
r=np.zeros(9); mujoco.mju_quat2Mat(r, np.array(b['hand']['quaternion_wxyz'])); R=r.reshape(3,3)
print(repr((R.T @ (np.array(b['paddle']['position_m']) - np.array(b['hand']['position_m']))).tolist()))
