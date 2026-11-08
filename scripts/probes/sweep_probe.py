"""Probe: rebuild the article's sweep joint tape and layout from its bundle.

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/sweep_probe.py ...
"""
import gzip, json, numpy as np, mujoco
from simbench.models import panda_tree, write, ARM_JOINTS, FINGER_JOINTS
from simbench.tapes import TAPE_DIR
d = json.load(gzip.open("reference/article/assets/sweep-comparison/repeat-1.json.gz"))
t = {x['engine']: x for x in d['traces']}['mujoco_cpu']
m = mujoco.MjModel.from_xml_path(str(write(panda_tree("pads", TAPE_DIR), TAPE_DIR / "probe.xml")))
def qmul(a, b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
conj = lambda q: q*np.array([1,-1,-1,-1])
def rot(q):
    r=np.zeros(9); mujoco.mju_quat2Mat(r, np.asarray(q,float)); return r.reshape(3,3)
names = ["link0","link1","link2","link3","link4","link5","link6","link7","hand","left_finger","right_finger","paddle"]
Q = []
for fr in t['frames']:
    b = {x['id']: x for x in fr['bodies']}
    q = []
    for i, j in enumerate(ARM_JOINTS):
        jid = m.joint(j).id; bid = m.jnt_bodyid[jid]
        parent = m.body(m.body_parentid[bid]).name; child = m.body(bid).name
        rel = qmul(conj(np.array(m.body_quat[bid])), qmul(conj(np.array(b[parent]['quaternion_wxyz'])), np.array(b[child]['quaternion_wxyz'])))
        ax = m.jnt_axis[jid]; q.append(2*np.arctan2(rel[1:]@ax, rel[0]))
    Rh = rot(b['hand']['quaternion_wxyz']); ph = np.array(b['hand']['position_m'])
    for j in FINGER_JOINTS:
        jid = m.joint(j).id; bid = m.jnt_bodyid[jid]
        local = Rh.T @ (np.array(b[m.body(bid).name]['position_m']) - ph) - m.body_pos[bid]
        q.append(local @ (rot(m.body_quat[bid]) @ m.jnt_axis[jid]))
    Q.append(q)
Q = np.array(Q); T = np.array([f['time_s'] for f in t['frames']])
np.set_printoptions(precision=5, suppress=True, linewidth=160)
print("t", T[:3], T[-2:]); print("q0", Q[0]); print("q min", Q.min(0)); print("q max", Q.max(0))
b = {x['id']: x for x in t['frames'][0]['bodies']}
Rh = rot(b['hand']['quaternion_wxyz']); ph = np.array(b['hand']['position_m'])
print("paddle in hand", Rh.T @ (np.array(b['paddle']['position_m']) - ph), qmul(conj(np.array(b['hand']['quaternion_wxyz'])), np.array(b['paddle']['quaternion_wxyz'])))
# FK check: set q and compare link positions
dd = mujoco.MjData(m); err = 0
for k in range(0, len(Q), 25):
    dd.qpos[:] = Q[k]; mujoco.mj_kinematics(m, dd)
    bb = {x['id']: x for x in t['frames'][k]['bodies']}
    for n in names[:11]:
        err = max(err, np.linalg.norm(dd.xpos[m.body(n).id] - bb[n]['position_m']))
print("max FK error", err)
print("joint speeds max", np.abs(np.diff(Q, axis=0)/0.02).max(0))
