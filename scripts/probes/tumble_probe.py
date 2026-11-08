"""RK4 orientation order probe: a T-handle tumbling about its intermediate axis (no gravity).

MuJoCo 4.0 made RK4 fourth-order for free/ball orientations (changelog item 5). Compare the
final orientation error against a fine-step reference across timesteps on 3.11 and 4.0.
uv run --project envs/mujoco4 python scripts/probes/tumble_probe.py
"""

import mujoco
import numpy as np

XML = """<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="{integ}"/><worldbody>
<body name="t"><freejoint/><inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 4e-4 5e-4"/>
<geom type="capsule" size="0.01 0.05" contype="0" conaffinity="0"/></body></worldbody></mujoco>"""
W0 = np.array([0.05, 6.0, 0.0])
T = 4.0


def final(dt, integ):
    m = mujoco.MjModel.from_xml_string(XML.format(dt=dt, integ=integ))
    d = mujoco.MjData(m)
    d.qvel[3:] = W0
    for _ in range(int(round(T / dt))):
        mujoco.mj_step(m, d)
    q = d.qpos[3:7].copy()
    m_ang = m.body_inertia[1] * d.qvel[3:]   # body-frame angular momentum magnitude
    return q, np.linalg.norm(m_ang)


ref, L0 = final(1e-5, "RK4")
print("mujoco", mujoco.__version__)
for integ in ("RK4", "Euler", "implicitfast"):
    for dt in (0.01, 0.005, 0.002, 0.001):
        q, L = final(dt, integ)
        err = 2 * np.arccos(min(1.0, abs(float(q @ ref))))
        print(f"{integ:12} dt {dt * 1e3:4g} ms: orientation error {np.degrees(err):10.5f} deg, "
              f"|L| drift {abs(L - L0) / L0:.2e}")
