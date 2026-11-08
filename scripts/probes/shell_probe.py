"""2D shell bending plasticity probe (MuJoCo 4.0): a cantilever strip pressed at the tip.

A 120 x 20 x 1 mm strip, clamped along x = 0, is pushed down 25 mm at its tip by a
capsule on a PD slide joint, held, and released. Elastic: the tip returns to 0.
Bending yield: it keeps a permanent set.
uv run --project envs/mujoco4 python scripts/probes/shell_probe.py [yield_bend] [dt]
"""

import sys
import time

import mujoco
import numpy as np

NX, NY, SP = 13, 3, 0.01
T, E, NU, RHO = 0.001, 2e9, 0.38, 1380.0
PRESS, KP, KD = 0.015, 4e5, 1200.0


E2D = "both"
INTEG = sys.argv[3] if len(sys.argv) > 3 else "discrete"


def build(dt, yb):
    vmass = RHO * T * (NX - 1) * (NY - 1) * SP**2 / (NX * NY)
    bodies, xml = [], []
    for j in range(NY):
        for i in range(NX):
            name = f"sheet_{j * NX + i}"
            bodies.append(name)
            pos = f"{i * SP} {(j - (NY - 1) / 2) * SP} 0"
            joints = "" if i <= 1 else "".join(
                f'<joint type="slide" axis="{a}"/>' for a in ("1 0 0", "0 1 0", "0 0 1"))
            xml.append(f'<body name="{name}" pos="{pos}"><inertial pos="0 0 0" mass="{vmass}" '
                       f'diaginertia="1e-10 1e-10 1e-10"/>{joints}</body>')
    tri = []
    for j in range(NY - 1):
        for i in range(NX - 1):
            a, b, c, d = j * NX + i, j * NX + i + 1, (j + 1) * NX + i, (j + 1) * NX + i + 1
            tri += [a, b, d, a, d, c]
    yld = f' yield="0 {yb}"' if yb else ""
    tip = (NX - 1) * SP
    return f"""<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="{INTEG}" solver="Newton"
      iterations="100" tolerance="1e-10" cone="elliptic"/><worldbody>
      <body name="pusher" pos="0.07 0 0.0095"><joint name="push" type="slide" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="1e-3 1e-3 1e-3"/>
        <geom type="sphere" size="0.008" contype="2" conaffinity="1"
          friction="0.3 0 0"/></body>
      {''.join(xml)}</worldbody>
      <deformable><flex name="sheet" dim="2" radius="{T / 2}" body="{' '.join(bodies)}"
        element="{' '.join(map(str, tri))}">
        <contact contype="1" conaffinity="2" condim="3" friction="0.3 0 0" solref="0.002 1" selfcollide="none"/>
        <elasticity young="{E}" poisson="{NU}" thickness="{T}" damping="1e-4"
          elastic2d="{E2D}"{yld}/></flex></deformable></mujoco>""", tri


def press(t):
    def s(t0, t1):
        x = min(max((t - t0) / (t1 - t0), 0), 1)
        return 3 * x * x - 2 * x**3
    return -(PRESS + 0.001) * (s(0.1, 0.6) - s(0.8, 1.3))


def run(dt, yb):
    xml, _ = build(dt, yb)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    tipb = [m.body(f"sheet_{j * NX + NX - 1}").id for j in range(NY)]
    t0, out = time.time(), {}
    for k in range(int(2.0 / dt)):
        t = k * dt
        d.qfrc_applied[0] = KP * (press(t) - d.qpos[0]) - KD * d.qvel[0]
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            return f"non-finite at {t:.3f}s"
        tz = d.xpos[tipb, 2].mean()
        if abs(t - 0.7) < dt / 2:
            out["held"] = tz
    out["final"] = d.xpos[tipb, 2].mean()
    return (f"tip held {out['held'] * 1e3:7.2f} mm, final {out['final'] * 1e3:7.2f} mm "
            f"({time.time() - t0:.1f}s wall, nplastic {m.nplastic})")


dt = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-5
for yb in ([float(sys.argv[1])] if len(sys.argv) > 1 else [0.0, 0.002, 0.001, 0.0005]):
    print(f"yield_bend {yb:g}: {run(dt, yb)}")
