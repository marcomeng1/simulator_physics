"""Thin-card probe: does a 0.3 mm card falling onto a thin plate tunnel, per speed/dt?

Tunneling = the card's centre passes below the plate's bottom face inside its footprint.
uv run --project envs/mujoco4 python scripts/probes/card_probe.py
"""

import sys

import mujoco

CARD = (0.044, 0.0315, 0.00015)   # half sizes: 88 x 63 x 0.3 mm poker card
MASS = 0.0018


def xml(dt, plate_half_t, card_half_t):
    return f"""<mujoco><option timestep="{dt}" integrator="Euler" solver="Newton"
      iterations="100" tolerance="1e-10" cone="elliptic"/>
    <worldbody><geom name="plate" type="box" pos="0 0 0.1" size="0.15 0.15 {plate_half_t}"
      friction="0.4 0 0" solref="0.01 1" solimp="0.99 0.999 0.001 0.5 2"/>
    <body name="card" pos="0 0 0.13"><freejoint/>
      <geom type="box" size="{CARD[0]} {CARD[1]} {card_half_t}" mass="{MASS}"
        friction="0.4 0 0" solref="0.01 1" solimp="0.99 0.999 0.001 0.5 2"/></body>
    </worldbody></mujoco>"""


def run(dt, v, w, spec, plate_t=0.001, card_t=0.0003):
    m = mujoco.MjModel.from_xml_string(xml(dt, plate_t / 2, card_t / 2))
    if spec:
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE)
    d = mujoco.MjData(m)
    d.qvel[2] = -v
    d.qvel[3] = w
    bottom = 0.1 - plate_t / 2
    for i in range(int(0.4 / dt)):
        mujoco.mj_step(m, d)
        x, y, z = d.qpos[:3]
        if z < bottom and abs(x) < 0.15 and abs(y) < 0.15:
            return f"TUNNEL@{(i + 1) * dt:.3f}s"
    return f"held z={(d.qpos[2] - 0.1 - plate_t / 2) * 1e3:.2f}mm"


mode = sys.argv[1] if len(sys.argv) > 1 else "thick"
if mode == "thick":
    for plate_t, card_t in ((0.001, 0.0003), (0.01, 0.0003), (0.001, 0.003), (0.02, 0.003)):
        for dt in (0.002, 0.001):
            for v in (0.1, 0.5, 1.0, 3.0):
                print(f"plate {plate_t * 1e3:g}mm card {card_t * 1e3:g}mm dt {dt * 1e3:g}ms "
                      f"v {v:3.1f}: off {run(dt, v, 0, False, plate_t, card_t):22} "
                      f"spec {run(dt, v, 0, True, plate_t, card_t)}")
else:
    for w in (0.0, 30.0, 100.0):
        for v in (0.5, 1.0, 2.0, 5.0, 10.0):
            print(f"w {w:5.0f} v {v:4.1f} dt 1ms: off {run(0.001, v, w, False):22} "
                  f"spec {run(0.001, v, w, True)}")
