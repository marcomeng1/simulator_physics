"""Probe: permanent set of the flex ball vs yield strain under plate compression.

Run on a MuJoCo >= 4.0 route, e.g.
    uv run --locked --project envs/mujoco4 python scripts/probes/plastic_probe.py 0 0.01 0.03
Prints the hold plate load, the extent along the squeeze axis at rest, at the hold and
after release, and the permanent set (rest - final).
"""

import sys

import mujoco
import numpy as np

from simbench.scenes.deform import ball_bodies, compress, squeeze


def run(yield_strain: float, legacy: bool = False) -> dict:
    scene = compress(yield_strain=yield_strain, legacy=legacy)
    m = mujoco.MjModel.from_xml_string(scene.mjcf)
    d = mujoco.MjData(m)
    ids = [m.body(b).id for b in ball_bodies()]
    lo, hi = m.joint("plate_lo"), m.joint("plate_hi")
    extent = lambda: float(np.ptp(d.xpos[ids, 2]))  # noqa: E731
    mujoco.mj_forward(m, d)
    rest, hold, load = extent(), [], []
    for _ in range(round(scene.duration / scene.dt)):
        z, zd = squeeze(d.time)
        ref = np.array([z, -z])
        q = np.array([d.qpos[lo.qposadr[0]], d.qpos[hi.qposadr[0]]])
        qd = np.array([d.qvel[lo.dofadr[0]], d.qvel[hi.dofadr[0]]])
        d.qfrc_applied[[lo.dofadr[0], hi.dofadr[0]]] = 1e5 * (ref - q) + 600 * (
            np.array([zd, -zd]) - qd)
        mujoco.mj_step(m, d)
        if 0.6 <= d.time <= 0.8:
            hold.append(extent())
            load.append(abs(d.qfrc_constraint[hi.dofadr[0]]))
    final = extent()
    return {"yield": yield_strain, "legacy": legacy, "nplastic": int(m.nplastic),
            "hold_load_N": float(np.mean(load)), "rest_mm": rest * 1e3,
            "hold_mm": float(np.mean(hold)) * 1e3, "final_mm": final * 1e3,
            "set_mm": (rest - final) * 1e3}


if __name__ == "__main__":
    args = sys.argv[1:] or ["0", "0.01", "0.03", "0.1"]
    for a in args:
        r = run(0.0, legacy=True) if a == "legacy" else run(float(a))
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in r.items()), flush=True)
