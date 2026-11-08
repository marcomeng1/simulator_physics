"""Frozen reference and feedforward tapes for the Panda scenes.

As in the article, tapes are generated once on the MuJoCo 3.11 route (`make tapes`) and
every engine replays the same numbers at the 1 kHz control rate: the joint reference,
its velocity and the unloaded rigid-body inverse dynamics along it. A tape is addressed
by the sha256 of its JSON spec, so a scene can never silently pick up a stale tape.

Spec kinds
  bias       gravity/Coriolis torque at rest at `q0` (the open-loop effort baseline).
  joint      quintic out-and-back joint-space path from `q0` by `offset`.
  cartesian  TCP waypoints joined by quintic straight lines at a fixed downward hand
             orientation, solved by damped least-squares IK at every control sample,
             plus independent quintic finger references.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from simbench.models import ARM_JOINTS, FINGER_JOINTS, Q_READY, panda_tree, write
from simbench.paths import ASSETS

TAPE_DIR = ASSETS / "generated" / "tapes"


def spec_sha(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def tape_path(spec: dict) -> Path:
    return TAPE_DIR / f"{spec['name']}-{spec_sha(spec)[:12]}.npz"


def load(spec: dict) -> dict[str, np.ndarray]:
    path = tape_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing; run `make tapes` (MuJoCo 3.11 route)")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def quintic(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest-to-rest quintic blend s(tau) on [0, 1] and its first two derivatives."""
    tau = np.clip(tau, 0.0, 1.0)
    s = tau**3 * (10 - 15 * tau + 6 * tau**2)
    ds = 30 * tau**2 * (1 - tau) ** 2
    dds = 60 * tau * (1 - tau) * (1 - 2 * tau)
    return s, ds, dds


def piecewise(t: np.ndarray, start, segments) -> np.ndarray:
    """Value held at `start`, then quintic moves `[t0, t1, end]`, held between moves."""
    start = np.asarray(start, dtype=float)
    out = np.broadcast_to(start, (len(t), *start.shape)).copy()
    value = start
    for t0, t1, end in segments:
        end = np.asarray(end, dtype=float)
        s, _, _ = quintic((t - t0) / (t1 - t0))
        active = t >= t0
        out[active] = value + np.multiply.outer(s[active], end - value)
        value = end
    return out


def control_times(spec: dict) -> np.ndarray:
    n = int(round(spec["duration"] / spec["ctrl_dt"])) + 1
    return np.arange(n) * spec["ctrl_dt"]


def _robot(variant: str):
    import mujoco

    path = write(panda_tree(variant, TAPE_DIR), TAPE_DIR / f"robot_{variant}.xml")
    m = mujoco.MjModel.from_xml_path(str(path))
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return m, mujoco.MjData(m)


def _inverse(m, d, qpos, qvel, qacc) -> np.ndarray:
    import mujoco

    out = np.zeros_like(qpos)
    for k in range(len(qpos)):
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos[k], qvel[k], qacc[k]
        mujoco.mj_inverse(m, d)
        out[k] = d.qfrc_inverse
    return out


def down_orientation(m, d, q_seed) -> np.ndarray:
    """Hand pointing straight down, keeping the yaw the seed configuration has."""
    import mujoco

    d.qpos[: len(q_seed)] = q_seed
    mujoco.mj_kinematics(m, d)
    x = d.xmat[m.body("hand").id].reshape(3, 3)[:, 0].copy()
    x[2] = 0.0
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def ik(m, d, site: str, target_pos, target_rot, q0, q_rest=Q_READY, tol=1e-12,
       damping=1e-6, max_step=0.2, iters=500):
    """Damped least-squares IK for the seven arm joints within their ranges.

    The redundant DOF is pulled towards `q_rest` in the Jacobian null space, so the
    solution is a smooth function of the target. Fingers are left untouched.
    """
    import mujoco

    sid = m.site(site).id
    arm = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
    lo, hi = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS]].T
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    q = np.array(q0, dtype=float)
    polish = False
    for _ in range(iters):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rot = d.site_xmat[sid].reshape(3, 3)
        err = np.concatenate([target_pos - d.site_xpos[sid],
                              0.5 * np.cross(rot, target_rot, axis=0).sum(axis=1)])
        if np.linalg.norm(err) < tol:
            return q
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        jac = np.vstack([jacp, jacr])[:, arm]
        pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping * np.eye(6))
        dq = pinv @ err
        polish = polish or np.linalg.norm(err) < 1e-8
        if not polish:
            null = np.eye(7) - np.linalg.pinv(jac) @ jac
            dq += null @ (0.5 * (np.asarray(q_rest) - q[arm]))
        dq *= min(1.0, max_step / np.abs(dq).max())
        q[arm] = np.clip(q[arm] + dq, lo, hi)
    raise RuntimeError(f"IK did not converge: |err|={np.linalg.norm(err):.3e} at {target_pos}")


def _bias(spec: dict) -> dict[str, np.ndarray]:
    import mujoco

    m, d = _robot(spec["model"])
    d.qpos[:] = spec["q0"]
    mujoco.mj_forward(m, d)
    return {"bias": d.qfrc_bias.copy()}


def _joint(spec: dict) -> dict[str, np.ndarray]:
    m, d = _robot(spec["model"])
    t = control_times(spec)
    half = spec["duration"] / 2
    q0, offset = np.asarray(spec["q0"]), np.asarray(spec["offset"])
    out_s, out_ds, out_dds = quintic(t / half)
    back_s, back_ds, back_dds = quintic((t - half) / half)
    first = t <= half
    s = np.where(first, out_s, 1 - back_s)
    ds = np.where(first, out_ds, -back_ds) / half
    dds = np.where(first, out_dds, -back_dds) / half**2
    qref, qdref, qddref = (q0 + np.outer(s, offset), np.outer(ds, offset),
                           np.outer(dds, offset))
    return {"t": t, "qref": qref, "qdref": qdref, "qddref": qddref,
            "ff": _inverse(m, d, qref, qdref, qddref)}


def _cartesian(spec: dict) -> dict[str, np.ndarray]:
    """TCP waypoints; each move is a quintic in joint space ("joint") or along the
    straight TCP line ("line", IK at every control sample)."""
    import mujoco

    m, d = _robot("pads")
    t = control_times(spec)
    rot = down_orientation(m, d, spec["q_seed"])
    arm = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
    fing = np.array([m.jnt_qposadr[m.joint(j).id] for j in FINGER_JOINTS])

    q = np.zeros(m.nq)
    q[arm] = spec["q_seed"]
    q = ik(m, d, "tcp", np.asarray(spec["tcp_start"]), rot, q)
    qref = np.broadcast_to(q, (len(t), m.nq)).copy()
    p_prev = np.asarray(spec["tcp_start"], dtype=float)
    for t0, t1, p_end, space in spec["tcp_segments"]:
        p_end = np.asarray(p_end, dtype=float)
        s, _, _ = quintic((t - t0) / (t1 - t0))
        if space == "joint":
            q_end = ik(m, d, "tcp", p_end, rot, q)
            qref[t >= t0] = q + np.outer(s[t >= t0], q_end - q)
        else:
            q_end = q
            for k in np.flatnonzero((t >= t0) & (t <= t1)):
                q_end = qref[k] = ik(m, d, "tcp", p_prev + s[k] * (p_end - p_prev), rot, q_end)
            qref[t > t1] = q_end
        q, p_prev = q_end, p_end
    qref[:, fing] = piecewise(t, spec["finger_start"], spec["finger_segments"])

    dt = spec["ctrl_dt"]
    qdref = np.gradient(qref, dt, axis=0, edge_order=2)
    qddref = np.gradient(qdref, dt, axis=0, edge_order=2)
    tcp = np.zeros((len(t), 3))
    sid = m.site("tcp").id
    for k in range(len(t)):
        d.qpos[:] = qref[k]
        mujoco.mj_kinematics(m, d)
        tcp[k] = d.site_xpos[sid]
    return {"t": t, "qref": qref, "qdref": qdref, "qddref": qddref,
            "ff": _inverse(m, d, qref, qdref, qddref), "tcp": tcp, "hand_rot": rot}


GENERATORS = {"bias": _bias, "joint": _joint, "cartesian": _cartesian}


def generate(spec: dict) -> Path:
    import mujoco

    arrays = GENERATORS[spec["kind"]](spec)
    path = tape_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, spec=json.dumps(spec), mujoco=mujoco.__version__)
    return path
