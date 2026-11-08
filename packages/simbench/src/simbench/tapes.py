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
  samples    joint angles recovered from the link poses of a published article trace,
             joined by a clamped cubic spline (rest to rest).
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


def clamped_spline(x: np.ndarray, y: np.ndarray, t: np.ndarray):
    """Cubic spline through (x, y) with zero end slopes; value, first and second derivative
    at `t`. `y` may have several columns."""
    n, h = len(x), np.diff(x)
    a = np.zeros((n, n))
    rhs = np.zeros((n, *y.shape[1:]))
    slope = np.diff(y, axis=0) / h[:, None]
    a[0, :2] = 2 * h[0], h[0]
    rhs[0] = 6 * slope[0]
    a[-1, -2:] = h[-1], 2 * h[-1]
    rhs[-1] = -6 * slope[-1]
    for i in range(1, n - 1):
        a[i, i - 1:i + 2] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6 * (slope[i] - slope[i - 1])
    m = np.linalg.solve(a, rhs)
    i = np.clip(np.searchsorted(x, t, side="right") - 1, 0, n - 2)
    hi, lo, up = h[i][:, None], (x[i + 1] - t)[:, None], (t - x[i])[:, None]
    c0 = y[i] / hi - m[i] * hi / 6
    c1 = y[i + 1] / hi - m[i + 1] * hi / 6
    value = m[i] * lo**3 / (6 * hi) + m[i + 1] * up**3 / (6 * hi) + c0 * lo + c1 * up
    first = -m[i] * lo**2 / (2 * hi) + m[i + 1] * up**2 / (2 * hi) - c0 + c1
    second = m[i] * lo / hi + m[i + 1] * up / hi
    return value, first, second


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def recorded_joints(m, frames: list[dict]) -> np.ndarray:
    """Arm and finger joint positions that reproduce recorded link poses (wxyz quats)."""
    from simbench.readback import quat_to_mat

    conj = np.array([1.0, -1.0, -1.0, -1.0])
    out = np.zeros((len(frames), m.nq))
    for k, frame in enumerate(frames):
        pose = {b["id"]: b for b in frame["bodies"]}
        for j in range(m.njnt):
            bid = m.jnt_bodyid[j]
            child, parent = pose[m.body(bid).name], pose[m.body(m.body_parentid[bid]).name]
            if m.jnt_type[j] == 3:  # hinge: rotation about the axis left after the body offset
                rel = _qmul(m.body_quat[bid] * conj, _qmul(
                    np.asarray(parent["quaternion_wxyz"]) * conj, child["quaternion_wxyz"]))
                out[k, m.jnt_qposadr[j]] = 2 * np.arctan2(rel[1:] @ m.jnt_axis[j], rel[0])
            else:  # slide, in the parent frame
                rot = quat_to_mat(parent["quaternion_wxyz"])
                local = rot.T @ (np.subtract(child["position_m"], parent["position_m"]))
                axis = quat_to_mat(m.body_quat[bid]) @ m.jnt_axis[j]
                out[k, m.jnt_qposadr[j]] = (local - m.body_pos[bid]) @ axis
    return out


def _samples(spec: dict) -> dict[str, np.ndarray]:
    from simbench.reference import load_bundle

    m, d = _robot(spec["model"])
    bundle = load_bundle(spec["bundle"])
    frames = next(tr for tr in bundle["traces"] if tr["engine"] == spec["engine"])["frames"]
    knots = np.array([f["time_s"] for f in frames])
    knots[0] = 0.0  # the first sample (5 ms) is still at the initial rest pose
    q = recorded_joints(m, frames)
    t = control_times(spec)
    qref, qdref, qddref = clamped_spline(knots, q, t)
    fing = [m.jnt_qposadr[m.joint(j).id] for j in FINGER_JOINTS]
    qref[:, fing], qdref[:, fing], qddref[:, fing] = spec["finger_hold"], 0.0, 0.0
    return {"t": t, "qref": qref, "qdref": qdref, "qddref": qddref,
            "ff": _inverse(m, d, qref, qdref, qddref), "knots": knots, "recorded": q}


GENERATORS = {"bias": _bias, "joint": _joint, "cartesian": _cartesian, "samples": _samples}


def generate(spec: dict) -> Path:
    import mujoco

    arrays = GENERATORS[spec["kind"]](spec)
    path = tape_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, spec=json.dumps(spec), mujoco=mujoco.__version__)
    return path
