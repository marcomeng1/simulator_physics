"""MuJoCo CPU route (float64). Serves both the 3.11 and 4.0 environments."""

import time
from pathlib import Path

import mujoco
import numpy as np

from simbench.readback import BodyReadback, JointReadback, ModelReadback, inertia_tensor

# Keyed by int: in MuJoCo 4.0 the mjtJoint enum no longer hashes like its value.
JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}
SCALAR_JOINTS = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def load_model(path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


def joint_ranges(m: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    return {
        m.joint(j).name: tuple(float(v) for v in m.jnt_range[j])
        for j in range(m.njnt)
        if int(m.jnt_type[j]) in SCALAR_JOINTS
    }


def static_readback(m: mujoco.MjModel, route: str, model: str) -> ModelReadback:
    rb = ModelReadback(route=route, model=model)
    for b in range(1, m.nbody):
        rb.bodies[m.body(b).name] = BodyReadback(
            mass=float(m.body_mass[b]),
            com=m.body_ipos[b].tolist(),
            inertia=inertia_tensor(m.body_inertia[b], m.body_iquat[b]).tolist(),
        )
    for j in range(m.njnt):
        dof = m.jnt_dofadr[j]
        rb.joints[m.joint(j).name] = JointReadback(
            type=JOINT_TYPES[int(m.jnt_type[j])],
            axis=m.jnt_axis[j].tolist(),
            range=m.jnt_range[j].tolist() if m.jnt_limited[j] else [0.0, 0.0],
            armature=float(m.dof_armature[dof]),
            damping=float(m.dof_damping[dof]),
            frictionloss=float(m.dof_frictionloss[dof]),
        )
    return rb


def readback(path: Path, route: str, model: str,
             configs: list[dict[str, float]]) -> ModelReadback:
    m = load_model(path)
    d = mujoco.MjData(m)
    rb = static_readback(m, route, model)
    for cfg in configs:
        mujoco.mj_resetData(m, d)
        for name, q in cfg.items():
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
        mujoco.mj_kinematics(m, d)
        rb.fk_qpos.append(cfg)
        rb.fk_poses.append({
            m.body(b).name: [*d.xpos[b].tolist(), *d.xquat[b].tolist()]
            for b in range(1, m.nbody)
        })
    return rb


def smoke(path: Path, steps: int = 1000) -> dict:
    m = load_model(path)
    d = mujoco.MjData(m)
    start = time.perf_counter()
    mujoco.mj_step(m, d, nstep=steps)
    elapsed = time.perf_counter() - start
    return {
        "device": "cpu",
        "precision": "float64",
        "nq": int(m.nq), "nv": int(m.nv), "nbody": int(m.nbody),
        "steps": steps,
        "steps_per_s": steps / elapsed,
        "finite": bool(np.all(np.isfinite(d.qpos))),
    }
