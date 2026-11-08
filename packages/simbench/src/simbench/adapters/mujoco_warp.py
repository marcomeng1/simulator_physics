"""Standalone MuJoCo-Warp route (GPU, float32).

Readback comes from the device-side Model arrays, not the host MjModel, so it
checks what the GPU actually simulates.
"""

import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from simbench.adapters.mujoco_cpu import JOINT_TYPES
from simbench.readback import BodyReadback, JointReadback, ModelReadback, inertia_tensor


def readback(path: Path, route: str, model: str,
             configs: list[dict[str, float]]) -> ModelReadback:
    mjm = mujoco.MjModel.from_xml_path(str(path))
    m = mjw.put_model(mjm)
    rb = ModelReadback(route=route, model=model)

    mass, ipos = m.body_mass.numpy()[0], m.body_ipos.numpy()[0]
    inertia, iquat = m.body_inertia.numpy()[0], m.body_iquat.numpy()[0]
    for b in range(1, mjm.nbody):
        rb.bodies[mjm.body(b).name] = BodyReadback(
            mass=float(mass[b]),
            com=ipos[b].astype(float).tolist(),
            inertia=inertia_tensor(inertia[b].astype(float),
                                   iquat[b].astype(float)).tolist(),
        )

    axis, rng = m.jnt_axis.numpy()[0], m.jnt_range.numpy()[0]
    armature, damping = m.dof_armature.numpy()[0], m.dof_damping.numpy()[0]
    frictionloss = m.dof_frictionloss.numpy()[0]
    for j in range(mjm.njnt):
        dof = mjm.jnt_dofadr[j]
        rb.joints[mjm.joint(j).name] = JointReadback(
            type=JOINT_TYPES[int(mjm.jnt_type[j])],
            axis=axis[j].astype(float).tolist(),
            range=rng[j].astype(float).tolist() if mjm.jnt_limited[j] else [0.0, 0.0],
            armature=float(armature[dof]),
            damping=float(damping[dof]),
            frictionloss=float(frictionloss[dof]),
        )

    d = mjw.make_data(mjm, nworld=len(configs))
    qpos = np.tile(mjm.qpos0, (len(configs), 1))
    for i, cfg in enumerate(configs):
        for name, q in cfg.items():
            qpos[i, mjm.jnt_qposadr[mjm.joint(name).id]] = q
    wp.copy(d.qpos, wp.array(qpos.astype(np.float32), dtype=d.qpos.dtype))
    mjw.kinematics(m, d)
    xpos, xquat = d.xpos.numpy(), d.xquat.numpy()
    for i, cfg in enumerate(configs):
        rb.fk_qpos.append(cfg)
        rb.fk_poses.append({
            mjm.body(b).name: [*xpos[i, b].astype(float).tolist(),
                               *xquat[i, b].astype(float).tolist()]
            for b in range(1, mjm.nbody)
        })
    return rb


def smoke(path: Path, steps: int = 1000) -> dict:
    mjm = mujoco.MjModel.from_xml_path(str(path))
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm)
    mjw.step(m, d)
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        mjw.step(m, d)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "device": str(wp.get_device()),
        "precision": "float32",
        "nq": int(mjm.nq), "nv": int(mjm.nv), "nbody": int(mjm.nbody),
        "steps": steps,
        "steps_per_s": steps / elapsed,
        "finite": bool(np.all(np.isfinite(d.qpos.numpy()))),
    }
