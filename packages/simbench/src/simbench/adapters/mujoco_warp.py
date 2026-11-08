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


class Sim:
    """MuJoCo-Warp simulation of one Scene in a single world (float32, GPU).

    The initial state is prepared with MuJoCo CPU (same compiler, same conventions)
    and uploaded with put_data.
    """

    def __init__(self, scene, seed: int = 0, options: dict | None = None,
                 deterministic: bool = False, naconmax: int = 256, njmax: int = 1024):
        from simbench.adapters.mujoco_cpu import Sim as CpuSim

        cpu = CpuSim(scene, seed=seed, options=options)
        self.scene, self.mjm = scene, cpu.m
        self.m = mjw.put_model(cpu.m)
        if deterministic:
            self.m.opt.deterministic = int(mjw.DeterminismType.ALL)
        self.d = mjw.put_data(cpu.m, cpu.d, naconmax=naconmax, njmax=njmax)
        self.body_ids, self.qposadr, self.dofadr = cpu.body_ids, cpu.qposadr, cpu.dofadr
        self.contact_names, self.track = cpu.contact_names, cpu.track
        self.geom_body = cpu.m.geom_bodyid.copy()
        self.root = cpu.m.body_rootid[cpu.body_ids]
        self._qfrc = np.zeros((1, cpu.m.nv), dtype=np.float32)
        self._ids = wp.array(np.arange(naconmax, dtype=np.int32), dtype=wp.int32)
        self._force = wp.zeros(naconmax, dtype=wp.spatial_vector)

    def settings(self) -> dict:
        from simbench.adapters.mujoco_cpu import options

        return {"engine": "mujoco_warp", "version": mjw.__version__ if hasattr(mjw, "__version__")
                else None, "warp": wp.__version__, "precision": "float32",
                "device": str(wp.get_device()),
                "deterministic": int(getattr(self.m.opt, "deterministic", 0)),
                **options(self.mjm)}

    def effective_model(self) -> dict:
        mass, inertia = self.m.body_mass.numpy()[0], self.m.body_inertia.numpy()[0]
        return {n: {"mass": float(mass[b]), "inertia": inertia[b].astype(float).tolist()}
                for n, b in zip(self.scene.bodies, self.body_ids, strict=True)}

    def state(self) -> dict:
        m, d, ids = self.m, self.d, self.body_ids
        mjw.kinematics(m, d)
        mjw.com_pos(m, d)
        mjw.com_vel(m, d)
        cvel = d.cvel.numpy()[0][ids].astype(float)
        com = d.xipos.numpy()[0][ids].astype(float)
        sub = d.subtree_com.numpy()[0][self.root].astype(float)
        ang = cvel[:, :3]
        return {"pos": d.xpos.numpy()[0][ids].astype(float), "com": com,
                "quat": d.xquat.numpy()[0][ids].astype(float),
                "linvel": cvel[:, 3:] + np.cross(ang, com - sub), "angvel": ang,
                "q": d.qpos.numpy()[0][self.qposadr].astype(float),
                "qd": d.qvel.numpy()[0][self.dofadr].astype(float)}

    def step(self, tau: np.ndarray) -> None:
        self._qfrc[:] = 0
        self._qfrc[0, self.dofadr] = tau
        wp.copy(self.d.qfrc_applied, wp.array(self._qfrc, dtype=float))
        mjw.step(self.m, self.d)

    def contacts(self) -> dict:
        d = self.d
        n = int(d.nacon.numpy()[0])
        net = np.zeros((len(self.body_ids), 3))
        if n == 0:
            e = np.zeros(0, int)
            return {"a": e, "b": e, "pos": np.zeros((0, 3)), "force": np.zeros((0, 3)),
                    "dist": np.zeros(0), "net": net}
        mjw.contact_force(self.m, d, self._ids, True, self._force)
        geom = d.contact.geom.numpy()[:n]
        a, b = self.geom_body[geom[:, 0]], self.geom_body[geom[:, 1]]
        force = self._force.numpy()[:n, :3].astype(float)
        for i in range(n):
            if b[i] in self.track:
                net[self.track[b[i]]] += force[i]
            if a[i] in self.track:
                net[self.track[a[i]]] -= force[i]
        return {"a": a, "b": b, "pos": d.contact.pos.numpy()[:n].astype(float), "force": force,
                "dist": d.contact.dist.numpy()[:n].astype(float), "net": net}


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
