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


def free_dofs(m: mujoco.MjModel, body: int) -> int | None:
    """DOF address of a body's free joint, if it has one."""
    for j in range(m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]):
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(m.jnt_dofadr[j])
    return None


def options(m: mujoco.MjModel) -> dict:
    o = m.opt
    return {
        "timestep": float(o.timestep), "integrator": int(o.integrator),
        "solver": int(o.solver), "cone": int(o.cone), "jacobian": int(o.jacobian),
        "iterations": int(o.iterations), "tolerance": float(o.tolerance),
        "ls_iterations": int(o.ls_iterations), "impratio": float(o.impratio),
        "noslip_iterations": int(o.noslip_iterations),
        "enableflags": int(o.enableflags), "disableflags": int(o.disableflags),
        "gravity": o.gravity.tolist(),
    }


def apply_options(m: mujoco.MjModel, options: dict) -> None:
    """Override compiled mjOption fields by name, e.g. for Phase 4 sensitivity runs.

    `enable` / `disable` take mjtEnableBit / mjtDisableBit names; `cone`, `jacobian`,
    `integrator` take enum suffixes ("PYRAMIDAL", "SPARSE", "DISCRETE"); anything else
    is set as a field.
    """
    for key, value in options.items():
        if key == "enable":
            for name in value:
                m.opt.enableflags |= int(getattr(mujoco.mjtEnableBit, f"mjENBL_{name}"))
        elif key == "disable":
            for name in value:
                m.opt.disableflags |= int(getattr(mujoco.mjtDisableBit, f"mjDSBL_{name}"))
        elif key == "cone":
            m.opt.cone = getattr(mujoco.mjtCone, f"mjCONE_{value}")
        elif key == "jacobian":
            m.opt.jacobian = getattr(mujoco.mjtJacobian, f"mjJAC_{value}")
        elif key == "integrator":
            m.opt.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{value}")
        else:
            setattr(m.opt, key, value)


class Sim:
    """MuJoCo CPU simulation of one Scene (float64)."""

    def __init__(self, scene, seed: int = 0, options: dict | None = None):
        self.scene = scene
        self.m = m = mujoco.MjModel.from_xml_path(str(scene.mjcf_path()))
        apply_options(m, options or {})
        self.d = d = mujoco.MjData(m)
        self.body_ids = np.array([m.body(n).id for n in scene.bodies])
        self.qposadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in scene.joints], dtype=int)
        self.dofadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in scene.joints], dtype=int)
        # Flex sides of contacts (geom id -1) are reported as body nbody + flex id.
        self.contact_names = ([m.body(b).name or "world" for b in range(m.nbody)]
                              + [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, f)
                                 for f in range(m.nflex)])
        self.track = {b: i for i, b in enumerate(self.body_ids)}

        for name, q in scene.joint_init.items():
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
        mujoco.mj_kinematics(m, d)
        for name, init in scene.free_init.items():
            b = m.body(name).id
            dof = free_dofs(m, b)
            # Free-joint qvel: world-frame linear velocity, body-frame angular velocity.
            d.qvel[dof:dof + 3] = init.linear
            d.qvel[dof + 3:dof + 6] = d.xmat[b].reshape(3, 3).T @ np.asarray(init.angular)
        mujoco.mj_forward(m, d)
        self._vel = np.zeros(6)
        self._f6 = np.zeros(6)

    def settings(self) -> dict:
        return {"engine": "mujoco", "version": mujoco.__version__, "precision": "float64",
                "device": "cpu", **options(self.m)}

    def effective_model(self) -> dict:
        m = self.m
        return {n: {"mass": float(m.body_mass[b]), "inertia": m.body_inertia[b].tolist()}
                for n, b in zip(self.scene.bodies, self.body_ids, strict=True)}

    def state(self) -> dict:
        m, d, ids = self.m, self.d, self.body_ids
        # mj_step integrates qpos/qvel but leaves xpos/cvel at the pre-step state.
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        lin, ang = np.zeros((len(ids), 3)), np.zeros((len(ids), 3))
        for i, b in enumerate(ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._vel, 0)
            ang[i], lin[i] = self._vel[:3], self._vel[3:]
        return {"pos": d.xpos[ids].copy(), "com": d.xipos[ids].copy(),
                "quat": d.xquat[ids].copy(), "linvel": lin, "angvel": ang,
                "q": d.qpos[self.qposadr].copy(), "qd": d.qvel[self.dofadr].copy()}

    def step(self, tau: np.ndarray) -> None:
        self.d.qfrc_applied[:] = 0
        self.d.qfrc_applied[self.dofadr] = tau
        mujoco.mj_step(self.m, self.d)

    def contacts(self) -> dict:
        m, d = self.m, self.d
        n = d.ncon
        a, b = np.zeros(n, int), np.zeros(n, int)
        pos, force, dist = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
        net = np.zeros((len(self.body_ids), 3))
        for i in range(n):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, self._f6)
            # Contact frame rows are (normal, tangent1, tangent2); the normal points
            # from geom1 to geom2, so this is the force geom1 exerts on geom2.
            f = c.frame.reshape(3, 3).T @ self._f6[:3]
            a[i], b[i] = (m.geom_bodyid[g] if g >= 0 else m.nbody + c.flex[k]
                          for k, g in enumerate(c.geom))
            pos[i], force[i], dist[i] = c.pos, f, c.dist
            if b[i] in self.track:
                net[self.track[b[i]]] += f
            if a[i] in self.track:
                net[self.track[a[i]]] -= f
        return {"a": a, "b": b, "pos": pos, "force": force, "dist": dist, "net": net}


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
