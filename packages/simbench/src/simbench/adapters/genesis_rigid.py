"""Genesis route: native rigid-body solver on GPU, float64.

Readback queries the solver's runtime fields (what Genesis simulates), not the
MJCF description. Genesis allows one gs.init per process; each CLI call is one process.
"""

import time
from pathlib import Path

import genesis as gs
import numpy as np

from simbench.readback import BodyReadback, JointReadback, ModelReadback, quat_to_mat

_INITIALIZED = False
JOINT_TYPES = {
    gs.JOINT_TYPE.REVOLUTE: "hinge",
    gs.JOINT_TYPE.PRISMATIC: "slide",
    gs.JOINT_TYPE.SPHERICAL: "ball",
    gs.JOINT_TYPE.FREE: "free",
}


def _init() -> None:
    global _INITIALIZED
    if not _INITIALIZED:
        gs.init(backend=gs.gpu, precision="64", logging_level="warning", seed=0)
        _INITIALIZED = True


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy().astype(float)


def _scalar_joints(entity) -> list:
    return [j for j in entity.joints
            if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]


def apply_mjcf_dof_properties(path: Path, entity) -> list[str]:
    """Set armature, damping and frictionloss to the values MuJoCo compiles from the MJCF.

    Genesis substitutes a 0.1 kg m^2 armature where the MJCF specifies zero. Returns
    one note per property whose imported value differed, recording the import default.
    """
    from simbench.adapters.mujoco_cpu import load_model

    m = load_model(path)
    joints = _scalar_joints(entity)
    dofs = [j.dofs_idx_local[0] for j in joints]
    mj_dof = [int(m.jnt_dofadr[m.joint(j.name).id]) for j in joints]
    notes = []
    for prop, mj_values in (("armature", m.dof_armature), ("damping", m.dof_damping),
                            ("frictionloss", m.dof_frictionloss)):
        imported = _np(getattr(entity, f"get_dofs_{prop}")(dofs))
        target = np.array([mj_values[d] for d in mj_dof])
        if not np.allclose(imported, target, rtol=0, atol=1e-12):
            notes.append(f"genesis import set {prop}={imported.tolist()}; "
                         f"overridden to MJCF {target.tolist()}")
        getattr(entity, f"set_dofs_{prop}")(target, dofs)
    return notes


def build(path: Path, dt: float = 0.002, n_envs: int = 0, mjcf_dof_properties: bool = True):
    """Genesis's rigid-solver default timestep is 10 ms, so dt is always set explicitly."""
    _init()
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt), show_viewer=False)
    entity = scene.add_entity(gs.morphs.MJCF(file=str(path)))
    scene.build(n_envs=n_envs)
    notes = apply_mjcf_dof_properties(path, entity) if mjcf_dof_properties else []
    return scene, entity, notes


def readback(path: Path, route: str, model: str,
             configs: list[dict[str, float]]) -> ModelReadback:
    scene, entity, notes = build(path)
    solver = entity.solver
    rb = ModelReadback(route=route, model=model, notes=notes)

    links_idx = [link.idx for link in entity.links]
    mass = _np(solver.get_links_mass(links_idx))
    com = _np(solver.get_links_COM(links_idx))
    inertia = _np(solver.get_links_inertia(links_idx))
    for i, link in enumerate(entity.links):
        rot = quat_to_mat(link.desc.inertial_quat)
        rb.bodies[link.name] = BodyReadback(
            mass=float(mass[i]), com=com[i].tolist(),
            inertia=(rot @ inertia[i] @ rot.T).tolist(),
        )

    joints = _scalar_joints(entity)
    dofs = [j.dofs_idx_local[0] for j in joints]
    lower, upper = (_np(t) for t in entity.get_dofs_limit(dofs))
    armature = _np(entity.get_dofs_armature(dofs))
    damping = _np(entity.get_dofs_damping(dofs))
    frictionloss = _np(entity.get_dofs_frictionloss(dofs))
    for k, j in enumerate(joints):
        link_rot = quat_to_mat(_np(j.link.get_quat()))
        axis_world = _np(j.get_anchor_axis())
        rb.joints[j.name] = JointReadback(
            type=JOINT_TYPES[j.type],
            axis=(link_rot.T @ axis_world).tolist(),
            range=[float(lower[k]), float(upper[k])],
            armature=float(armature[k]),
            damping=float(damping[k]),
            frictionloss=float(frictionloss[k]),
        )

    qs_of = {j.name: j.qs_idx_local[0] for j in joints}
    q0 = _np(entity.get_qpos())
    for cfg in configs:
        q = q0.copy()
        for name, value in cfg.items():
            q[qs_of[name]] = value
        entity.set_qpos(q)
        rb.fk_qpos.append(cfg)
        rb.fk_poses.append({
            link.name: [*_np(link.get_pos()).tolist(), *_np(link.get_quat()).tolist()]
            for link in entity.links
        })
    return rb


def rigid_options(mjcf_path: Path, dt: float) -> dict:
    """Solver settings matched to the scene's MJCF <option> (Genesis does not read them)."""
    import mujoco

    from simbench.adapters.mujoco_cpu import load_model

    o = load_model(mjcf_path).opt
    integrators = {int(mujoco.mjtIntegrator.mjINT_EULER): gs.integrator.Euler,
                   int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): gs.integrator.implicitfast}
    cones = {int(mujoco.mjtCone.mjCONE_ELLIPTIC): gs.friction_cone.elliptic,
             int(mujoco.mjtCone.mjCONE_PYRAMIDAL): gs.friction_cone.pyramidal}
    return {"dt": dt, "integrator": integrators[int(o.integrator)],
            "constraint_solver": gs.constraint_solver.Newton,
            "iterations": int(o.iterations), "tolerance": float(o.tolerance),
            "friction_cone": cones[int(o.cone)], "gravity": tuple(o.gravity)}


class Sim:
    """Genesis rigid solver, single environment (float64, GPU)."""

    def __init__(self, scene, seed: int = 0, matched_options: bool = True):
        """matched_options=False keeps Genesis's own RigidOptions defaults (only dt is set)."""
        import torch

        self.torch = torch
        _init()
        self.scene = scene
        path = scene.mjcf_path()
        opts = rigid_options(path, scene.dt) if matched_options else {"dt": scene.dt}
        self.rigid_opts = {k: str(v) for k, v in opts.items()}
        gravity = opts.pop("gravity", (0.0, 0.0, -9.81))
        dt = opts.pop("dt")
        self.gs_scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
            rigid_options=gs.options.RigidOptions(**opts),
            show_viewer=False,
        )
        self.entity = e = self.gs_scene.add_entity(gs.morphs.MJCF(file=str(path)))
        self.gs_scene.build()
        self.notes = apply_mjcf_dof_properties(path, e)

        links = {link.name: link for link in e.links}
        self.links = [links[n] for n in scene.bodies]
        joints = {j.name: j for j in e.joints}
        self.dofs = [joints[n].dofs_idx_local[0] for n in scene.joints]
        self.qs = [joints[n].qs_idx_local[0] for n in scene.joints]
        solver = e.solver
        self.link_global = [link.idx for link in self.links]
        self.contact_names = [link.name for link in solver.links]
        self.track = {g: i for i, g in enumerate(self.link_global)}

        if scene.joint_init:
            q = _np(e.get_qpos())
            for name, value in scene.joint_init.items():
                q[joints[name].qs_idx_local[0]] = value
            e.set_qpos(q)
        if scene.free_init:
            vel = _np(e.get_dofs_velocity())
            for name, init in scene.free_init.items():
                link = links[name]
                joint = next(j for j in link.joints if j.type == gs.JOINT_TYPE.FREE)
                start = joint.dofs_idx_local[0]
                rot = quat_to_mat(_np(link.get_quat()))
                # Free-joint dofs: world linear velocity, body-frame angular velocity.
                vel[start:start + 3] = init.linear
                vel[start + 3:start + 6] = rot.T @ np.asarray(init.angular)
            e.set_dofs_velocity(vel)
        self._tau = torch.zeros(e.n_dofs, dtype=gs.tc_float, device=gs.device)

    def settings(self) -> dict:
        return {"engine": "genesis", "version": gs.__version__, "precision": "float64",
                "device": str(gs.device), **self.rigid_opts}

    def effective_model(self) -> dict:
        solver = self.entity.solver
        mass = _np(solver.get_links_mass(self.link_global))
        return {n: {"mass": float(mass[i])} for i, n in enumerate(self.scene.bodies)}

    def state(self) -> dict:
        solver = self.entity.solver
        idx = self.link_global
        pos = _np(solver.get_links_pos(idx))
        quat = _np(solver.get_links_quat(idx))
        com = pos + np.array([quat_to_mat(q) @ c for q, c in
                              zip(quat, _np(solver.get_links_COM(idx)), strict=True)])
        lin = _np(solver.get_links_vel(idx, ref=gs.link_ref_frame.link_COM))
        ang = _np(solver.get_links_ang(idx))
        q, qd = _np(self.entity.get_qpos()), _np(self.entity.get_dofs_velocity())
        return {"pos": pos, "com": com, "quat": quat, "linvel": lin, "angvel": ang,
                "q": q[self.qs] if self.qs else np.zeros(0),
                "qd": qd[self.dofs] if self.dofs else np.zeros(0)}

    def step(self, tau: np.ndarray) -> None:
        if self.dofs:
            self._tau.zero_()
            self._tau[self.dofs] = self.torch.as_tensor(tau, dtype=gs.tc_float, device=gs.device)
            self.entity.control_dofs_force(self._tau)
        self.gs_scene.step()

    def contacts(self) -> dict:
        c = self.entity.get_contacts()
        a = c["link_a"].cpu().numpy().astype(int)
        b = c["link_b"].cpu().numpy().astype(int)
        pos = _np(c["position"]).reshape(-1, 3)
        force = _np(c["force_b"]).reshape(-1, 3)
        net = np.zeros((len(self.links), 3))
        for i in range(len(a)):
            if b[i] in self.track:
                net[self.track[b[i]]] += force[i]
            if a[i] in self.track:
                net[self.track[a[i]]] += _np(c["force_a"][i])
        dist = _np(c["penetration"]).reshape(-1) * -1 if "penetration" in c else np.zeros(len(a))
        return {"a": a, "b": b, "pos": pos, "force": force, "dist": dist, "net": net}


def smoke(path: Path, steps: int = 1000) -> dict:
    import torch

    scene, entity, _ = build(path)
    scene.step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        scene.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "device": str(gs.device),
        "precision": "float64",
        "nq": int(entity.n_qs), "nv": int(entity.n_dofs), "nbody": int(entity.n_links),
        "steps": steps,
        "steps_per_s": steps / elapsed,
        "finite": bool(np.all(np.isfinite(_np(entity.get_qpos())))),
    }
