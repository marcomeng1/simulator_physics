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
