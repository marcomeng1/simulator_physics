"""Newton route: Newton model + SolverMuJoCo (MuJoCo-Warp backend, GPU, float32).

Two readbacks are produced: the Newton Model (what the importer built) and the
MjModel that SolverMuJoCo generates from it (what MuJoCo-Warp actually simulates).
"""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import newton
import numpy as np
import warp as wp

from simbench.adapters import mujoco_cpu
from simbench.readback import BodyReadback, JointReadback, ModelReadback

# newton.JointType values for the joints the benchmark uses.
JOINT_TYPES = {
    int(newton.JointType.PRISMATIC): "slide",
    int(newton.JointType.REVOLUTE): "hinge",
    int(newton.JointType.BALL): "ball",
    int(newton.JointType.FREE): "free",
}


def _mjcf_gravity(path: Path) -> tuple[float, float, float] | None:
    """add_mjcf does not read <option gravity>; scenes without it keep Newton's default."""
    option = ET.parse(path).getroot().find("option")
    g = option.get("gravity") if option is not None else None
    return tuple(float(v) for v in g.split()) if g else None


def build(path: Path, hydro: dict | None = None) -> newton.Model:
    """`hydro` ({"kh", "sdf_resolution"}) makes every colliding volumetric shape hydroelastic."""
    builder = newton.ModelBuilder(gravity=_mjcf_gravity(path))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(str(path))
    if hydro:
        flags = newton.ShapeFlags
        for i, f in enumerate(builder.shape_flags):
            if f & flags.COLLIDE_SHAPES and builder.shape_type[i] not in (
                    newton.GeoType.PLANE, newton.GeoType.HFIELD):
                builder.shape_flags[i] = f | flags.HYDROELASTIC
                builder.shape_material_kh[i] = hydro["kh"]
                builder.shape_sdf_max_resolution[i] = hydro["sdf_resolution"]
    return builder.finalize()


def _short(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _quat_xyzw_to_wxyz(q: np.ndarray) -> list[float]:
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def readback(path: Path, route: str, model: str,
             configs: list[dict[str, float]]) -> list[ModelReadback]:
    m = build(path)
    rb = ModelReadback(route=route, model=model)

    mass, com, inertia = m.body_mass.numpy(), m.body_com.numpy(), m.body_inertia.numpy()
    for b, label in enumerate(m.body_label):
        rb.bodies[_short(label)] = BodyReadback(
            mass=float(mass[b]), com=com[b].astype(float).tolist(),
            inertia=inertia[b].astype(float).tolist(),
        )

    jtype, q_start = m.joint_type.numpy(), m.joint_q_start.numpy()
    qd_start, x_c = m.joint_qd_start.numpy(), m.joint_X_c.numpy()
    axis = m.joint_axis.numpy()
    lower, upper = m.joint_limit_lower.numpy(), m.joint_limit_upper.numpy()
    armature, damping = m.joint_armature.numpy(), m.joint_damping.numpy()
    friction = m.joint_friction.numpy()
    joint_dof: dict[str, int] = {}
    for j, label in enumerate(m.joint_label):
        if int(jtype[j]) not in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)):
            continue
        dof = int(qd_start[j])
        name = _short(label)
        joint_dof[name] = int(q_start[j])
        rb.joints[name] = JointReadback(
            type=JOINT_TYPES[int(jtype[j])],
            # Newton joint axes live in the child joint frame; express them in the body frame.
            axis=_rotate_xyzw(x_c[j][3:].astype(float), axis[dof].astype(float)).tolist(),
            range=[float(lower[dof]), float(upper[dof])],
            armature=float(armature[dof]),
            damping=float(damping[dof]),
            frictionloss=float(friction[dof]),
        )

    state = m.state()
    for cfg in configs:
        q = m.joint_q.numpy().copy()
        for name, value in cfg.items():
            q[joint_dof[name]] = value
        newton.eval_fk(m, wp.array(q, dtype=float), m.joint_qd, state)
        body_q = state.body_q.numpy()
        rb.fk_qpos.append(cfg)
        rb.fk_poses.append({
            _short(label): [*body_q[b][:3].astype(float).tolist(),
                            *_quat_xyzw_to_wxyz(body_q[b][3:].astype(float))]
            for b, label in enumerate(m.body_label)
        })

    return [rb, _generated_mjmodel_readback(m, route, model, configs)]


def _generated_mjmodel_readback(m: newton.Model, route: str, model: str,
                                configs: list[dict[str, float]]) -> ModelReadback:
    """Readback of the MjModel SolverMuJoCo generates, renamed to MJCF names.

    The generated model names elements by their Newton label with '/' replaced by '_'.
    """
    import mujoco

    mjm = newton.solvers.SolverMuJoCo(m).mj_model
    short = {label.replace("/", "_"): _short(label) for label in [*m.body_label, *m.joint_label]}
    rb = mujoco_cpu.static_readback(mjm, f"{route}-mjmodel", model)
    rb.bodies = {short.get(k, k): v for k, v in rb.bodies.items()}
    rb.joints = {short.get(k, k): v for k, v in rb.joints.items()}
    rb.notes.append("MjModel generated by newton.solvers.SolverMuJoCo")

    qposadr = {short.get(mjm.joint(j).name, mjm.joint(j).name): int(mjm.jnt_qposadr[j])
               for j in range(mjm.njnt)}
    d = mujoco.MjData(mjm)
    for cfg in configs:
        mujoco.mj_resetData(mjm, d)
        for name, value in cfg.items():
            d.qpos[qposadr[name]] = value
        mujoco.mj_kinematics(mjm, d)
        rb.fk_qpos.append(cfg)
        rb.fk_poses.append({
            short.get(mjm.body(b).name, mjm.body(b).name): [*d.xpos[b].tolist(),
                                                            *d.xquat[b].tolist()]
            for b in range(1, mjm.nbody)
        })
    return rb


class Sim:
    """Newton model stepped by SolverMuJoCo with native MuJoCo-Warp contacts (float32, GPU).

    Body state comes from Newton's State; contact forces come from the solver's
    MuJoCo-Warp data, mapped back to MJCF body names.
    """

    def __init__(self, scene, seed: int = 0, naconmax: int = 256, njmax: int = 1024,
                 contacts: str = "mujoco", kh: float = 1e11, sdf_resolution: int = 64,
                 reduce: bool = True, mujoco_invweight: bool = False, **solver_options):
        """`contacts`: "mujoco" (MuJoCo-Warp collision), "newton" (Newton's collision
        pipeline, point contacts) or "hydroelastic" (Newton pipeline, SDF pressure patches).

        `mujoco_invweight` replaces the body_invweight0 SolverMuJoCo generates with the
        values MuJoCo compiles from the same MJCF. They set the constraint regularizer, and
        Newton 1.7 gives a hinged body a translational weight where MuJoCo gives 0.
        """
        import mujoco_warp as mjw

        self.mjw = mjw
        self.scene = scene
        self.contact_mode = contacts
        path = scene.mjcf_path()
        hydro = {"kh": kh, "sdf_resolution": sdf_resolution} if contacts == "hydroelastic" else None
        self.model = m = build(path, hydro)
        integrator = scene.params.get("integrator", "euler").lower()
        opts = {"solver": "newton", "integrator": integrator, "cone": "elliptic",
                "iterations": 100, "tolerance": 1e-10, "nconmax": naconmax, "njmax": njmax,
                "use_mujoco_contacts": contacts == "mujoco"}
        opts.update(solver_options)
        if isinstance(opts.get("deterministic"), str):  # plans are JSON: "RUN_TO_RUN" etc.
            opts["deterministic"] = wp.DeterministicMode[opts["deterministic"]]
        self.solver = newton.solvers.SolverMuJoCo(m, **opts)
        self.pipeline = self.newton_contacts = None
        if contacts != "mujoco":
            config = (newton.geometry.HydroelasticSDF.Config(reduce_contacts=reduce)
                      if hydro else None)
            self.pipeline = newton.CollisionPipeline(
                m, reduce_contacts=reduce, rigid_contact_max=naconmax * 8,
                sdf_hydroelastic_config=config)
            self.newton_contacts = self.pipeline.contacts()
        self.dt = scene.dt

        body_index = {_short(label): b for b, label in enumerate(m.body_label)}
        self.body_ids = np.array([body_index[n] for n in scene.bodies])
        jtype, q_start, qd_start = (m.joint_type.numpy(), m.joint_q_start.numpy(),
                                    m.joint_qd_start.numpy())
        joint_index = {_short(label): j for j, label in enumerate(m.joint_label)}
        self.q_idx = np.array([q_start[joint_index[n]] for n in scene.joints], dtype=int)
        self.qd_idx = np.array([qd_start[joint_index[n]] for n in scene.joints], dtype=int)
        self.body_com = m.body_com.numpy().astype(float)

        q, qd = m.joint_q.numpy().copy(), m.joint_qd.numpy().copy()
        for name, value in scene.joint_init.items():
            q[q_start[joint_index[name]]] = value
        free_joint_of = {int(m.joint_child.numpy()[j]): j for j in range(m.joint_count)
                         if int(jtype[j]) == int(newton.JointType.FREE)}
        for name, init in scene.free_init.items():
            # Free-joint qd: COM linear velocity and angular velocity, both world frame.
            start = qd_start[free_joint_of[body_index[name]]]
            qd[start:start + 3] = init.linear
            qd[start + 3:start + 6] = init.angular
        self.s0, self.s1, self.control = m.state(), m.state(), m.control()
        self.s0.joint_q.assign(q)
        self.s0.joint_qd.assign(qd)
        newton.eval_fk(m, self.s0.joint_q, self.s0.joint_qd, self.s0)
        self._joint_f = np.zeros(m.joint_dof_count, dtype=np.float32)

        # Contact geoms -> MJCF body names via the generated MjModel's naming.
        mjm = self.solver.mj_model
        short = {label.replace("/", "_"): _short(label) for label in m.body_label}
        self.contact_names = [short.get(mjm.body(b).name, mjm.body(b).name) or "world"
                              for b in range(mjm.nbody)]
        self.contact_names[0] = "world"
        self.mujoco_invweight = mujoco_invweight
        if mujoco_invweight:
            import mujoco

            ref = mujoco.MjModel.from_xml_path(str(path))
            weights = self.solver.mjw_model.body_invweight0.numpy()
            for b in range(1, mjm.nbody):
                weights[:, b] = ref.body(self.contact_names[b]).invweight0
            self.solver.mjw_model.body_invweight0.assign(weights)
        self.geom_body = mjm.geom_bodyid.copy()
        self.track = {self.contact_names.index(n): i for i, n in enumerate(scene.bodies)}
        self._ids = wp.array(np.arange(self.solver.mjw_data.naconmax, dtype=np.int32),
                             dtype=wp.int32)
        self._force = wp.zeros(self.solver.mjw_data.naconmax, dtype=wp.spatial_vector)

    def settings(self) -> dict:
        from simbench.adapters.mujoco_cpu import options

        return {"engine": "newton", "version": newton.__version__, "warp": wp.__version__,
                "mujoco_warp": getattr(self.mjw, "__version__", None), "precision": "float32",
                "device": str(wp.get_device()),
                "deterministic": int(getattr(self.solver.mjw_model.opt, "deterministic", 0)),
                "contacts": self.contact_mode, "mujoco_invweight": self.mujoco_invweight,
                **options(self.solver.mj_model)}

    def effective_model(self) -> dict:
        mass, inertia = self.model.body_mass.numpy(), self.model.body_inertia.numpy()
        return {n: {"mass": float(mass[b]), "inertia": inertia[b].astype(float).tolist()}
                for n, b in zip(self.scene.bodies, self.body_ids, strict=True)}

    def state(self) -> dict:
        ids = self.body_ids
        body_q = self.s0.body_q.numpy()[ids].astype(float)
        body_qd = self.s0.body_qd.numpy()[ids].astype(float)
        pos, quat_xyzw = body_q[:, :3], body_q[:, 3:]
        com = pos + np.array([_rotate_xyzw(q, c) for q, c in
                              zip(quat_xyzw, self.body_com[ids], strict=True)])
        q, qd = self.s0.joint_q.numpy(), self.s0.joint_qd.numpy()
        return {"pos": pos, "com": com,
                "quat": np.array([_quat_xyzw_to_wxyz(x) for x in quat_xyzw]),
                "linvel": body_qd[:, :3], "angvel": body_qd[:, 3:],
                "q": q[self.q_idx].astype(float), "qd": qd[self.qd_idx].astype(float)}

    def step(self, tau: np.ndarray) -> None:
        self._joint_f[:] = 0
        self._joint_f[self.qd_idx] = tau
        self.control.joint_f.assign(self._joint_f)
        if self.pipeline is not None:
            self.pipeline.collide(self.s0, self.newton_contacts)
        self.solver.step(self.s0, self.s1, self.control, self.newton_contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0

    def contacts(self) -> dict:
        d = self.solver.mjw_data
        n = int(d.nacon.numpy()[0])
        net = np.zeros((len(self.body_ids), 3))
        if n == 0:
            e = np.zeros(0, int)
            return {"a": e, "b": e, "pos": np.zeros((0, 3)), "force": np.zeros((0, 3)),
                    "dist": np.zeros(0), "net": net}
        self.mjw.contact_force(self.solver.mjw_model, d, self._ids, True, self._force)
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
    m = build(path)
    solver = newton.solvers.SolverMuJoCo(m)
    s0, s1, control = m.state(), m.state(), m.control()
    dt = float(solver.mj_model.opt.timestep)
    solver.step(s0, s1, control, None, dt)
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
    wp.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "device": str(wp.get_device()),
        "precision": "float32",
        "nq": int(m.joint_coord_count), "nv": int(m.joint_dof_count), "nbody": int(m.body_count),
        "steps": steps,
        "steps_per_s": steps / elapsed,
        "finite": bool(np.all(np.isfinite(s0.joint_q.numpy()))),
    }
