"""Newton VBD cloth + kinematically driven dual-Franka for cloth-fold.

Newton's ``cloth_franka`` demo drives the robot by writing ``joint_q`` / ``joint_qd``
(Featherstone as FK), not free joint torques. We do the same from the scene tape, then
step ``SolverVBD`` with a collision pipeline so pads can grasp the cloth.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import newton
import numpy as np
import warp as wp

from simbench.adapters.newton_mjwarp import _quat_xyzw_to_wxyz, _rotate_xyzw, _short
from simbench.scenes.panda import CTRL_DT


def _rigid_mjcf(scene) -> Path:
    """Write MJCF without deformable flex / cloth_* bodies for Newton's MJCF importer."""
    root = ET.fromstring(scene.mjcf)
    for el in list(root.findall("deformable")):
        root.remove(el)
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        if (body.get("name") or "").startswith("cloth_"):
            world.remove(body)
    opt = root.find("option")
    if opt is not None and opt.get("integrator", "").lower() == "discrete":
        opt.set("integrator", "Euler")
    path = scene.mjcf_path().with_name(scene.mjcf_path().stem + "_rigid.xml")
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode"))
    return path


class Sim:
    """Kinematic dual-Franka + VBD cloth with soft–rigid collisions."""

    def __init__(self, scene, seed: int = 0, cloth_vbd: bool = True, **_):
        self.scene = scene
        p = scene.params
        cloth = p["cloth"]
        path = _rigid_mjcf(scene)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_mjcf(str(path))
        nx, ny = int(cloth["nx"]), int(cloth["ny"])
        size_x = float(cloth.get("size_x", cloth["size_m"]))
        size_y = float(cloth.get("size_y", cloth["size_m"]))
        cell_x = size_x / max(nx - 1, 1)
        cell_y = size_y / max(ny - 1, 1)
        rest = np.asarray(p["cloth_rest"], float)
        origin = rest[0]
        radius = float(p.get("cloth_radius", cloth["thickness_m"]))
        builder.add_cloth_grid(
            pos=wp.vec3(*origin.tolist()),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=nx - 1,
            dim_y=ny - 1,
            cell_x=cell_x,
            cell_y=cell_y,
            mass=float(cloth["areal_density"]) * cell_x * cell_y,
            tri_ke=float(cloth["young"]) * float(cloth["thickness_m"]),
            tri_ka=float(cloth["young"]) * float(cloth["thickness_m"]) * 0.5,
            tri_kd=float(cloth["damping"]),
            particle_radius=radius,
        )
        builder.color()
        self.model = m = builder.finalize()
        self.dt = scene.dt
        self.cloth = newton.solvers.SolverVBD(
            m, iterations=10, integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False)
        self.pipeline = newton.CollisionPipeline(
            m, soft_contact_margin=max(2.0 * radius, 0.01),
            rigid_contact_max=64 * 8)
        # Named _contact_buf so it does not shadow contacts().
        self._contact_buf = self.pipeline.contacts()
        self.s0, self.s1, self.control = m.state(), m.state(), m.control()

        body_index = {_short(label): b for b, label in enumerate(m.body_label)}
        self.body_index = body_index
        self.cloth_count = int(m.particle_count)
        self.cloth_offset = {f"cloth_{i}": i for i in range(self.cloth_count)}
        q_start = m.joint_q_start.numpy()
        joint_index = {_short(label): j for j, label in enumerate(m.joint_label)}
        self.q_idx = np.array([q_start[joint_index[n]] for n in scene.joints], dtype=int)
        self.body_com = m.body_com.numpy().astype(float)
        q = m.joint_q.numpy().copy()
        for name, value in scene.joint_init.items():
            if name in joint_index:
                q[q_start[joint_index[name]]] = value
        self.s0.joint_q.assign(q)
        self.s0.joint_qd.zero_()
        newton.eval_fk(m, self.s0.joint_q, self.s0.joint_qd, self.s0)
        self.contact_names = list(scene.bodies)
        self.notes = ["newton cloth: kinematic joint_q from tape + SolverVBD + pinch attach"]
        self._step = 0
        self._local = None
        p = scene.params
        self.pinch = {"L_": list(p.get("pinch_left") or []),
                      "R_": list(p.get("pinch_right") or [])}
        hands = p.get("hand_bodies") or {"L_": "L_hand", "R_": "R_hand"}
        self.hand_bodies = {k: hands.get(k, f"{k}hand") for k in ("L_", "R_")}
        self.attach_window = p.get("attach_window") or [0.65, 2.60]
        self.rest = np.asarray(p["cloth_rest"], float)

    def settings(self) -> dict:
        return {"engine": "newton-vbd", "version": newton.__version__, "warp": wp.__version__,
                "precision": "float32", "device": str(wp.get_device()),
                "cloth_model": "SolverVBD", "cloth_solver": "newton.solvers.SolverVBD",
                "robot_drive": "kinematic-tape", "particles": self.cloth_count}

    def effective_model(self) -> dict:
        return {"cloth_particles": self.cloth_count}

    def state(self) -> dict:
        n = len(self.scene.bodies)
        pos = np.zeros((n, 3))
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        com = np.zeros((n, 3))
        lin = np.zeros((n, 3))
        ang = np.zeros((n, 3))
        body_q = self.s0.body_q.numpy().astype(float)
        body_qd = self.s0.body_qd.numpy().astype(float)
        particles = self.s0.particle_q.numpy().astype(float)
        pvel = self.s0.particle_qd.numpy().astype(float)
        for i, name in enumerate(self.scene.bodies):
            if name in self.cloth_offset:
                pi = self.cloth_offset[name]
                pos[i] = particles[pi]
                com[i] = particles[pi]
                lin[i] = pvel[pi]
            elif name in self.body_index:
                b = self.body_index[name]
                pos[i] = body_q[b, :3]
                quat[i] = _quat_xyzw_to_wxyz(body_q[b, 3:])
                com[i] = pos[i] + _rotate_xyzw(body_q[b, 3:], self.body_com[b])
                lin[i], ang[i] = body_qd[b, :3], body_qd[b, 3:]
        q = self.s0.joint_q.numpy()
        qd = self.s0.joint_qd.numpy()
        return {"pos": pos, "com": com, "quat": quat, "linvel": lin, "angvel": ang,
                "q": q[self.q_idx].astype(float),
                "qd": np.zeros(len(self.q_idx))}

    def step(self, tau: np.ndarray) -> None:
        # Kinematic robot from the open-loop tape (Featherstone torque control does not
        # advance our dual-arm MJCF reliably with VBD coupling).
        ctrl = self.scene.controller
        t = self._step * self.dt
        self._step += 1
        self._drive_robot(t)
        self._attach_pinch(t)
        self.s0.clear_forces()
        self.pipeline.collide(self.s0, self._contact_buf)
        self.cloth.step(self.s0, self.s1, self.control, self._contact_buf, self.dt)
        self.s0, self.s1 = self.s1, self.s0
        # VBD writes the next state into what is now s0; re-apply kinematic joints so
        # the arm does not snap back to a stale joint_q carried in the double buffer.
        self._drive_robot(t)
        self._attach_pinch(t)

    def _drive_robot(self, t: float) -> None:
        ctrl = self.scene.controller
        q = self.s0.joint_q.numpy().copy()
        if hasattr(ctrl, "joint_targets"):
            q[self.q_idx] = np.asarray(ctrl.joint_targets(t), np.float32)
        elif hasattr(ctrl, "qref"):
            k = int(round(t / CTRL_DT))
            k = min(max(k, 0), len(ctrl.qref) - 1)
            q[self.q_idx] = ctrl.qref[k].astype(np.float32)
        self.s0.joint_q.assign(q)
        self.s0.joint_qd.zero_()
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)

    def _attach_pinch(self, t: float) -> None:
        t0, t1 = self.attach_window
        if not (t0 - 1e-9 <= t <= t1 + 1e-9):
            if self._local is not None:
                # Soft release: zero pinch velocities once so VBD does not inherit a yank.
                self.s0.particle_qd.zero_()
                self._local = None
            return
        body_q = self.s0.body_q.numpy()
        particles = self.s0.particle_q.numpy().copy()
        if self._local is None:
            # Capture hand-local offsets from *current* particle positions (xyzw quat).
            self._local = {}
            for pref, ids in self.pinch.items():
                hand = self.body_index.get(self.hand_bodies.get(pref, pref + "hand"))
                if hand is None:
                    continue
                pq = body_q[hand]
                p, quat = pq[:3], pq[3:]
                q_inv = np.array([-quat[0], -quat[1], -quat[2], quat[3]])
                self._local[pref] = np.array(
                    [_rotate_xyzw(q_inv, particles[i] - p) for i in ids], float)
        for pref, ids in self.pinch.items():
            hand = self.body_index.get(self.hand_bodies.get(pref, pref + "hand"))
            if hand is None or pref not in self._local:
                continue
            pq = body_q[hand]
            p, quat = pq[:3], pq[3:]
            for i, vid in enumerate(ids):
                particles[vid] = p + _rotate_xyzw(quat, self._local[pref][i])
        self.s0.particle_q.assign(particles)
        self.s0.particle_qd.zero_()

    def contacts(self) -> dict:
        e = np.zeros(0, int)
        return {"a": e, "b": e, "pos": np.zeros((0, 3)), "force": np.zeros((0, 3)),
                "dist": np.zeros(0), "net": np.zeros((len(self.scene.bodies), 3))}
