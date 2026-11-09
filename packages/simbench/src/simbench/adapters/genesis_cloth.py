"""Genesis PBD Cloth + rigid dual-Franka for the cloth-fold scene."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rigid_mjcf(scene) -> Path:
    """MJCF without flex / cloth_* bodies (Genesis loads robots from this)."""
    root = ET.fromstring(scene.mjcf)
    for el in list(root.findall("deformable")):
        root.remove(el)
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        if (body.get("name") or "").startswith("cloth_"):
            world.remove(body)
    path = scene.mjcf_path().with_name(scene.mjcf_path().stem + "_rigid.xml")
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode"))
    return path


def _init():
    import genesis as gs
    if not getattr(gs, "_simbench_inited", False):
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        gs._simbench_inited = True
    return gs


def _write_cloth_obj(path, rest: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# simbench cloth-fold\n")
        for v in rest:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q`` in wxyz order."""
    w, u = q[0], q[1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


class Sim:
    def __init__(self, scene, seed: int = 0, cloth_pbd: bool = True, **_):
        import genesis as gs
        import torch

        self.torch = torch
        self.gs = gs
        _init()
        self.scene = scene
        p = scene.params
        cloth = p["cloth"]
        path = _rigid_mjcf(scene)
        rest = np.asarray(p["cloth_rest"], float).copy()
        faces = np.asarray(p["faces"], int)
        # Match MuJoCo grid spacing; oversized PBD particles made kinematic pads blast the sheet.
        size = float(cloth["size_m"])
        nx = max(int(cloth["nx"]), 2)
        particle_size = min(size / (nx - 1), 0.02)
        # Sit on the table top (z=0): rest z must clear the PBD particle radius.
        rest[:, 2] = 0.5 * particle_size + 0.002
        obj = path.with_name(path.stem + "_cloth.obj")
        _write_cloth_obj(obj, rest, faces)

        self.gs_scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=scene.dt, gravity=(0, 0, -9.81)),
            rigid_options=gs.options.RigidOptions(dt=scene.dt),
            pbd_options=gs.options.PBDOptions(
                dt=scene.dt,
                particle_size=particle_size,
                # Floor at table top (cloth has collision=False vs kinematic pads).
                lower_bound=(-1.0, -1.0, 0.5 * particle_size),
                upper_bound=(2.0, 2.0, 2.0),
            ),
            show_viewer=False,
        )
        self.robot = self.gs_scene.add_entity(gs.morphs.MJCF(file=str(path)))
        # No rigid↔PBD contacts: kinematic pads + soft contacts launched the napkin.
        # Grasp is the shared closed-finger attach; floor is PBD lower_bound.
        self.cloth = self.gs_scene.add_entity(
            morph=gs.morphs.Mesh(file=str(obj), fixed=False, collision=False),
            material=gs.materials.PBD.Cloth(
                rho=float(cloth["areal_density"]),
                static_friction=float(cloth["friction"]),
                kinetic_friction=float(cloth["friction"]),
                stretch_compliance=1e-6,
                bending_compliance=1e-4,
                stretch_relaxation=0.3,
                bending_relaxation=0.15,
                air_resistance=0.0,
            ),
            surface=gs.surfaces.Default(color=(0.85, 0.82, 0.72)),
            name="cloth",
        )
        self.gs_scene.build()
        joints = {j.name: j for j in self.robot.joints}
        self.dofs = [joints[n].dofs_idx_local[0] for n in scene.joints if n in joints]
        self.qs = [joints[n].qs_idx_local[0] for n in scene.joints if n in joints]
        q = self.robot.get_qpos().detach().cpu().numpy().astype(float).copy()
        for name, value in scene.joint_init.items():
            if name in joints:
                q[joints[name].qs_idx_local[0]] = value
        self.robot.set_qpos(q)
        self._tau = torch.zeros(self.robot.n_dofs, dtype=gs.tc_float, device=gs.device)
        self.rest0 = rest.copy()
        self.contact_names = list(scene.bodies)
        self.notes = ["genesis cloth: PBD.Cloth + kinematic tape + pinch attach"]
        self._link = {link.name: link for link in self.robot.links}
        self.attach_window = p.get("attach_window") or [0.65, 2.60]
        self._step = 0
        self._local = None
        self._pinch_particles = {"L_": [], "R_": []}
        from simbench.scenes.panda import CTRL_DT
        self.ctrl_dt = float(p.get("ctrl_dt") or CTRL_DT)

        particles = self._all_particles()
        self._vert_particle = np.array(
            [int(np.linalg.norm(particles - r, axis=1).argmin()) for r in rest], int)
        # 3 closest particles per pinch (matches DualTapeController.ATTACH_N).
        for pref, key in (("L_", "pinch_left"), ("R_", "pinch_right")):
            targets = [rest[int(vid)] for vid in (p.get(key) or [])[:3]]
            ids = []
            for target in targets:
                ids.append(int(np.linalg.norm(particles - target, axis=1).argmin()))
            self._pinch_particles[pref] = sorted(set(ids))

    def settings(self) -> dict:
        return {"engine": "genesis-pbd-cloth", "version": self.gs.__version__,
                "precision": "float32",
                "cloth_model": "PBD.Cloth", "cloth_solver": "genesis PBDSolver",
                "robot_drive": "kinematic-tape",
                "n_particles": int(self.cloth.n_particles)}

    def effective_model(self) -> dict:
        return {"cloth": "pbd", "n_vert": len(self.rest0),
                "n_particles": int(self.cloth.n_particles)}

    def _all_particles(self) -> np.ndarray:
        arr = self.cloth.get_particles_pos()
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, float)
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    def _cloth_verts(self) -> np.ndarray:
        particles = self._all_particles()
        return particles[self._vert_particle]

    def state(self) -> dict:
        n = len(self.scene.bodies)
        pos = np.zeros((n, 3))
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        com = np.zeros((n, 3))
        lin = np.zeros((n, 3))
        ang = np.zeros((n, 3))
        cloth = self._cloth_verts()
        for i, name in enumerate(self.scene.bodies):
            if name.startswith("cloth_"):
                idx = int(name.split("_", 1)[1])
                pos[i] = cloth[idx] if idx < len(cloth) else cloth[0]
                com[i] = pos[i]
            elif name in self._link:
                link = self._link[name]
                pos[i] = link.get_pos().detach().cpu().numpy().astype(float)
                quat[i] = link.get_quat().detach().cpu().numpy().astype(float)
                com[i] = pos[i]
        q = self.robot.get_qpos().detach().cpu().numpy().astype(float)
        qd = self.robot.get_dofs_velocity().detach().cpu().numpy().astype(float)
        return {"pos": pos, "com": com, "quat": quat, "linvel": lin, "angvel": ang,
                "q": q[self.qs] if self.qs else np.zeros(len(self.scene.joints)),
                "qd": qd[self.dofs] if self.dofs else np.zeros(len(self.scene.joints))}

    def step(self, tau: np.ndarray) -> None:
        t = self._step * self.scene.dt
        self._step += 1
        self._drive_robot(t)
        self._attach_pinch(t)
        self.gs_scene.step()
        self._drive_robot(t)
        self._attach_pinch(t)

    def _drive_robot(self, t: float) -> None:
        """Open-loop joint tape via position control (same kinematic intent as Newton)."""
        ctrl = self.scene.controller
        if not (self.dofs and self.qs and hasattr(ctrl, "qref")):
            return
        k = min(max(int(round(t / self.ctrl_dt)), 0), len(ctrl.qref) - 1)
        qref = self.torch.as_tensor(
            ctrl.qref[k], dtype=self.gs.tc_float, device=self.gs.device)
        self.robot.set_dofs_position(qref, self.dofs)
        self.robot.control_dofs_position(qref, self.dofs)

    def _attach_pinch(self, t: float) -> None:
        """Per-step pinch pin in the hand frame (coupler path was numerically unstable)."""
        t0, t1 = self.attach_window
        if not (t0 - 1e-9 <= t <= t1 + 1e-9):
            self._local = None
            return
        particles = self._all_particles()
        if self._local is None:
            self._local = {}
            for pref, pids in self._pinch_particles.items():
                link = self._link.get(pref + "hand")
                if link is None or not pids:
                    continue
                p = link.get_pos().detach().cpu().numpy().astype(float)
                quat = link.get_quat().detach().cpu().numpy().astype(float)  # wxyz
                q_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
                # Preserve current particle world poses on the first attach frame.
                self._local[pref] = np.array(
                    [_rotate_wxyz(q_inv, particles[i] - p) for i in pids], float)
        for pref, pids in self._pinch_particles.items():
            link = self._link.get(pref + "hand")
            if link is None or pref not in self._local or not pids:
                continue
            p = link.get_pos().detach().cpu().numpy().astype(float)
            quat = link.get_quat().detach().cpu().numpy().astype(float)
            worlds = np.array(
                [p + _rotate_wxyz(quat, self._local[pref][i]) for i in range(len(pids))],
                float)
            # Sanity: reject a bogus first write that would teleport the sheet.
            if float(np.linalg.norm(worlds - particles[pids])) > 0.25:
                self._local = None
                return
            self.cloth.set_particles_pos(
                self.torch.as_tensor(worlds, dtype=self.gs.tc_float, device=self.gs.device),
                particles_idx_local=pids)
            self.cloth.set_particles_vel(
                self.torch.zeros((len(pids), 3), dtype=self.gs.tc_float, device=self.gs.device),
                particles_idx_local=pids)

    def contacts(self) -> dict:
        e = np.zeros(0, int)
        return {"a": e, "b": e, "pos": np.zeros((0, 3)), "force": np.zeros((0, 3)),
                "dist": np.zeros(0), "net": np.zeros((len(self.scene.bodies), 3))}
