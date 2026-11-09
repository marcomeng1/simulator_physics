"""Genesis FEM soft-insertion hybrids (float32, implicit FEM, GPU).

Modes (``fem_mode``):
  peg   A1: FEM polymer peg + rigid bore; mount verts welded to hand
  bore  A2: rigid peg + FEM polymer bore sleeve fixed in world

Attachment for A1 (see scripts/probes/insertion_fem_spike.py):
  FEMOptions(use_implicit_solver=True, enable_vertex_constraints=True)
  + LegacyCouplerOptions(rigid_fem=True)
  + hard set_vertex_constraints(mount, link=hand)

Fallback: kinematic mount drive (one-way).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import genesis as gs
import numpy as np

from simbench.adapters.genesis_rigid import apply_mjcf_dof_properties
from simbench.readback import quat_to_mat
from simbench.scenes.insertion import BORE_ORIGIN, PEG_LENGTH, PEG_MOUNT, PEG_R, peg_mount
from simbench.scenes.soft_insertion import BORE_WALL, bore_sleeve_mesh

_INITIALIZED = False


def _init() -> None:
    global _INITIALIZED
    if not _INITIALIZED:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
        _INITIALIZED = True


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy().astype(float)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _patch_mjcf(path, *, disable_peg: bool, disable_bore: bool) -> Path:
    tree = ET.parse(path)
    if disable_peg:
        peg_geom = tree.find(".//geom[@name='peg_geom']")
        if peg_geom is not None:
            peg_geom.set("contype", "0")
            peg_geom.set("conaffinity", "0")
    if disable_bore:
        for g in tree.findall(".//geom"):
            name = g.get("name") or ""
            if name.startswith("bore_"):
                g.set("contype", "0")
                g.set("conaffinity", "0")
    tag = ("nocolpeg" if disable_peg else "") + ("nocolbore" if disable_bore else "")
    out = path.with_name(path.stem + f"_{tag or 'fem'}.xml")
    tree.write(out)
    return out


def _fem_verts(entity) -> np.ndarray:
    arr = _np(entity.get_state().pos)
    return arr[0] if arr.ndim == 3 else arr


class Sim:
    """Genesis rigid Panda + FEM soft peg and/or soft bore sleeve."""

    def __init__(self, scene, seed: int = 0, matched_options: bool = True,
                 fem_mode: str = "peg",
                 E: float = 1e6, nu: float = 0.3, rho: float = 1000.0,
                 friction_mu: float = 0.5, hydroelastic_modulus: float = 1e7,
                 attachment: str = "coupler", n_mount: int = 48,
                 fem_damping: float = 20.0, n_newton_iterations: int = 5,
                 **_extra):
        import torch

        if fem_mode not in ("peg", "bore"):
            raise ValueError(f"fem_mode must be 'peg' or 'bore', got {fem_mode!r}")
        self.torch = torch
        _init()
        self.scene = scene
        self.fem_mode = fem_mode
        self.attachment = attachment
        self.fem_material = {
            "E": E, "nu": nu, "rho": rho, "friction_mu": friction_mu,
            "hydroelastic_modulus": hydroelastic_modulus, "mode": fem_mode,
        }
        case = scene.params.get("case", "aligned")
        clearance = float(scene.params.get("clearance_mm", 0.10)) * 1e-3
        path = _patch_mjcf(scene.mjcf_path(),
                           disable_peg=(fem_mode == "peg"),
                           disable_bore=(fem_mode == "bore"))

        self.gs_scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=scene.dt, gravity=(0.0, 0.0, -9.81)),
            rigid_options=gs.options.RigidOptions(dt=scene.dt),
            fem_options=gs.options.FEMOptions(
                dt=scene.dt, damping=fem_damping,
                use_implicit_solver=True, enable_vertex_constraints=True,
                n_newton_iterations=n_newton_iterations,
            ),
            coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=True),
            show_viewer=False,
        )
        self.entity = e = self.gs_scene.add_entity(gs.morphs.MJCF(file=str(path)))
        mat = gs.materials.FEM.Elastic(
            E=E, nu=nu, rho=rho, friction_mu=friction_mu,
            hydroelastic_modulus=hydroelastic_modulus)

        self.peg = None
        self.bore = None
        self.mount: list[int] = []
        if fem_mode == "peg":
            self.peg = self.gs_scene.add_entity(
                morph=gs.morphs.Cylinder(
                    height=PEG_LENGTH, radius=PEG_R, pos=(0.0, 0.0, 1.0),
                    force_retet=True, visualization=True, collision=True),
                material=mat, name="fem_peg")
        else:
            verts, faces = bore_sleeve_mesh(clearance, wall=BORE_WALL)
            obj = path.with_name(path.stem + "_sleeve.obj")
            _write_obj(obj, verts, faces)
            self.bore = self.gs_scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=str(obj), pos=tuple(BORE_ORIGIN), force_retet=True,
                    visualization=True, collision=True, scale=1.0),
                material=mat, name="fem_bore")

        self.gs_scene.build()
        self.notes = apply_mjcf_dof_properties(path, e)
        self.notes.append(
            f"fem_mode={fem_mode} E={E:g} Pa nu={nu} rho={rho} mu={friction_mu}; "
            f"attachment={attachment}")

        links = {link.name: link for link in e.links}
        # All MJCF bodies including rigid peg (A2) stay in the link list.
        self.links = [links[n] for n in scene.bodies if n in links]
        self.hand = links["hand"]
        self.peg_body_idx = scene.bodies.index("peg") if "peg" in scene.bodies else 0
        joints = {j.name: j for j in e.joints}
        self.dofs = [joints[n].dofs_idx_local[0] for n in scene.joints]
        self.qs = [joints[n].qs_idx_local[0] for n in scene.joints]
        solver = e.solver
        self.link_global = [link.idx for link in self.links]
        self.contact_names = [link.name for link in solver.links]
        if fem_mode == "peg" and "peg" not in self.contact_names:
            self.contact_names = [*self.contact_names, "peg"]
        if fem_mode == "bore":
            self.contact_names = [*self.contact_names, "fem_bore"]
        self.fem_contact_idx = len(self.contact_names) - 1
        self.track = {g: i for i, g in enumerate(self.link_global)}
        self._body_of_link = {self.link_global[i]: scene.bodies.index(self.links[i].name)
                             for i in range(len(self.links))}

        if scene.joint_init:
            q = _np(e.get_qpos())
            for name, value in scene.joint_init.items():
                q[joints[name].qs_idx_local[0]] = value
            e.set_qpos(q)
        self.gs_scene.step()

        self._R_mount = peg_mount(case)
        self._peak_strain = 0.0
        self._rest_local = None
        if fem_mode == "peg":
            self._place_and_attach_peg(n_mount)
            if attachment == "kinematic":
                self._cache_mount_local()
        else:
            self._fix_bore_outer()

        self._tau = torch.zeros(e.n_dofs, dtype=gs.tc_float, device=gs.device)

    def _place_and_attach_peg(self, n_mount: int) -> None:
        hand_pos = _np(self.hand.get_pos())
        R_hand = quat_to_mat(_np(self.hand.get_quat()))
        R_peg = R_hand @ self._R_mount
        peg_com = hand_pos + R_hand @ np.array([0.0, 0.0, PEG_MOUNT])
        tip_dir = R_peg @ np.array([0.0, 0.0, 1.0])
        rest = _np(self.peg.init_positions)
        if rest.ndim == 3:
            rest = rest[0]
        local = rest - rest.mean(axis=0)
        self.peg.set_position(peg_com + (R_peg @ local.T).T)
        self.peg.set_velocity(np.zeros(3))
        arr = _fem_verts(self.peg)
        proj = (arr - peg_com) @ tip_dir
        self.mount = np.argsort(proj)[:n_mount].tolist()
        if self.attachment == "coupler":
            self.peg.set_vertex_constraints(
                self.mount, link=self.hand, is_soft_constraint=False)
        else:
            self.peg.set_vertex_constraints(
                self.mount, target_poss=arr[self.mount], is_soft_constraint=False)

    def _fix_bore_outer(self) -> None:
        """Pin outer-radius sleeve verts in world so the lining stays seated."""
        arr = _fem_verts(self.bore)
        com = arr.mean(axis=0)
        radial = np.linalg.norm(arr[:, :2] - com[:2], axis=1)
        # Outer half of verts by radius.
        thresh = np.median(radial)
        self.mount = np.where(radial >= thresh)[0].tolist()
        self.bore.set_vertex_constraints(
            self.mount, target_poss=arr[self.mount], is_soft_constraint=False)

    def _cache_mount_local(self) -> None:
        R = quat_to_mat(_np(self.hand.get_quat()))
        lp = _np(self.hand.get_pos())
        arr = _fem_verts(self.peg)
        self._rest_local = (R.T @ (arr[self.mount] - lp).T).T

    def _drive_mount_kinematic(self) -> None:
        R = quat_to_mat(_np(self.hand.get_quat()))
        lp = _np(self.hand.get_pos())
        tgt = lp + (R @ self._rest_local.T).T
        self.peg.set_vertex_constraints(
            self.mount, target_poss=tgt, is_soft_constraint=False)

    def _update_strain(self, entity) -> None:
        arr = _fem_verts(entity)
        com = arr.mean(axis=0)
        rest = _np(entity.init_positions)
        if rest.ndim == 3:
            rest = rest[0]
        disp = (np.linalg.norm(arr - com, axis=1)
                - np.linalg.norm(rest - rest.mean(axis=0), axis=1))
        self._peak_strain = max(self._peak_strain, float(np.max(np.abs(disp))))

    def _fem_peg_pose(self) -> tuple[np.ndarray, np.ndarray]:
        arr = _fem_verts(self.peg)
        com = arr.mean(axis=0)
        R_hand = quat_to_mat(_np(self.hand.get_quat()))
        tip_dir = R_hand @ self._R_mount @ np.array([0.0, 0.0, 1.0])
        z = tip_dir / (np.linalg.norm(tip_dir) + 1e-12)
        x = np.cross(R_hand[:, 1], z)
        if np.linalg.norm(x) < 1e-8:
            x = np.cross(R_hand[:, 0], z)
        x /= np.linalg.norm(x) + 1e-12
        y = np.cross(z, x)
        self._update_strain(self.peg)
        return com, _mat_to_quat(np.column_stack([x, y, z]))

    def settings(self) -> dict:
        return {"engine": "genesis-fem", "version": gs.__version__, "precision": "float32",
                "device": str(gs.device), "fem": self.fem_material,
                "fem_mode": self.fem_mode, "attachment": self.attachment,
                "peak_vertex_disp_m": self._peak_strain}

    def effective_model(self) -> dict:
        solver = self.entity.solver
        mass = _np(solver.get_links_mass(self.link_global))
        out = {self.links[i].name: {"mass": float(mass[i])} for i in range(len(self.links))}
        out["fem"] = dict(self.fem_material)
        return out

    def state(self) -> dict:
        solver = self.entity.solver
        idx = self.link_global
        pos = _np(solver.get_links_pos(idx))
        quat = _np(solver.get_links_quat(idx))
        com = pos + np.array([quat_to_mat(q) @ c for q, c in
                              zip(quat, _np(solver.get_links_COM(idx)), strict=True)])
        lin = _np(solver.get_links_vel(idx, ref=gs.link_ref_frame.link_COM))
        ang = _np(solver.get_links_ang(idx))
        n = len(self.scene.bodies)
        # Map solver link rows → scene body rows by name.
        name_of = [link.name for link in self.links]
        def _scatter(arr, n_col):
            out = np.zeros((n, n_col))
            for j, name in enumerate(name_of):
                if name in self.scene.bodies:
                    out[self.scene.bodies.index(name)] = arr[j]
            return out

        out_pos, out_com = _scatter(pos, 3), _scatter(com, 3)
        out_quat, out_lin = _scatter(quat, 4), _scatter(lin, 3)
        out_ang = _scatter(ang, 3)
        if self.fem_mode == "peg":
            peg_com, peg_quat = self._fem_peg_pose()
            out_pos[self.peg_body_idx] = peg_com
            out_com[self.peg_body_idx] = peg_com
            out_quat[self.peg_body_idx] = peg_quat
        elif self.bore is not None:
            self._update_strain(self.bore)
        q, qd = _np(self.entity.get_qpos()), _np(self.entity.get_dofs_velocity())
        return {"pos": out_pos, "com": out_com, "quat": out_quat, "linvel": out_lin,
                "angvel": out_ang,
                "q": q[self.qs] if self.qs else np.zeros(0),
                "qd": qd[self.dofs] if self.dofs else np.zeros(0)}

    def step(self, tau: np.ndarray) -> None:
        if self.fem_mode == "peg" and self.attachment == "kinematic":
            self._drive_mount_kinematic()
        if self.dofs:
            self._tau.zero_()
            self._tau[self.dofs] = self.torch.as_tensor(
                tau, dtype=gs.tc_float, device=gs.device)
            self.entity.control_dofs_force(self._tau)
        self.gs_scene.step()

    def contacts(self) -> dict:
        c = self.entity.get_contacts()
        a = c["link_a"].cpu().numpy().astype(int)
        b = c["link_b"].cpu().numpy().astype(int)
        pos = _np(c["position"]).reshape(-1, 3)
        force = _np(c["force_b"]).reshape(-1, 3)
        net = np.zeros((len(self.scene.bodies), 3))
        for i in range(len(a)):
            if b[i] in self._body_of_link:
                net[self._body_of_link[b[i]]] += force[i]
            if a[i] in self._body_of_link:
                net[self._body_of_link[a[i]]] += _np(c["force_a"][i])
        dist = (_np(c["penetration"]).reshape(-1) * -1
                if "penetration" in c else np.zeros(len(a)))

        fem = self.peg if self.fem_mode == "peg" else self.bore
        if fem is not None:
            forces = fem._solver.get_forces()
            peg_force = np.zeros(3)
            peg_pos = _fem_verts(fem).mean(axis=0)
            if forces is not None:
                f = _np(forces)
                if f.ndim == 3:
                    f = f[0]
                mask = np.ones(len(f), dtype=bool)
                if self.mount:
                    mask[self.mount] = False
                if mask.any():
                    peg_force = f[mask].sum(axis=0)
                    w = np.linalg.norm(f[mask], axis=1)
                    if w.sum() > 1e-9:
                        peg_pos = (_fem_verts(fem)[mask] * w[:, None]).sum(0) / w.sum()
            if np.linalg.norm(peg_force) > 1e-6:
                # Hand-wrench metrics key off the rigid "peg" contact name. For A2 the
                # solver force lives on the bore sleeve — report the reaction on the peg.
                if self.fem_mode == "bore":
                    peg_force = -peg_force
                    cid = (self.contact_names.index("peg")
                           if "peg" in self.contact_names else self.fem_contact_idx)
                else:
                    cid = self.fem_contact_idx
                a = np.append(a, cid)
                b = np.append(b, 0)
                pos = np.vstack([pos, peg_pos]) if len(pos) else peg_pos.reshape(1, 3)
                force = (np.vstack([force, peg_force]) if len(force)
                         else peg_force.reshape(1, 3))
                dist = np.append(dist, 0.0)
                net[self.peg_body_idx] += peg_force
        return {"a": a, "b": b, "pos": pos, "force": force, "dist": dist, "net": net}
