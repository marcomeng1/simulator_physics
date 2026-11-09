"""Deformable soft-insertion: continuum peg/bore material cases (not the rigid insertion suite).

Kinds (same c100-tilted impedance motion as the rigid headline):
  fem-peg   A1: Genesis FEM polymer peg, rigid bore (sim: fem_mode=peg)
  fem-bore  A2: rigid peg, Genesis FEM polymer bore sleeve (sim: fem_mode=bore)
  flex-peg  M1/M2: MuJoCo flex elastic/plastic peg, rigid bore (routes FLEX only)

Hand-frame contact wrench is the primary force metric; depth uses the same tip
kinematics as rigid insertion.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from simbench import tapes
from simbench.models import ARM_JOINTS, MASKS, PANDA_BODIES, Q_READY, _fmt, contact_attrs, panda_tree
from simbench.scene import SCENE_DIR, SOLIMP, SOLREF, Scene
from simbench.scenes.insertion import (
    BORE_DEPTH,
    BORE_ORIGIN,
    BORE_OUTER,
    CHAMFER,
    DEPTH_PASS,
    DEPTH_TARGET,
    DESCENT,
    DURATION,
    FACETS,
    FEM_PEG,
    F_CAP,
    F_STOP,
    GUIDED,
    M_CAP,
    M_STOP,
    OFFSET,
    PASS_WINDOW,
    PEG_LENGTH,
    PEG_MOUNT,
    PEG_R,
    START_GAP,
    TABLE_HALF,
    TABLE_POS,
    T_END,
    T_START,
    Impedance,
    bore_pieces,
    peg_mount,
    peg_tip_hand,
    peg_vertices,
)
from simbench.scenes.panda import CTRL_DT

# Continuum polymer card (matches Genesis FEM.Elastic default used in A1/A2).
POLYMER = dict(FEM_PEG)
# MuJoCo flex peg uses the ball-suite soft card (20 kPa): E=1e6 + weld constraints
# is immediately unstable at practical timesteps. Genesis A1/A2 keep E=1e6 separately.
# Soft card (same E/nu as the ball suite) but higher elastic damping: a hand-mounted
# flex peg under insertion contact is much stiffer than free-ball compression.
FLEX_YOUNG, FLEX_POISSON, FLEX_DAMPING = 2e4, 0.3, 2e-3
# Plastic yield strain for M2 (von Mises return mapping, MuJoCo >= 4.0).
FLEX_YIELD = 0.05
BORE_WALL = 0.006  # FEM sleeve radial thickness [m]
FLEX_NR, FLEX_NTH, FLEX_NZ = 1, 6, 3  # very coarse cylinder lattice for flex peg
FLEX_MOUNT_LAYERS = 2  # fix mount + one shaft ring under the hand (cuts tip flop)


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def cylinder_tet_mesh(radius: float, height: float, nr: int = FLEX_NR,
                      nth: int = FLEX_NTH, nz: int = FLEX_NZ) -> dict:
    """Coarse solid cylinder: radial × angular × axial lattice → tets.

    Vertices are ordered so the mount end (z = -height/2 in peg frame, toward the
    hand after peg_mount) is the first nth*(nr+1) vertices when we later reorient —
    here z runs from -h/2 (mount) to +h/2 (tip).
    """
    zs = np.linspace(-height / 2, height / 2, nz + 1)
    rs = np.linspace(0.0, radius, nr + 1)
    thetas = np.linspace(0.0, 2 * np.pi, nth, endpoint=False)
    verts = []
    index = {}
    for iz, z in enumerate(zs):
        for ir, r in enumerate(rs):
            if ir == 0:
                index[(iz, ir, 0)] = len(verts)
                verts.append([0.0, 0.0, z])
                continue
            for it, th in enumerate(thetas):
                index[(iz, ir, it)] = len(verts)
                verts.append([r * np.cos(th), r * np.sin(th), z])
    verts = np.asarray(verts, float)

    def vid(iz, ir, it):
        return index[(iz, ir, 0 if ir == 0 else it % nth)]

    tets = []
    for iz in range(nz):
        for ir in range(nr):
            for it in range(nth):
                # Prism between rings; split into tets. Centre column (ir=0) is a wedge.
                if ir == 0:
                    a0, a1 = vid(iz, 0, 0), vid(iz + 1, 0, 0)
                    b0, b1 = vid(iz, 1, it), vid(iz + 1, 1, it)
                    c0, c1 = vid(iz, 1, it + 1), vid(iz + 1, 1, it + 1)
                    tets += [[a0, b0, c0, a1], [b0, c0, a1, b1], [c0, a1, b1, c1]]
                else:
                    # Hex-prism corners
                    p000, p001 = vid(iz, ir, it), vid(iz, ir, it + 1)
                    p010, p011 = vid(iz, ir + 1, it), vid(iz, ir + 1, it + 1)
                    p100, p101 = vid(iz + 1, ir, it), vid(iz + 1, ir, it + 1)
                    p110, p111 = vid(iz + 1, ir + 1, it), vid(iz + 1, ir + 1, it + 1)
                    # Standard 5- or 6-tet hex split (6 tets).
                    tets += [
                        [p000, p001, p010, p100],
                        [p001, p011, p010, p111],
                        [p001, p010, p100, p111],
                        [p001, p100, p101, p111],
                        [p010, p100, p110, p111],
                        [p100, p101, p111, p110],
                    ]
    # Surface faces for viz (boundary triangles of the outer wall + caps — approximate).
    faces = []
    for iz in range(nz):
        for it in range(nth):
            a, b = vid(iz, nr, it), vid(iz, nr, it + 1)
            c, d = vid(iz + 1, nr, it), vid(iz + 1, nr, it + 1)
            faces += [[a, b, d], [a, d, c]]
    for it in range(nth):
        faces.append([vid(0, 0, 0), vid(0, nr, it + 1), vid(0, nr, it)])
        faces.append([vid(nz, 0, 0), vid(nz, nr, it), vid(nz, nr, it + 1)])
    # Mount: first FLEX_MOUNT_LAYERS axial rings (hand end), so the shaft is not a free
    # single-layer hinge under contact.
    n_mount_z = min(FLEX_MOUNT_LAYERS, nz + 1)
    mount = [vid(iz, ir, it)
             for iz in range(n_mount_z)
             for ir in range(nr + 1)
             for it in (range(1) if ir == 0 else range(nth))]
    mass = np.full(len(verts), POLYMER["rho"] * np.pi * radius**2 * height / len(verts))
    return {"vertices": verts, "tets": np.asarray(tets, int), "faces": np.asarray(faces, int),
            "mass": mass, "mount": mount}


def bore_sleeve_mesh(clearance: float, wall: float = BORE_WALL,
                     nth: int = 24, nz: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Watertight annular cylinder (bore lining) as triangle mesh for Genesis tet remesh."""
    r_in = PEG_R + clearance
    r_out = r_in + wall
    zs = np.linspace(-BORE_DEPTH, 0.0, nz + 1)
    thetas = np.linspace(0.0, 2 * np.pi, nth, endpoint=False)
    # Vertex rings: inner then outer at each z.
    verts = []
    for z in zs:
        for r in (r_in, r_out):
            for th in thetas:
                verts.append([r * np.cos(th), r * np.sin(th), z])
    verts = np.asarray(verts, float)
    faces = []

    def idx(iz, side, it):  # side 0=inner 1=outer
        return iz * 2 * nth + side * nth + (it % nth)

    for iz in range(nz):
        for it in range(nth):
            # Outer wall
            faces += [[idx(iz, 1, it), idx(iz, 1, it + 1), idx(iz + 1, 1, it + 1)],
                      [idx(iz, 1, it), idx(iz + 1, 1, it + 1), idx(iz + 1, 1, it)]]
            # Inner wall (inward winding)
            faces += [[idx(iz, 0, it), idx(iz + 1, 0, it), idx(iz + 1, 0, it + 1)],
                      [idx(iz, 0, it), idx(iz + 1, 0, it + 1), idx(iz, 0, it + 1)]]
    # Caps at z min / z max
    for iz in (0, nz):
        for it in range(nth):
            a, b = idx(iz, 0, it), idx(iz, 0, it + 1)
            c, d = idx(iz, 1, it), idx(iz, 1, it + 1)
            if iz == 0:
                faces += [[a, c, d], [a, d, b]]
            else:
                faces += [[a, d, c], [a, b, d]]
    return verts, np.asarray(faces, int)


def _rigid_insertion_mjcf(dt: float, clearance: float, case: str,
                          disable_bore: bool = False, disable_peg: bool = False) -> str:
    import mujoco

    tree = panda_tree("peg", SCENE_DIR)
    root = tree.getroot()
    root.find("option").set("timestep", repr(dt))
    asset, world = root.find("asset"), root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="box", pos=_fmt(TABLE_POS),
                  size=_fmt(TABLE_HALF), rgba="0.55 0.5 0.45 1", **contact_attrs("static"))
    bore_attrs = contact_attrs("static")
    if disable_bore:
        bore_attrs = {**bore_attrs, "contype": "0", "conaffinity": "0"}
    for k, piece in enumerate(bore_pieces(clearance)):
        ET.SubElement(asset, "mesh", name=f"bore_{k:03d}", vertex=_fmt(np.ravel(piece)))
        ET.SubElement(world, "geom", name=f"bore_{k:03d}", type="mesh", mesh=f"bore_{k:03d}",
                      pos=_fmt(BORE_ORIGIN), rgba="0.6 0.62 0.66 1", **bore_attrs)
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="hand_origin", size="0.005")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, peg_mount(case).ravel())
    peg = ET.SubElement(hand, "body", name="peg", pos=_fmt([0.0, 0.0, PEG_MOUNT]),
                        quat=_fmt(quat))
    from simbench.models import PADDLE_MASS
    m = PADDLE_MASS
    ixx = m * (3 * PEG_R**2 + PEG_LENGTH**2) / 12
    ET.SubElement(peg, "inertial", pos="0 0 0", mass=repr(m),
                  diaginertia=_fmt([ixx, ixx, m * PEG_R**2 / 2]))
    peg_attrs = contact_attrs("paddle")
    if disable_peg:
        peg_attrs = {**peg_attrs, "contype": "0", "conaffinity": "0"}
    ET.SubElement(asset, "mesh", name="peg_mesh", vertex=_fmt(np.ravel(peg_vertices())))
    ET.SubElement(peg, "geom", name="peg_geom", type="mesh", mesh="peg_mesh",
                  rgba="0.85 0.55 0.2 1", **peg_attrs)
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode")


def _flex_peg_mjcf(dt: float, clearance: float, case: str, yield_strain: float) -> tuple[str, dict]:
    """Rigid panda + bore; soft peg is a flex cylinder parented under the hand.

    Mount vertices are welded to the hand in the hand frame (no world equality yank).
    Tip vertices keep three slide DOFs in the hand frame so the continuum can deform.
    """
    import mujoco

    mesh = cylinder_tet_mesh(PEG_R, PEG_LENGTH)
    tree = panda_tree("pads", SCENE_DIR)
    root = tree.getroot()
    root.find("option").set("timestep", repr(dt))
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="box", pos=_fmt(TABLE_POS),
                  size=_fmt(TABLE_HALF), rgba="0.55 0.5 0.45 1", **contact_attrs("static"))
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for k, piece in enumerate(bore_pieces(clearance)):
        ET.SubElement(asset, "mesh", name=f"bore_{k:03d}", vertex=_fmt(np.ravel(piece)))
        ET.SubElement(world, "geom", name=f"bore_{k:03d}", type="mesh", mesh=f"bore_{k:03d}",
                      pos=_fmt(BORE_ORIGIN), rgba="0.6 0.62 0.66 1", **contact_attrs("static"))
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="hand_origin", size="0.005")

    R = peg_mount(case)
    mount_set = set(mesh["mount"])
    bodies = []
    for i, local in enumerate(mesh["vertices"]):
        name = f"peg_{i}"
        bodies.append(name)
        # Peg-frame vertex → hand frame (hand z along peg for untilted).
        p_hand = np.array([0.0, 0.0, PEG_MOUNT]) + R @ local
        body = ET.SubElement(hand, "body", name=name, pos=_fmt(p_hand))
        ET.SubElement(body, "inertial", pos="0 0 0", mass=repr(float(mesh["mass"][i])),
                      diaginertia="1e-9 1e-9 1e-9")
        if i in mount_set:
            # Mount ring: no joints → fixed in the hand frame.
            pass
        else:
            for ax, axis in zip("xyz", ("1 0 0", "0 1 0", "0 0 1"), strict=True):
                ET.SubElement(body, "joint", name=f"{name}_{ax}", type="slide", axis=axis)

    contype, _ = MASKS["paddle"]
    flex = ET.SubElement(ET.SubElement(root, "deformable"), "flex", name="peg", dim="3",
                         radius="0", body=" ".join(bodies),
                         element=" ".join(str(int(i)) for i in mesh["tets"].ravel()),
                         rgba="0.9 0.5 0.25 1")
    ET.SubElement(flex, "contact", contype=str(contype), conaffinity=str(1 | 2), condim="3",
                  friction="0.5 0 0", solref=SOLREF, solimp=SOLIMP, selfcollide="none")
    el_kw = {"young": repr(FLEX_YOUNG), "poisson": repr(FLEX_POISSON),
             "damping": repr(FLEX_DAMPING)}
    if yield_strain:
        el_kw["yield"] = repr(yield_strain)
    ET.SubElement(flex, "elasticity", **el_kw)

    # Ghost rigid peg for tip kinematics / body index "peg".
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.ravel())
    peg = ET.SubElement(hand, "body", name="peg", pos=_fmt([0.0, 0.0, PEG_MOUNT]),
                        quat=_fmt(quat))
    ET.SubElement(peg, "inertial", pos="0 0 0", mass="1e-4", diaginertia="1e-8 1e-8 1e-8")
    ET.SubElement(peg, "geom", name="peg_geom", type="cylinder",
                  size=_fmt([PEG_R * 0.5, PEG_LENGTH / 2]),
                  rgba="0.85 0.55 0.2 0.15", contype="0", conaffinity="0")

    ET.indent(tree)
    meta = {"vertices": mesh["vertices"].tolist(), "tets": mesh["tets"].tolist(),
            "faces": mesh["faces"].tolist(), "mount": mesh["mount"],
            "node_mass": mesh["mass"].tolist(), "young": FLEX_YOUNG, "poisson": FLEX_POISSON,
            "yield": yield_strain, "damping": FLEX_DAMPING}
    return ET.tostring(root, encoding="unicode"), meta


def soft_insertion(dt: float = 0.001, clearance_mm: float = 0.10, case: str = "tilted",
                   guided: bool = False, kind: str = "fem-peg",
                   yield_strain: float = 0.0) -> Scene:
    """kind: fem-peg | fem-bore | flex-peg."""
    import mujoco

    if kind not in ("fem-peg", "fem-bore", "flex-peg"):
        raise ValueError(f"unknown soft-insertion kind {kind!r}")
    clearance = clearance_mm * 1e-3
    r_tip = peg_tip_hand(case)
    tip0 = BORE_ORIGIN + [OFFSET if case == "offset" else 0.0, 0.0, START_GAP]
    rot0 = np.diag([1.0, -1.0, -1.0])
    flex_meta = {}
    if kind == "flex-peg":
        mjcf, flex_meta = _flex_peg_mjcf(dt, clearance, case, yield_strain)
        bodies = [*PANDA_BODIES, "peg", *[f"peg_{i}" for i in range(len(flex_meta["vertices"]))]]
    else:
        mjcf = _rigid_insertion_mjcf(
            dt, clearance, case,
            disable_bore=(kind == "fem-bore"),
            disable_peg=(kind == "fem-peg"))
        bodies = [*PANDA_BODIES, "peg"]

    params = {
        "task": "soft-insertion", "kind": kind, "dt": dt, "duration": DURATION,
        "clearance_mm": clearance_mm, "case": case,
        "guided": GUIDED if guided else None, "ctrl_dt": CTRL_DT,
        "bore_origin": BORE_ORIGIN.tolist(), "peg_tip_hand": r_tip.tolist(),
        "depth_target_m": DEPTH_TARGET, "depth_pass_m": DEPTH_PASS,
        "pass_window_s": PASS_WINDOW,
        "gains": {"k_t": 1500.0, "k_r": 30.0, "d_t": 80.0, "d_r": 1.0,
                  "k_null": 5.0, "d_null": 1.0},
        "caps": [F_CAP, M_CAP], "stops": [F_STOP, M_STOP], "friction": 0.5,
        "polymer": POLYMER, "yield_strain": yield_strain,
        "t_start": T_START, "t_end": T_END, "descent": DESCENT,
    }
    if flex_meta:
        params["flex_peg"] = flex_meta
        params["faces"] = flex_meta["faces"]
        params["tets"] = flex_meta["tets"]
        params["flex_prefix"] = "peg_"
    if kind == "fem-bore":
        v, f = bore_sleeve_mesh(clearance)
        params["bore_sleeve"] = {"vertices": v.tolist(), "faces": f.tolist(),
                                 "wall_m": BORE_WALL}

    # Flex runs at 50 us; record at 2 ms like the ball suite so traces stay manageable.
    record_dt = 0.002 if kind == "flex-peg" else None
    scene = Scene(name="soft-insertion", mjcf=mjcf, dt=dt, duration=DURATION,
                  bodies=bodies, joints=ARM_JOINTS, params=params, record_dt=record_dt)
    path = scene.mjcf_path()
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    q_seed = np.asarray(d.qpos, dtype=float).copy()
    for name, value in zip(ARM_JOINTS, Q_READY, strict=True):
        q_seed[m.joint(name).qposadr[0]] = value
    q_full = tapes.ik(m, d, "hand_origin", tip0 - rot0 @ r_tip, rot0, q_seed)
    q0 = np.array([q_full[m.joint(n).qposadr[0]] for n in ARM_JOINTS])
    scene.joint_init = dict(zip(ARM_JOINTS, q0.tolist(), strict=True))
    if kind == "flex-peg":
        scene.controller = SoftImpedance(path, q0, tip0, r_tip, GUIDED if guided else None, dt)
    else:
        scene.controller = Impedance(path, q0, tip0, r_tip, GUIDED if guided else None, dt)
    return scene


class SoftImpedance(Impedance):
    """Impedance for models whose nq >> 7 (MuJoCo flex peg vertices).

    Writes only the seven arm coordinates into the controller's MjData; flex free
    vertices keep whatever the physics step left. Contact observe aggregates all
    ``peg`` / ``peg_*`` contact-name indices for overload and guided feedback.
    """

    def __init__(self, path, q0, tip0, r_tip, guided, dt):
        super().__init__(path, q0, tip0, r_tip, guided, dt)
        self.arm_q = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_v = np.array([self.m.jnt_dofadr[self.m.joint(j).id] for j in ARM_JOINTS])
        self.peg_ids = None

    def __call__(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        from simbench.scenes.insertion import (
            CTRL_DT as _CD, D_NULL, D_R, D_T, F_CAP, K_NULL, K_R, K_T, M_CAP, _clip_norm,
        )
        from simbench.scenes.panda import LIMIT

        i = t / _CD
        k = int(round(i))
        if self.u is not None and abs(i - k) >= 1e-6:
            return self.u
        mj, m, d = self.mj, self.m, self.d
        d.qpos[self.arm_q] = q
        d.qvel[self.arm_v] = qd
        mj.mj_forward(m, d)
        x, rot = d.xpos[self.hand].copy(), d.xmat[self.hand].reshape(3, 3)
        mj.mj_jacBody(m, d, self.jacp, self.jacr, self.hand)
        jac = np.vstack([self.jacp[:, self.arm_v], self.jacr[:, self.arm_v]])
        v = jac @ qd
        x_ref, rot_ref, v_ref = self.reference(t)
        e_rot = 0.5 * np.cross(rot, rot_ref, axis=0).sum(axis=1)
        force = _clip_norm(K_T * (x_ref - x) + D_T * (v_ref - v[:3]), F_CAP)
        moment = _clip_norm(K_R * e_rot - D_R * v[3:], M_CAP)
        null = np.eye(7) - jac.T @ np.linalg.pinv(jac).T
        # Bias for arm dofs only.
        bias = d.qfrc_bias[self.arm_v]
        tau = (jac.T @ np.concatenate([force, moment]) + bias
               + null @ (K_NULL * (self.q0 - q) - D_NULL * qd))
        self.x, self.tip = x, x + rot @ self.r_tip
        self.u = np.clip(tau, -LIMIT[:7], LIMIT[:7])
        return self.u

    def observe(self, t: float, contacts: dict, names: list[str]):
        from simbench.scenes.insertion import _clip_norm

        if self.peg_ids is None:
            self.peg_ids = [i for i, n in enumerate(names)
                            if n == "peg" or n.startswith("peg_")]
        if not self.peg_ids:
            return None
        on = np.zeros(len(contacts["a"]))
        for pid in self.peg_ids:
            on += np.where(contacts["b"] == pid, 1.0, 0.0) - np.where(
                contacts["a"] == pid, 1.0, 0.0)
        f = contacts["force"] * on[:, None]
        force = f.sum(axis=0)
        # No rigid F_STOP: foam/flex contact spikes are expected and are the metric
        # (peak_hand_force_N), not a run abort. Nonfinite/inverted tets still stop via
        # the simulate() loop.
        g = self.guided
        if g is None:
            return None
        tip_moment = np.cross(contacts["pos"] - self.tip, f).sum(axis=0)
        alpha = self.dt / (g["force_filter_s"] + self.dt)
        self.f_filt += alpha * (force - self.f_filt)
        self.m_filt += alpha * (tip_moment - self.m_filt)
        rate = _clip_norm(g["lateral_admittance_m_Ns"] * self.f_filt[:2],
                          g["lateral_speed_m_s"])
        self.lateral = _clip_norm(self.lateral + rate * self.dt, g["lateral_travel_m"])
        self.relief = min(g["upward_relief_m_N"] * max(self.f_filt[2], 0.0),
                          g["upward_relief_limit_m"])
        rate = _clip_norm(g["angular_admittance_rad_Nms"] * self.m_filt[:2],
                          g["angular_speed_rad_s"])
        self.tilt = _clip_norm(self.tilt + rate * self.dt, g["angular_limit_rad"])
        return None
