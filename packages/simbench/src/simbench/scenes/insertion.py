"""Phase 6 precision insertion: a torque-driven Panda inserts a rigid 20 mm peg into a bore.

Geometry follows the article's recorded insertion scene (`make fetch`): a 64-facet peg
(10 mm circumradius, 60 mm long, centred 135 mm down the hand axis) and a bore of 64
convex wedge pieces (inner apothem 10 mm + radial clearance, 1 mm 45 deg chamfer, 30 mm
deep). Cases: aligned, 0.25 mm lateral offset, or the peg tilted 1 deg about the hand Y
axis, always starting with the tip 2 mm above the bore.

The article's impedance controller is described but not published; this one is a
Cartesian impedance on the hand origin with gravity/Coriolis compensation from MuJoCo's
model of the same MJCF. Its stiffness (1.5 kN/m, 30 N m/rad) is inferred from the
article's stalled tilted runs; damping, the posture null-space term and the descent
timing are fitted to its free-motion depth curve. Task commands are capped at 20 N /
2 N m and the run stops once the peg contact resultant exceeds 40 N or its moment about
the hand origin exceeds 4 N m. The contact-guided variant uses the article's published
feedback parameters on the filtered peg contact force and its moment about the peg tip.
"""

import xml.etree.ElementTree as ET

import numpy as np

from simbench import tapes
from simbench.models import (
    ARM_JOINTS,
    PADDLE_MASS,
    PANDA_BODIES,
    Q_READY,
    _fmt,
    contact_attrs,
    panda_tree,
)
from simbench.scene import SCENE_DIR, Scene
from simbench.scenes.panda import CTRL_DT, LIMIT

DURATION = 4.0
BORE_ORIGIN = np.array([0.3063669943990663, 0.0, 0.42328662144868595])  # top centre
TABLE_POS, TABLE_HALF = (BORE_ORIGIN[0], 0.0, 0.37328662144868596), (0.1, 0.1, 0.01)
FACETS = 64
PEG_R, PEG_LENGTH, PEG_MOUNT = 0.01, 0.06, 0.135
BORE_DEPTH, CHAMFER, BORE_OUTER = 0.03, 0.001, 0.0249699
START_GAP, DESCENT, DEPTH_TARGET, DEPTH_PASS = 0.002, 0.022, 0.020, 0.019
OFFSET, TILT_DEG = 0.00025, 1.0
T_START, T_END, PASS_WINDOW = 0.3, 3.2, 0.25

K_T, K_R = 1500.0, 30.0
D_T, D_R = 80.0, 1.0
K_NULL, D_NULL = 5.0, 1.0
F_CAP, M_CAP = 20.0, 2.0
F_STOP, M_STOP = 40.0, 4.0
GUIDED = {"force_filter_s": 0.02, "lateral_admittance_m_Ns": 0.001,
          "lateral_speed_m_s": 0.001, "lateral_travel_m": 0.002,
          "upward_relief_m_N": 5e-5, "upward_relief_limit_m": 0.001,
          "angular_admittance_rad_Nms": 0.1, "angular_speed_rad_s": 0.02,
          "angular_limit_rad": np.deg2rad(3.0)}


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rotvec(v: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(v)
    if angle < 1e-15:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n <= limit else v * (limit / n)


def peg_vertices() -> np.ndarray:
    a = np.arange(FACETS) * 2 * np.pi / FACETS
    ring = np.column_stack([PEG_R * np.cos(a), PEG_R * np.sin(a)])
    return np.vstack([np.column_stack([ring, np.full(FACETS, z)])
                      for z in (-PEG_LENGTH / 2, PEG_LENGTH / 2)])


def bore_pieces(clearance: float) -> list[np.ndarray]:
    """Wedge k spans k * 5.625 deg +- 2.8125 deg, in the bore frame (top centre, z up)."""
    inner = PEG_R + clearance
    half = np.tan(np.pi / FACETS)
    profile = [(inner, -BORE_DEPTH), (inner, -CHAMFER), (inner + CHAMFER, 0.0),
               (BORE_OUTER, -BORE_DEPTH), (BORE_OUTER, 0.0)]
    local = np.array([(r, s * r * half, z) for r, z in profile for s in (-1, 1)])
    pieces = []
    for k in range(FACETS):
        c, s = np.cos(2 * np.pi * k / FACETS), np.sin(2 * np.pi * k / FACETS)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pieces.append(local @ rot.T)
    return pieces


def peg_mount(case: str) -> np.ndarray:
    """Peg orientation in the hand frame (hand Z points down, along the peg)."""
    return _rot_y(-TILT_DEG if case == "tilted" else 0.0)


def peg_tip_hand(case: str) -> np.ndarray:
    return np.array([0.0, 0.0, PEG_MOUNT]) + peg_mount(case) @ [0.0, 0.0, PEG_LENGTH / 2]


def _mjcf(dt: float, clearance: float, case: str) -> str:
    import mujoco

    tree = panda_tree("peg", SCENE_DIR)
    root = tree.getroot()
    root.find("option").set("timestep", repr(dt))
    asset, world = root.find("asset"), root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="box", pos=_fmt(TABLE_POS),
                  size=_fmt(TABLE_HALF), rgba="0.55 0.5 0.45 1", **contact_attrs("static"))
    for k, piece in enumerate(bore_pieces(clearance)):
        ET.SubElement(asset, "mesh", name=f"bore_{k:03d}", vertex=_fmt(np.ravel(piece)))
        ET.SubElement(world, "geom", name=f"bore_{k:03d}", type="mesh", mesh=f"bore_{k:03d}",
                      pos=_fmt(BORE_ORIGIN), rgba="0.6 0.62 0.66 1", **contact_attrs("static"))
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="hand_origin", size="0.005")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, peg_mount(case).ravel())
    peg = ET.SubElement(hand, "body", name="peg", pos=_fmt([0.0, 0.0, PEG_MOUNT]),
                        quat=_fmt(quat))
    m = PADDLE_MASS
    ixx = m * (3 * PEG_R**2 + PEG_LENGTH**2) / 12
    ET.SubElement(peg, "inertial", pos="0 0 0", mass=repr(m),
                  diaginertia=_fmt([ixx, ixx, m * PEG_R**2 / 2]))
    ET.SubElement(asset, "mesh", name="peg_mesh", vertex=_fmt(np.ravel(peg_vertices())))
    ET.SubElement(peg, "geom", name="peg_geom", type="mesh", mesh="peg_mesh",
                  rgba="0.85 0.55 0.2 1", **contact_attrs("paddle"))
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode")


class Impedance:
    """tau = J^T clip(K e - D v) + bias + null-space posture, sampled and held at 1 kHz.

    The reference is defined for the peg tip, so guided orientation corrections rotate
    the peg about its tip. `observe` reads the step's contacts on the peg, applies the
    overload stop and (if `guided`) updates the bounded admittance offsets.
    """

    def __init__(self, path, q0, tip0, r_tip, guided: dict | None, dt: float):
        import mujoco

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
                                    | mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.d = mujoco.MjData(self.m)
        self.hand = self.m.body("hand").id
        self.q0, self.tip0, self.r_tip = np.asarray(q0), np.asarray(tip0), np.asarray(r_tip)
        self.rot0 = np.diag([1.0, -1.0, -1.0])
        self.guided, self.dt = guided, dt
        self.jacp, self.jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        self.lateral, self.relief = np.zeros(2), 0.0
        self.tilt = np.zeros(2)
        self.f_filt, self.m_filt = np.zeros(3), np.zeros(3)
        self.x = None
        self.u = None
        self.peg = None

    def reference(self, t: float):
        s, sd, _ = tapes.quintic(np.clip((t - T_START) / (T_END - T_START), 0.0, 1.0))
        tip = self.tip0 + np.array([*self.lateral, self.relief - DESCENT * s])
        rot = _rotvec(np.array([*self.tilt, 0.0])) @ self.rot0
        vel = np.array([0.0, 0.0, -DESCENT * sd / (T_END - T_START)])
        if not T_START < t < T_END:
            vel[:] = 0.0
        return tip - rot @ self.r_tip, rot, vel

    def __call__(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        i = t / CTRL_DT
        k = int(round(i))
        if self.u is not None and abs(i - k) >= 1e-6:
            return self.u
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:], d.qvel[:] = q, qd
        mj.mj_forward(m, d)
        x, rot = d.xpos[self.hand].copy(), d.xmat[self.hand].reshape(3, 3)
        mj.mj_jacBody(m, d, self.jacp, self.jacr, self.hand)
        jac = np.vstack([self.jacp, self.jacr])
        v = jac @ qd
        x_ref, rot_ref, v_ref = self.reference(t)
        e_rot = 0.5 * np.cross(rot, rot_ref, axis=0).sum(axis=1)
        force = _clip_norm(K_T * (x_ref - x) + D_T * (v_ref - v[:3]), F_CAP)
        moment = _clip_norm(K_R * e_rot - D_R * v[3:], M_CAP)
        null = np.eye(7) - jac.T @ np.linalg.pinv(jac).T
        tau = (jac.T @ np.concatenate([force, moment]) + d.qfrc_bias
               + null @ (K_NULL * (self.q0 - q) - D_NULL * qd))
        self.x, self.tip = x, x + rot @ self.r_tip
        self.u = np.clip(tau, -LIMIT[:7], LIMIT[:7])
        return self.u

    def observe(self, t: float, contacts: dict, names: list[str]):
        if self.peg is None:
            self.peg = names.index("peg")
        on_peg = np.where(contacts["b"] == self.peg, 1.0, 0.0) - np.where(
            contacts["a"] == self.peg, 1.0, 0.0)
        f = contacts["force"] * on_peg[:, None]
        force = f.sum(axis=0)
        hand_moment = np.cross(contacts["pos"] - self.x, f).sum(axis=0)
        if np.linalg.norm(force) > F_STOP or np.linalg.norm(hand_moment) > M_STOP:
            return ("overload_stop", f"peg contact {np.linalg.norm(force):.1f} N / "
                    f"{np.linalg.norm(hand_moment):.2f} N m at t={t:.3f}s")
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


def insertion(dt: float = 0.001, clearance_mm: float = 0.10, case: str = "aligned",
              guided: bool = False) -> Scene:
    import mujoco

    clearance = clearance_mm * 1e-3
    mjcf = _mjcf(dt, clearance, case)
    r_tip = peg_tip_hand(case)
    tip0 = BORE_ORIGIN + [OFFSET if case == "offset" else 0.0, 0.0, START_GAP]
    rot0 = np.diag([1.0, -1.0, -1.0])
    params = {"task": "insertion", "dt": dt, "duration": DURATION, "clearance_mm": clearance_mm,
              "case": case, "guided": GUIDED if guided else None, "ctrl_dt": CTRL_DT,
              "bore_origin": BORE_ORIGIN.tolist(), "peg_tip_hand": r_tip.tolist(),
              "depth_target_m": DEPTH_TARGET, "depth_pass_m": DEPTH_PASS,
              "pass_window_s": PASS_WINDOW, "gains": {"k_t": K_T, "k_r": K_R, "d_t": D_T,
                                                      "d_r": D_R, "k_null": K_NULL,
                                                      "d_null": D_NULL},
              "caps": [F_CAP, M_CAP], "stops": [F_STOP, M_STOP], "friction": 0.5}
    scene = Scene(name="insertion", mjcf=mjcf, dt=dt, duration=DURATION,
                  bodies=[*PANDA_BODIES, "peg"], joints=ARM_JOINTS, params=params)
    path = scene.mjcf_path()
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    q0 = tapes.ik(m, d, "hand_origin", tip0 - rot0 @ r_tip, rot0, Q_READY)
    scene.joint_init = dict(zip(ARM_JOINTS, q0.tolist(), strict=True))
    scene.controller = Impedance(path, q0, tip0, r_tip, GUIDED if guided else None, dt)
    return scene
