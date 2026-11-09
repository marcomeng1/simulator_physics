"""Dual-Franka scripted fold of a cotton-ish napkin (short horizon).

Both arms pinch the near-edge midpoints with high-friction pads, lift, fold one half
onto the other, press and release (~3 s). MuJoCo routes use a 2D flex grid; Newton VBD
and Genesis PBD Cloth rebuild the continuum from ``params`` (they do not read MJCF flex).

Cloth is 0.4×0.4 m (not 1×1): a metre square put the pinch edges outside a reliable
Franka frictional grasp and made the fold travel awkward.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np

from simbench import tapes
from simbench.models import (
    ARM_JOINTS,
    FINGER_JOINTS,
    PANDA_BODIES,
    Q_READY,
    _fmt,
    contact_attrs,
    panda_tree,
)
from simbench.scene import SCENE_DIR, SOLIMP, Scene
from simbench.scenes.panda import CTRL_DT, KD, KP, LIMIT, OPEN

DURATION = 3.0
DT = 1e-4
RECORD_DT = 0.01
# 11×11 over 0.4 m (spacing ≈ 40 mm).
CLOTH_N = 11
CLOTH_SIZE = 0.40
# Thicker / heavier napkin: the 3 mm @ 0.15 kg/m² card looked massless and rang.
CLOTH_THICK = 0.008
# Contact radius ≥ half-thickness so pads catch the sheet.
CLOTH_RADIUS = 0.010
# Felt-ish towel (shared intent; engines approximate differently). Gravity only — no wind.
CLOTH = {
    "nx": CLOTH_N, "ny": CLOTH_N, "size_m": CLOTH_SIZE, "thickness_m": CLOTH_THICK,
    "young": 5e4, "poisson": 0.3, "areal_density": 0.80, "damping": 0.05, "friction": 1.5,
}
PAD_MU = 2.0
CLOSED = 0.003
# Table / cloth layout (world metres).
TABLE_POS = (0.50, 0.0, -0.03)
TABLE_HALF = (0.45, 0.45, 0.03)
CLOTH_Z = CLOTH_RADIUS + 0.001
CLOTH_CENTER = np.array([0.50, 0.0, CLOTH_Z])
HALF = CLOTH_SIZE / 2
# Near-edge midpoints (fold along +y → −y).
PINCH_L = np.array([0.50, HALF, CLOTH_Z])
PINCH_R = np.array([0.50, -HALF, CLOTH_Z])
# Bases just outside the napkin so both pinches are easy reaches.
BASE_L = np.array([0.12, 0.32, 0.0])
BASE_R = np.array([0.12, -0.32, 0.0])
LIFT_Z = 0.16
# Left edge lands on the right half.
FOLD_Y = -0.06
# Pass gates (see metrics).
COVERAGE_PASS = 0.80
SLIP_PASS_MM = 25.0

PREFIXES = ("L_", "R_")


def cloth_bodies() -> list[str]:
    return [f"cloth_{i}" for i in range(CLOTH["nx"] * CLOTH["ny"])]


def cloth_faces() -> list[list[int]]:
    nx, ny = CLOTH["nx"], CLOTH["ny"]
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
            faces += [[a, b, d], [a, d, c]]
    return faces


def cloth_rest_positions() -> np.ndarray:
    """Vertex rest positions: row-major in x then y, cloth centered at CLOTH_CENTER."""
    nx, ny, s = CLOTH["nx"], CLOTH["ny"], CLOTH["size_m"]
    xs = np.linspace(-s / 2, s / 2, nx) + CLOTH_CENTER[0]
    ys = np.linspace(-s / 2, s / 2, ny) + CLOTH_CENTER[1]
    verts = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            verts.append([x, y, CLOTH_CENTER[2]])
    return np.asarray(verts, float)


def pinch_vertex_ids() -> tuple[list[int], list[int]]:
    """Cloth vertex indices nearest each pinch (3×3 patch for slip metric)."""
    rest = cloth_rest_positions()
    left, right = [], []
    for target, out in ((PINCH_L, left), (PINCH_R, right)):
        d = np.linalg.norm(rest - target, axis=1)
        out.extend(np.argsort(d)[:9].tolist())
    return left, right


def _prefix_names(el: ET.Element, prefix: str) -> None:
    """Prefix body/joint/site/geom names; leave mesh references shared."""
    tag = el.tag
    if tag in ("body", "joint", "site", "geom", "camera", "light") and el.get("name"):
        el.set("name", prefix + el.get("name"))
    for child in el:
        _prefix_names(child, prefix)


def _set_pad_friction(tree: ET.ElementTree, mu: float) -> None:
    for geom in tree.findall(".//geom"):
        name = geom.get("name") or ""
        if name.endswith("_pad") or name.endswith("pad"):
            geom.set("friction", f"{mu} 0.05 0.001")


def _extract_link0(tree: ET.ElementTree) -> ET.Element:
    world = tree.getroot().find("worldbody")
    for child in list(world):
        if child.tag == "body" and (child.get("name") or "").endswith("link0"):
            world.remove(child)
            return child
        # Menagerie: link0 is direct child named link0
        if child.tag == "body" and child.get("name") == "link0":
            world.remove(child)
            return child
    raise RuntimeError("panda link0 not found")


def _dual_panda_cloth_mjcf(dt: float) -> str:
    """One MJCF: shared mesh assets, two prefixed Pandas, table, 2D flex cloth."""
    left_tree = panda_tree("pads", SCENE_DIR)
    _set_pad_friction(left_tree, PAD_MU)
    right_tree = panda_tree("pads", SCENE_DIR)
    _set_pad_friction(right_tree, PAD_MU)

    root = left_tree.getroot()
    root.set("model", "cloth_fold")
    opt = root.find("option")
    opt.set("timestep", repr(dt))
    opt.set("gravity", "0 0 -9.81")
    # 2D flex elasticity requires discrete (MuJoCo 4); match the sheet scene.
    opt.set("integrator", "discrete")

    world = root.find("worldbody")
    # Drop Menagerie lights / extras cluttering dual scene — keep geoms on robots only.
    for child in list(world):
        if child.tag != "body":
            world.remove(child)

    link0_l = _extract_link0(left_tree)
    link0_r = right_tree.getroot().find("worldbody").find("body[@name='link0']")
    if link0_r is None:
        raise RuntimeError("right link0 missing")
    link0_r = copy.deepcopy(link0_r)

    _prefix_names(link0_l, "L_")
    _prefix_names(link0_r, "R_")
    # Prefix was applied to name="L_link0"; set base pose.
    link0_l.set("pos", _fmt(BASE_L))
    link0_r.set("pos", _fmt(BASE_R))
    world.append(link0_l)
    world.append(link0_r)

    ET.SubElement(world, "geom", name="table", type="box", pos=_fmt(TABLE_POS),
                  size=_fmt(TABLE_HALF), rgba="0.55 0.5 0.45 1", **contact_attrs("static"))

    # Cloth vertices as slide-joint bodies (sheet pattern).
    nx, ny = CLOTH["nx"], CLOTH["ny"]
    rest = cloth_rest_positions()
    vmass = CLOTH["areal_density"] * CLOTH_SIZE**2 / (nx * ny)
    names = cloth_bodies()
    for name, p in zip(names, rest, strict=True):
        body = ET.SubElement(world, "body", name=name, pos=_fmt(p))
        # Non-tiny inertia (flex forbids joint damping — use elasticity damping instead).
        inert = max(vmass * CLOTH_RADIUS ** 2 / 6.0, 1e-7)
        ET.SubElement(body, "inertial", pos="0 0 0", mass=repr(float(vmass)),
                      diaginertia=_fmt((inert, inert, inert)))
        for axis in ("1 0 0", "0 1 0", "0 0 1"):
            ET.SubElement(body, "joint", type="slide", axis=axis)

    faces = cloth_faces()
    flex = ET.SubElement(ET.SubElement(root, "deformable"), "flex", name="cloth", dim="2",
                         radius=repr(CLOTH_RADIUS), body=" ".join(names),
                         element=" ".join(str(v) for f in faces for v in f),
                         rgba="0.85 0.82 0.72 1")
    # Pads (bit 1) and statics (bit 4); no cloth self-collide for speed/stability.
    ET.SubElement(flex, "contact", contype="2", conaffinity=str(1 | 4), condim="3",
                  friction=f"{CLOTH['friction']} 0.1 0.01", solref="0.004 1", solimp=SOLIMP,
                  selfcollide="none")
    ET.SubElement(flex, "elasticity", young=repr(CLOTH["young"]), poisson=repr(CLOTH["poisson"]),
                  thickness=repr(CLOTH_THICK), damping=repr(CLOTH["damping"]),
                  elastic2d="both")

    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def _arm_joints(prefix: str) -> list[str]:
    return [prefix + j for j in ARM_JOINTS + FINGER_JOINTS]


def _smoothstep(t: float, t0: float, t1: float) -> float:
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    s = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return 3 * s * s - 2 * s * s * s


def _tcp_path_left(t: float) -> np.ndarray:
    """Left arm: press into edge → close → slow lift → fold onto the right half → press."""
    p0 = PINCH_L + np.array([0.0, 0.025, 0.05])
    p_press_in = PINCH_L + np.array([0.0, 0.0, 0.003])
    p_lift = np.array([PINCH_L[0], PINCH_L[1], LIFT_Z])
    p_over = np.array([PINCH_L[0], FOLD_Y, LIFT_Z])
    p_down = np.array([PINCH_L[0], FOLD_Y, CLOTH_Z + 0.02])
    if t < 0.35:
        a = _smoothstep(t, 0.0, 0.35)
        return (1 - a) * p0 + a * p_press_in
    if t < 0.70:
        return p_press_in  # dwell while fingers close and friction settles
    if t < 2.05:
        if t < 1.45:
            a = _smoothstep(t, 0.70, 1.45)
            return (1 - a) * p_press_in + a * p_lift
        a = _smoothstep(t, 1.45, 2.05)
        return (1 - a) * p_lift + a * p_over
    a = _smoothstep(t, 2.05, 2.75)
    return (1 - a) * p_over + a * p_down


def _tcp_path_right(t: float) -> np.ndarray:
    """Right arm: press → close → slow lift as fold anchor → settle."""
    p0 = PINCH_R + np.array([0.0, -0.025, 0.05])
    p_press_in = PINCH_R + np.array([0.0, 0.0, 0.003])
    p_hold = np.array([PINCH_R[0], PINCH_R[1], LIFT_Z * 0.55])
    p_settle = PINCH_R + np.array([0.0, 0.0, CLOTH_Z + 0.012])
    if t < 0.35:
        a = _smoothstep(t, 0.0, 0.35)
        return (1 - a) * p0 + a * p_press_in
    if t < 0.70:
        return p_press_in
    if t < 2.05:
        a = _smoothstep(t, 0.70, 1.40)
        return (1 - a) * p_press_in + a * p_hold
    a = _smoothstep(t, 2.05, 2.75)
    return (1 - a) * p_hold + a * p_settle


def _finger_open(t: float) -> float:
    # Close only after pads are in the sheet; stay closed through the fold.
    if t < 0.35:
        return OPEN
    if t < 0.65:
        a = _smoothstep(t, 0.35, 0.65)
        return (1 - a) * OPEN + a * CLOSED
    if t < 2.6:
        return CLOSED
    a = _smoothstep(t, 2.6, 2.95)
    return (1 - a) * CLOSED + a * OPEN


def _ik_prefixed(m, d, prefix: str, site: str, target_pos, target_rot, q0,
                 q_rest=Q_READY, tol=1e-9, damping=1e-6, max_step=0.2, iters=400):
    """Damped least-squares IK for one prefixed Panda arm (seven hinge joints)."""
    import mujoco

    sid = m.site(site).id
    names = [prefix + j for j in ARM_JOINTS]
    qadr = np.array([m.joint(j).qposadr[0] for j in names])
    dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in names])
    lo, hi = m.jnt_range[[m.joint(j).id for j in names]].T
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    q = np.array(q0, dtype=float)
    polish = False
    err = np.zeros(6)
    for _ in range(iters):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rot = d.site_xmat[sid].reshape(3, 3)
        err = np.concatenate([target_pos - d.site_xpos[sid],
                              0.5 * np.cross(rot, target_rot, axis=0).sum(axis=1)])
        if np.linalg.norm(err) < tol:
            return q
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        jac = np.vstack([jacp, jacr])[:, dadr]
        pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping * np.eye(6))
        dq = pinv @ err
        polish = polish or np.linalg.norm(err) < 1e-8
        if not polish:
            null = np.eye(7) - np.linalg.pinv(jac) @ jac
            dq += null @ (0.5 * (np.asarray(q_rest) - q[qadr]))
        dq *= min(1.0, max_step / (np.abs(dq).max() + 1e-12))
        q[qadr] = np.clip(q[qadr] + dq, lo, hi)
    # Soft accept: scripted fold tolerates ~1 mm / few degrees residual.
    if np.linalg.norm(err) < 5e-3:
        return q
    raise RuntimeError(f"IK did not converge ({prefix}): |err|={np.linalg.norm(err):.3e}")


def _build_dual_tape(path: str) -> dict:
    """IK both TCPs at coarse knots, interpolate joints at 1 kHz, then ID feedforward."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    knot_dt = 0.05
    knots = np.arange(0.0, DURATION + 1e-12, knot_dt)
    t = np.arange(0.0, DURATION + 1e-12, CTRL_DT)
    q_knots = np.zeros((len(knots), m.nq))
    q = np.zeros(m.nq)
    for pref in PREFIXES:
        for jname, q0 in zip(ARM_JOINTS, Q_READY, strict=True):
            q[m.joint(pref + jname).qposadr[0]] = q0
        for jname in FINGER_JOINTS:
            q[m.joint(pref + jname).qposadr[0]] = OPEN
    rot_down = np.diag([1.0, -1.0, -1.0])
    for k, tk in enumerate(knots):
        for pref, tcp_fn in (("L_", _tcp_path_left), ("R_", _tcp_path_right)):
            q = _ik_prefixed(m, d, pref, pref + "tcp", tcp_fn(tk), rot_down, q)
            fing = _finger_open(tk)
            for jname in FINGER_JOINTS:
                q[m.joint(pref + jname).qposadr[0]] = fing
        q_knots[k] = q

    qref = np.zeros((len(t), m.nq))
    for j in range(m.nq):
        qref[:, j] = np.interp(t, knots, q_knots[:, j])
    qdref = np.gradient(qref, CTRL_DT, axis=0, edge_order=2)
    qddref = np.gradient(qdref, CTRL_DT, axis=0, edge_order=2)
    ff = np.zeros_like(qref)
    # ID every 5 ms is enough; hold between samples.
    id_stride = max(1, int(round(0.005 / CTRL_DT)))
    for k in range(0, len(t), id_stride):
        d.qpos[:] = qref[k]
        d.qvel[:] = 0
        d.qacc[:] = 0
        for pref in PREFIXES:
            for jname in ARM_JOINTS + FINGER_JOINTS:
                j = m.joint(pref + jname)
                d.qacc[j.dofadr[0]] = qddref[k, j.qposadr[0]]
                d.qvel[j.dofadr[0]] = qdref[k, j.qposadr[0]]
        mujoco.mj_inverse(m, d)
        for pref in PREFIXES:
            for jname in ARM_JOINTS + FINGER_JOINTS:
                j = m.joint(pref + jname)
                ff[k, j.qposadr[0]] = d.qfrc_inverse[j.dofadr[0]]
        end = min(k + id_stride, len(t))
        ff[k:end] = ff[k]
    return {"t": t, "qref": qref, "qdref": qdref, "qddref": qddref, "ff": ff}


class DualTapeController:
    """Open-loop dual-arm joint tape + closed-finger cloth pinch attach.

    All engines follow the same kinematic ``qref`` (Newton already did; MuJoCo PD and
    Genesis torque tracking rang against the napkin and looked like a zero-inertia
    explode). Cloth continuum + grasp attach are what differ across routes.
    """

    # Hold through finger-open so engines don't yank-release into a VBD blow-up.
    ATTACH_T0, ATTACH_T1 = 0.65, 2.95
    # Few welded verts: enough to lift, not enough to shred the sheet.
    ATTACH_N = 3

    def __init__(self, tape: dict, joint_qadr: np.ndarray, joint_names: list[str],
                 pinch_left: list[int], pinch_right: list[int], rest: np.ndarray):
        self.tape = tape
        self.joint_qadr = joint_qadr
        self.names = joint_names
        self.qref = np.stack([tape["qref"][:, adr] for adr in joint_qadr], axis=1)
        self.qdref = np.stack([tape["qdref"][:, adr] for adr in joint_qadr], axis=1)
        # Full patch for slip metric; short list for kinematic weld.
        self.pinch = {"L_": list(pinch_left), "R_": list(pinch_right)}
        self.attach_ids = {"L_": list(pinch_left)[: self.ATTACH_N],
                           "R_": list(pinch_right)[: self.ATTACH_N]}
        self.rest = np.asarray(rest, float)
        self._local = None  # prefix -> (n_pinch, 3) hand-local offsets
        self._zero = np.zeros(len(joint_names))

    def __call__(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        # Kinematic tape: adapters that honor ``prepare``/``qref`` need no torques.
        return self._zero

    def _tape_k(self, t: float) -> int:
        return min(max(int(round(t / CTRL_DT)), 0), len(self.qref) - 1)

    def attaching(self, t: float) -> bool:
        return self.ATTACH_T0 - 1e-9 <= t <= self.ATTACH_T1 + 1e-9

    def prepare(self, t: float, sim) -> None:
        """Write arm/finger qpos from the tape before the physics step (MuJoCo routes)."""
        self._drive_mujoco(t, sim)

    def post_step(self, t: float, sim) -> None:
        """Re-assert kinematic arms, then weld pinch flex verts while fingers are closed."""
        self._drive_mujoco(t, sim)
        if not self.attaching(t):
            self._local = None
            return
        m = getattr(sim, "mjm", None)
        if m is None and hasattr(sim, "m") and hasattr(sim.m, "body"):
            m = sim.m
        if m is None or not hasattr(sim, "d"):
            return

        import mujoco

        d = sim.d
        warp = hasattr(d.qpos, "numpy")
        if warp:
            import mujoco_warp as mjw
            import warp as wp

            mjw.kinematics(sim.m, d)
            xpos = d.xpos.numpy()[0]
            xmat = d.xmat.numpy()[0]
            qpos = d.qpos.numpy()[0].copy()
            qvel = d.qvel.numpy()[0].copy()
        else:
            xpos, xmat = d.xpos, d.xmat
            qpos, qvel = d.qpos, d.qvel

        if self._local is None:
            # Capture from *current* cloth positions to avoid a rest→hand teleport yank.
            self._local = {}
            for pref, ids in self.attach_ids.items():
                hand = m.body(pref + "hand").id
                R = np.asarray(xmat[hand]).reshape(3, 3)
                p = np.asarray(xpos[hand])
                locs = []
                for vid in ids:
                    bid = m.body(f"cloth_{vid}").id
                    # World pos of flex vertex = body xpos (slide joints from rest).
                    locs.append(R.T @ (np.asarray(xpos[bid]) - p))
                self._local[pref] = np.asarray(locs, float)
        for pref, ids in self.attach_ids.items():
            hand = m.body(pref + "hand").id
            R = np.asarray(xmat[hand]).reshape(3, 3)
            p = np.asarray(xpos[hand])
            for i, vid in enumerate(ids):
                world = p + R @ self._local[pref][i]
                bid = m.body(f"cloth_{vid}").id
                rest = self.rest[vid]
                for j in range(m.njnt):
                    if m.jnt_bodyid[j] != bid:
                        continue
                    axis = m.jnt_axis[j]
                    qpos[m.jnt_qposadr[j]] = float((world - rest) @ axis)
                    qvel[m.jnt_dofadr[j]] = 0.0

        if warp:
            wp.copy(d.qpos, wp.array(qpos.reshape(1, -1).astype(np.float32), dtype=d.qpos.dtype))
            wp.copy(d.qvel, wp.array(qvel.reshape(1, -1).astype(np.float32), dtype=d.qvel.dtype))
            mjw.kinematics(sim.m, d)
        else:
            mujoco.mj_forward(m, d)

    def _drive_mujoco(self, t: float, sim) -> None:
        """Set arm/finger joints to the open-loop tape (CPU or Warp MjData)."""
        m = getattr(sim, "mjm", None)
        if m is None and hasattr(sim, "m") and hasattr(sim.m, "body"):
            m = sim.m
        if m is None or not hasattr(sim, "d"):
            return
        import mujoco

        k = self._tape_k(t)
        d = sim.d
        warp = hasattr(d.qpos, "numpy")
        if warp:
            import mujoco_warp as mjw
            import warp as wp

            qpos = d.qpos.numpy()[0].copy()
            qvel = d.qvel.numpy()[0].copy()
        else:
            qpos, qvel = d.qpos, d.qvel
        for name, q, qd in zip(self.names, self.qref[k], self.qdref[k], strict=True):
            j = m.joint(name)
            qpos[j.qposadr[0]] = float(q)
            qvel[j.dofadr[0]] = float(qd)
        if warp:
            wp.copy(d.qpos, wp.array(np.asarray(qpos).reshape(1, -1).astype(np.float32),
                                     dtype=d.qpos.dtype))
            wp.copy(d.qvel, wp.array(np.asarray(qvel).reshape(1, -1).astype(np.float32),
                                     dtype=d.qvel.dtype))
            mjw.kinematics(sim.m, d)
        else:
            mujoco.mj_forward(m, d)


def cloth_fold(dt: float = DT) -> Scene:
    mjcf = _dual_panda_cloth_mjcf(dt)
    scene_bodies = (
        [f"L_{b}" for b in PANDA_BODIES] + [f"R_{b}" for b in PANDA_BODIES] + cloth_bodies()
    )
    joints = _arm_joints("L_") + _arm_joints("R_")
    left_pinch, right_pinch = pinch_vertex_ids()
    faces = cloth_faces()
    rest = cloth_rest_positions()
    params = {
        "task": "cloth-fold", "dt": dt, "duration": DURATION, "ctrl_dt": CTRL_DT,
        "cloth": dict(CLOTH), "faces": faces, "flex_prefix": "cloth_",
        "cloth_radius": CLOTH_RADIUS,
        "cloth_rest": rest.tolist(),
        "pinch_left": left_pinch, "pinch_right": right_pinch,
        "pinch_targets": {"L": PINCH_L.tolist(), "R": PINCH_R.tolist()},
        "base_L": BASE_L.tolist(), "base_R": BASE_R.tolist(),
        "fold_axis": "y", "coverage_pass": COVERAGE_PASS, "slip_pass_mm": SLIP_PASS_MM,
        "pad_mu": PAD_MU, "record_dt": RECORD_DT,
        "attach_window": [DualTapeController.ATTACH_T0, DualTapeController.ATTACH_T1],
        "robot_drive": "kinematic-tape",
        # Shown on the replay page so fair-comparison / cloth stacks are explicit.
        "method": {
            "controller": (
                "Open-loop dual-arm joint tape (DualTapeController): every engine writes the "
                "same qref/qdref kinematically each step — no torque PD. That keeps the robot "
                "trajectory matched so the cloth continuum is what differs."
            ),
            "grasp": (
                f"Closed-finger pinch attach on {DualTapeController.ATTACH_N} edge vertices per "
                f"hand during t∈[{DualTapeController.ATTACH_T0:g}, {DualTapeController.ATTACH_T1:g}] s "
                "(friction alone could not lift the napkin reliably)."
            ),
            "cloth_card": dict(CLOTH),
            "engines": {
                "mujoco4": "MuJoCo 2D flex elasticity (dim=2, discrete integrator)",
                "mjwarp4": "MuJoCo-Warp 2D flex elasticity (same MJCF flex as mujoco4)",
                "newton": "Newton SolverVBD cloth grid + CollisionPipeline (kinematic joint_q)",
                "genesis": "Genesis materials.PBD.Cloth (kinematic tape; pad↔cloth collision off; "
                           "PBD floor bound; pinch particle attach)",
            },
        },
    }
    scene = Scene(name="cloth-fold", mjcf=mjcf, dt=dt, duration=DURATION,
                  bodies=scene_bodies, joints=joints, params=params, record_dt=RECORD_DT)
    path = scene.mjcf_path()
    tape = _build_dual_tape(str(path))
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    qadr = np.array([m.joint(n).qposadr[0] for n in joints], dtype=int)
    scene.joint_init = {n: float(tape["qref"][0, m.joint(n).qposadr[0]]) for n in joints}
    scene.controller = DualTapeController(
        tape, qadr, joints, left_pinch, right_pinch, rest)
    scene.params["tape_sha"] = tapes.spec_sha(
        {"name": "cloth-fold", "kind": "dual-cartesian", "duration": DURATION})
    return scene
