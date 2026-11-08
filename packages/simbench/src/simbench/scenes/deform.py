"""Phase 6 deformable ball: MuJoCo flex compression control and Panda grasp.

The ball is the article's authored mesh (`make fetch`): 125 vertices on a spherified
5 x 5 x 5 grid (radius 30 mm), 384 tetrahedra and volume-lumped node masses (105 g),
with E = 20 kPa, nu = 0.3, elastic damping 2e-5 s and friction 0.5. Each vertex is a
body with three slide joints carrying its lumped mass; the flex is dim 3 with zero
radius, collides with pads and statics only, and runs at 50 us as in the article.

ball-compress  two plates on stiff slide-joint PD close 6 mm each (cubic smoothstep
               0.1-0.6 s), hold to 0.8 s, release by 1.3 s; gravity off (1.5 s)
ball-grasp     the Phase 3 pad Panda at the ready pose with the ball centred on its TCP:
               close 0.3-0.8 s with a per-finger effort cap, lift 80 mm 1.0-1.8 s, hold
               to 2.3 s, lower by 3.1 s, open 3.3-3.7 s (4.5 s)
"""

import xml.etree.ElementTree as ET
from functools import cache

import numpy as np

from simbench import tapes
from simbench.models import (
    ARM_JOINTS,
    FINGER_JOINTS,
    MASKS,
    PANDA_BODIES,
    Q_READY,
    _fmt,
    contact_attrs,
)
from simbench.reference import DEFORM, load_bundle
from simbench.scene import OPTION, SOLIMP, SOLREF, Scene
from simbench.scenes.panda import (
    CLOSED,
    CTRL_DT,
    KD,
    KP,
    LIMIT,
    OPEN,
    TapeController,
    _hold,
    _mjcf,
)

DT, RECORD_DT = 5e-5, 0.002
YOUNG, POISSON, DAMPING = 20000.0, 0.3, 2e-5
PLATE_HALF, PLATE_GAP0, SQUEEZE = (0.05, 0.05, 0.005), 0.0351, 0.006
PLATE_KP, PLATE_KD, PLATE_MASS = 1e5, 600.0, 1.0
BALL_CENTER = (0.3068905665922811, 0.0, 0.48738205230337767)
TABLE_POS, TABLE_HALF = (BALL_CENTER[0], 0.0, 0.4323820523033776), (0.15, 0.15, 0.025)
LIFT = 0.08
HOLD_WINDOW = (1.9, 2.2)


@cache
def ball_mesh() -> dict:
    trace = next(t for t in load_bundle(DEFORM["grasp-2N"])["traces"]
                 if t["backend"] == "mujoco_cpu")
    mesh = trace["scene"]["ball_mesh"]
    return {"vertices": np.array(mesh["vertices_m"]), "tets": np.array(mesh["tetrahedra"]),
            "faces": np.array(mesh["triangles"]), "mass": np.array(mesh["node_mass_kg"])}


def ball_bodies() -> list[str]:
    return [f"ball_{i}" for i in range(len(ball_mesh()["vertices"]))]


def _add_ball(root: ET.Element, center, young: float = YOUNG, poisson: float = POISSON,
              yield_strain: float = 0.0, solref: str = SOLREF) -> None:
    mesh = ball_mesh()
    world = root.find("worldbody")
    for name, v, mass in zip(ball_bodies(), mesh["vertices"], mesh["mass"], strict=True):
        body = ET.SubElement(world, "body", name=name, pos=_fmt(np.asarray(center) + v))
        ET.SubElement(body, "inertial", pos="0 0 0", mass=repr(float(mass)),
                      diaginertia="1e-9 1e-9 1e-9")
        for axis in ("1 0 0", "0 1 0", "0 0 1"):
            ET.SubElement(body, "joint", type="slide", axis=axis)
    contype, _ = MASKS["object"]
    flex = ET.SubElement(ET.SubElement(root, "deformable"), "flex", name="ball", dim="3",
                         radius="0", body=" ".join(ball_bodies()),
                         element=" ".join(str(int(i)) for i in mesh["tets"].ravel()),
                         rgba="0.9 0.45 0.55 1")
    # Pads and statics only: no flex self-collision, as in the article.
    ET.SubElement(flex, "contact", contype=str(contype), conaffinity=str(1 | 4), condim="3",
                  friction="0.5 0 0", solref=solref, solimp=SOLIMP, selfcollide="none")
    # `yield` (von Mises return mapping, MuJoCo >= 4.0) is only written when plastic, so
    # the elastic ball still compiles on 3.11.
    ET.SubElement(flex, "elasticity", young=repr(young), poisson=repr(poisson),
                  damping=repr(DAMPING),
                  **({"yield": repr(yield_strain)} if yield_strain else {}))


def legacy_constants(young: float = YOUNG, poisson: float = POISSON) -> tuple[float, float]:
    """MuJoCo 4.0 changelog: E', nu' that reproduce the pre-4.0 3D flex (doubled lambda)."""
    nu = 2 * poisson / (1 + 2 * poisson)
    return young * (1 + nu) / (1 + poisson), nu


def _params(task: str, duration: float, center, young: float = YOUNG,
            poisson: float = POISSON, yield_strain: float = 0.0) -> dict:
    mesh = ball_mesh()
    return {"task": task, "dt": DT, "duration": duration, "record_dt": RECORD_DT,
            "young": young, "poisson": poisson, "yield": yield_strain, "damping": DAMPING,
            "friction": 0.5,
            "ball_center": list(center), "tets": mesh["tets"].tolist(),
            "faces": mesh["faces"].tolist(), "node_mass": mesh["mass"].tolist()}


class PlatePD:
    """Prescribed plate gap on two slide joints (evaluated every step, no hold)."""

    def __call__(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        z, zd = squeeze(t)
        ref, vref = np.array([z, -z]), np.array([zd, -zd])
        return PLATE_KP * (ref - q) + PLATE_KD * (vref - qd)


def squeeze(t: float) -> tuple[float, float]:
    """Inward travel of each plate [m] and its rate."""
    def smooth(t0, t1):
        s = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        return 3 * s**2 - 2 * s**3, (6 * s - 6 * s**2) / (t1 - t0)
    (a, ad), (b, bd) = smooth(0.1, 0.6), smooth(0.8, 1.3)
    return SQUEEZE * (a - b), SQUEEZE * (ad - bd)


def compress(dt: float = DT, yield_strain: float = 0.0, legacy: bool = False) -> Scene:
    young, poisson = legacy_constants() if legacy else (YOUNG, POISSON)
    root = ET.fromstring(f'<mujoco model="ball_compress"><option timestep="{dt!r}" '
                         f'gravity="0 0 0" {OPTION}/><worldbody/></mujoco>')
    world = root.find("worldbody")
    for name, sign in (("plate_lo", -1), ("plate_hi", 1)):
        body = ET.SubElement(world, "body", name=name, pos=_fmt([0, 0, sign * PLATE_GAP0]))
        ET.SubElement(body, "joint", name=name, type="slide", axis="0 0 1")
        ET.SubElement(body, "inertial", pos="0 0 0", mass=repr(PLATE_MASS),
                      diaginertia="1e-3 1e-3 1e-3")
        ET.SubElement(body, "geom", name=f"{name}_geom", type="box", size=_fmt(PLATE_HALF),
                      rgba="0.55 0.5 0.45 0.6", **contact_attrs("static"))
    _add_ball(root, (0.0, 0.0, 0.0), young, poisson, yield_strain)
    ET.indent(root)
    params = _params("ball-compress", 1.5, (0.0, 0.0, 0.0), young, poisson, yield_strain) | {
        "dt": dt, "plate_gap0": PLATE_GAP0, "squeeze": SQUEEZE, "plate_half": PLATE_HALF,
        "plate_pd": [PLATE_KP, PLATE_KD], "legacy_constants": legacy}
    return Scene(name="ball-compress", mjcf=ET.tostring(root, encoding="unicode"), dt=dt,
                 duration=1.5, bodies=["plate_lo", "plate_hi", *ball_bodies()],
                 joints=["plate_lo", "plate_hi"], controller=PlatePD(), params=params,
                 record_dt=RECORD_DT)


def grasp_spec() -> dict:
    x, y, z = BALL_CENTER
    return {"name": "ball-grasp", "kind": "cartesian", "duration": 4.5, "ctrl_dt": CTRL_DT,
            "q_seed": Q_READY, "tcp_start": [x, y, z],
            "tcp_segments": [[1.0, 1.8, [x, y, z + LIFT], "joint"],
                             [2.3, 3.1, [x, y, z], "joint"]],
            "finger_start": [OPEN, OPEN],
            "finger_segments": [[0.3, 0.8, [CLOSED, CLOSED]], [3.3, 3.7, [OPEN, OPEN]]]}


def grasp(dt: float = DT, effort_N: float = 2.0, yield_strain: float = 0.0,
          contact_tc: float | None = None) -> Scene:
    """`contact_tc` [s] replaces the article's 10 ms pad-ball contact time constant."""
    spec = grasp_spec()
    tape = tapes.load(spec)
    tree = ET.fromstring(_mjcf("pads", dt))
    ET.SubElement(tree.find("worldbody"), "geom", name="table", type="box",
                  pos=_fmt(TABLE_POS), size=_fmt(TABLE_HALF), rgba="0.55 0.5 0.45 1",
                  **contact_attrs("static"))
    solref = SOLREF if contact_tc is None else f"{contact_tc!r} 1"
    _add_ball(tree, BALL_CENTER, yield_strain=yield_strain, solref=solref)
    ET.indent(tree)
    limit = LIMIT.copy()
    limit[7:] = effort_N
    params = _params("ball-grasp", spec["duration"], BALL_CENTER,
                     yield_strain=yield_strain) | {
        "dt": dt, "effort_N": effort_N, "tape_spec": spec, "tape": tapes.spec_sha(spec),
        "kp": KP.tolist(), "kd": KD.tolist(), "limit": limit.tolist(), "ctrl_dt": CTRL_DT,
        "table_top": TABLE_POS[2] + TABLE_HALF[2], "hold_window": HOLD_WINDOW, "lift": LIFT}
    if contact_tc is not None:
        params["contact_solref"] = solref
    return Scene(name="ball-grasp", mjcf=ET.tostring(tree, encoding="unicode"), dt=dt,
                 duration=spec["duration"], bodies=[*PANDA_BODIES, *ball_bodies()],
                 joints=ARM_JOINTS + FINGER_JOINTS, joint_init=_hold(tape),
                 controller=TapeController(tape, 9, limit=limit), params=params,
                 record_dt=RECORD_DT)


SHEET = {"nx": 13, "ny": 3, "spacing": 0.01, "thickness": 0.001, "young": 2e9,
         "poisson": 0.38, "density": 1380.0, "damping": 1e-4, "press": 0.015,
         "press_x": 0.07, "pusher_radius": 0.008, "pd": (4e5, 1200.0), "duration": 2.0,
         "hold_t": 0.7}


def sheet_bodies() -> list[str]:
    return [f"sheet_{i}" for i in range(SHEET["nx"] * SHEET["ny"])]


class PusherPD:
    """Prescribed pusher descent: down `press` past first touch 0.1-0.6 s, back 0.8-1.3 s."""

    def __call__(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        kp, kd = SHEET["pd"]
        a, b = (_smooth(t, 0.1, 0.6), _smooth(t, 0.8, 1.3))
        ref = -(SHEET["press"] + 0.001) * (a - b)
        return np.array([kp * (ref - q[0]) - kd * qd[0]])


def _smooth(t: float, t0: float, t1: float) -> float:
    s = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return 3 * s**2 - 2 * s**3


def sheet(dt: float = 1e-4, yield_bend: float = 0.0) -> Scene:
    """PET-like 120 x 20 x 1 mm cantilever strip pressed by a sphere and released.

    The first two vertex columns are welded to the world (bending acts across interior
    edges, so one clamped column would leave a free hinge at the root). The shell uses
    `elastic2d="both"` under the `discrete` integrator. `yield_bend > 0` writes
    `yield="0 yield_bend"` (MuJoCo >= 4.0 shell bending plasticity).
    """
    p = SHEET
    nx, ny, sp, th = p["nx"], p["ny"], p["spacing"], p["thickness"]
    vmass = p["density"] * th * (nx - 1) * (ny - 1) * sp**2 / (nx * ny)
    root = ET.fromstring(
        f'<mujoco model="sheet"><option timestep="{dt!r}" gravity="0 0 0" '
        f'integrator="discrete" solver="Newton" iterations="100" tolerance="1e-10" '
        f'cone="elliptic"/><worldbody/></mujoco>')
    world = root.find("worldbody")
    r = p["pusher_radius"]
    pusher = ET.SubElement(world, "body", name="pusher",
                           pos=_fmt([p["press_x"], 0.0, r + th / 2 + 0.001]))
    ET.SubElement(pusher, "joint", name="push", type="slide", axis="0 0 1")
    ET.SubElement(pusher, "inertial", pos="0 0 0", mass="1.0", diaginertia="1e-3 1e-3 1e-3")
    ET.SubElement(pusher, "geom", name="pusher_geom", type="sphere", size=repr(r),
                  contype="2", conaffinity="1", friction="0.3 0 0", rgba="0.55 0.5 0.45 1")
    names = sheet_bodies()
    for k, name in enumerate(names):
        i, j = k % nx, k // nx
        body = ET.SubElement(world, "body", name=name,
                             pos=_fmt([i * sp, (j - (ny - 1) / 2) * sp, 0.0]))
        ET.SubElement(body, "inertial", pos="0 0 0", mass=repr(vmass),
                      diaginertia="1e-10 1e-10 1e-10")
        if i > 1:
            for axis in ("1 0 0", "0 1 0", "0 0 1"):
                ET.SubElement(body, "joint", type="slide", axis=axis)
    tri = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
            tri += [[a, b, d], [a, d, c]]
    flex = ET.SubElement(ET.SubElement(root, "deformable"), "flex", name="sheet", dim="2",
                         radius=repr(th / 2), body=" ".join(names),
                         element=" ".join(str(v) for f in tri for v in f),
                         rgba="0.35 0.6 0.9 1")
    ET.SubElement(flex, "contact", contype="1", conaffinity="2", condim="3",
                  friction="0.3 0 0", solref="0.002 1", selfcollide="none")
    ET.SubElement(flex, "elasticity", young=repr(p["young"]), poisson=repr(p["poisson"]),
                  thickness=repr(th), damping=repr(p["damping"]), elastic2d="both",
                  **({"yield": f"0 {yield_bend!r}"} if yield_bend else {}))
    ET.indent(root)
    tip = [f"sheet_{j * nx + nx - 1}" for j in range(ny)]
    params = {"task": "sheet", "dt": dt, **p, "yield_bend": yield_bend, "faces": tri,
              "tip_bodies": tip, "flex_prefix": "sheet_", "record_dt": RECORD_DT}
    return Scene(name="sheet", mjcf=ET.tostring(root, encoding="unicode"), dt=dt,
                 duration=p["duration"], bodies=["pusher", *names], joints=["push"],
                 controller=PusherPD(), params=params, record_dt=RECORD_DT)


def tape_specs() -> list[dict]:
    return [grasp_spec()]
