"""Phase 1 and 2 scenes: sliding block, drop and settle, hinged link, collision chain.

The MJCF text reproduces the article's recorded `model_xml` (reference/article/replay)
so the MuJoCo 3.11 route can be checked against the published traces frame by frame.
"""

import numpy as np

from simbench.scene import OPTION, SOLIMP, SOLREF, FreeInit, Scene

CUBE_SIDE = 0.04
CUBE_MASS = 0.1
CUBE_INERTIA = CUBE_MASS * CUBE_SIDE**2 / 6


def _contact(friction: float) -> str:
    return f'friction="{friction} 0 0" condim="3" solref="{SOLREF}" solimp="{SOLIMP}"'


def _cube_mjcf(dt: float, pos: tuple[float, float, float], friction: float,
               floor: bool = True, gravity: float = 9.81) -> str:
    inertia = f"{CUBE_INERTIA!r} " * 3
    floor_geom = (f'<geom name="floor" type="plane" size="2 2 .05" {_contact(friction)} />'
                  if floor else "")
    return (
        f'<mujoco model="simcompare_primitives"><option timestep="{dt}" '
        f'gravity="0 0 -{gravity}" {OPTION} /><worldbody>{floor_geom}'
        f'<body name="cube" pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint name="cube_free" />'
        f'<inertial pos="0 0 0" mass="{CUBE_MASS}" diaginertia="{inertia.strip()}" />'
        f'<geom name="cube_geom" type="box" size="{CUBE_SIDE / 2} {CUBE_SIDE / 2} '
        f'{CUBE_SIDE / 2}" {_contact(friction)} /></body></worldbody></mujoco>'
    )


def slide(dt: float = 0.002, friction: float = 0.4, duration: float = 2.0) -> Scene:
    """100 g, 4 cm cube sliding at 1 m/s on a plane until it stops."""
    pos = (-0.3, 0.0, 0.02)
    return Scene(
        name="slide" if friction > 0 else "slide-frictionless",
        mjcf=_cube_mjcf(dt, pos, friction), dt=dt, duration=duration,
        bodies=["cube"], free_init={"cube": FreeInit(linear=(1.0, 0.0, 0.0))},
        params={"task": "slide", "dt": dt, "duration": duration, "mass": CUBE_MASS,
                "side": CUBE_SIDE, "friction": friction, "position": pos,
                "velocity": [1.0, 0.0, 0.0]},
    )


def drop(dt: float = 0.001, duration: float = 1.0) -> Scene:
    """The same cube released from rest with its COM at 35 cm."""
    pos = (-0.3, 0.0, 0.35)
    return Scene(
        name="drop", mjcf=_cube_mjcf(dt, pos, 0.4), dt=dt, duration=duration,
        bodies=["cube"], free_init={"cube": FreeInit()},
        params={"task": "drop", "dt": dt, "duration": duration, "mass": CUBE_MASS,
                "side": CUBE_SIDE, "friction": 0.4, "position": pos},
    )


HINGE = {"mass": 1.0, "length": 0.3, "width": 0.04, "pivot": (0.0, 0.0, 0.45), "q0": 0.5,
         "torque": 0.2, "pulse": (0.1, 0.2), "gravity": 9.81}


def hinge_inertia() -> tuple[float, float, float]:
    m, length, w = HINGE["mass"], HINGE["length"], HINGE["width"]
    side = m * (length**2 + w**2) / 12
    return side, side, m * (2 * w**2) / 12


def hinge(dt: float = 0.001, duration: float = 1.0) -> Scene:
    """Uniform 1 kg, 30 cm link on a frictionless hinge with a 0.2 Nm torque pulse."""
    ixx, iyy, izz = hinge_inertia()
    half = HINGE["length"] / 2
    px, py, pz = HINGE["pivot"]
    mjcf = (
        f'<mujoco model="hinge"><option timestep="{dt}" gravity="0 0 -{HINGE["gravity"]}" '
        f'{OPTION} /><worldbody><body name="link" pos="{px} {py} {pz}">'
        f'<joint name="hinge" type="hinge" axis="0 1 0" damping="0" armature="0" '
        f'frictionloss="0" />'
        f'<inertial pos="0 0 {-half}" mass="{HINGE["mass"]}" '
        f'diaginertia="{ixx!r} {iyy!r} {izz!r}" />'
        f'<geom name="link_geom" type="box" pos="0 0 {-half}" '
        f'size="{HINGE["width"] / 2} {HINGE["width"] / 2} {half}" contype="0" '
        f'conaffinity="0" /></body></worldbody></mujoco>'
    )
    t0, t1 = HINGE["pulse"]

    def effort(t: float) -> np.ndarray:
        # Small offset guards the half-open interval against floating-point time.
        return np.array([HINGE["torque"] if t0 - 1e-9 <= t < t1 - 1e-9 else 0.0])

    return Scene(
        name="hinge", mjcf=mjcf, dt=dt, duration=duration, bodies=["link"],
        joints=["hinge"], joint_init={"hinge": HINGE["q0"]}, effort=effort,
        params={"task": "hinge", "dt": dt, "duration": duration, **HINGE,
                "inertia_com": [ixx, iyy, izz]},
    )


PLASTIC_HINGE = {"stiffness": 2.0, "damping": 0.2, "target_deg": 10.0, "ramp": (0.2, 0.7),
                 "release": (1.0, 1.5), "pd": (20.0, 1.0), "duration": 2.5}


def _smooth(t: float, t0: float, t1: float) -> float:
    s = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return 3 * s**2 - 2 * s**3


class HingeBend:
    """Displacement control: a PD bends the link to `target_deg`, holds, then fades out.

    An ideal elasto-plastic spring saturates at stiffness x yield, so a torque load above
    that flows without bound; a prescribed bend keeps the plastic shift finite.
    """

    def __call__(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        p = PLASTIC_HINGE
        kp, kd = p["pd"]
        ref = np.radians(p["target_deg"]) * _smooth(t, *p["ramp"])
        gain = 1.0 - _smooth(t, *p["release"])
        return gain * (kp * (ref - q) - kd * qd)


def hinge_plastic(dt: float = 0.001, yield_deg: float = 5.0) -> Scene:
    """The hinge link with a joint spring and ideal elasto-plasticity (MuJoCo >= 4.0).

    Gravity is off. After the bend is released an ideal elasto-plastic spring rests at
    peak - yield, and an elastic one returns to 0. `yield_deg = 0` writes no
    `springyield`, so the elastic case compiles on every route.
    """
    ixx, iyy, izz = hinge_inertia()
    half = HINGE["length"] / 2
    px, py, pz = HINGE["pivot"]
    p = PLASTIC_HINGE
    # Hinge springyield is in the compiler's angle unit (degrees by default).
    springyield = f' springyield="{float(yield_deg)!r}"' if yield_deg else ""
    mjcf = (
        f'<mujoco model="hinge_plastic"><option timestep="{dt}" gravity="0 0 0" {OPTION} />'
        f'<worldbody><body name="link" pos="{px} {py} {pz}">'
        f'<joint name="hinge" type="hinge" axis="0 1 0" stiffness="{p["stiffness"]}" '
        f'damping="{p["damping"]}" armature="0" frictionloss="0"{springyield} />'
        f'<inertial pos="0 0 {-half}" mass="{HINGE["mass"]}" '
        f'diaginertia="{ixx!r} {iyy!r} {izz!r}" />'
        f'<geom name="link_geom" type="box" pos="0 0 {-half}" '
        f'size="{HINGE["width"] / 2} {HINGE["width"] / 2} {half}" contype="0" '
        f'conaffinity="0" /></body></worldbody></mujoco>'
    )
    return Scene(
        name="hinge-plastic", mjcf=mjcf, dt=dt, duration=p["duration"], bodies=["link"],
        joints=["hinge"], joint_init={"hinge": 0.0}, controller=HingeBend(),
        params={"task": "hinge-plastic", "dt": dt, **p,
                "yield_rad": float(np.radians(yield_deg))},
    )


PRISM_VERTICES = [
    [0, 0.05196152422706631, -0.03], [-0.045, -0.025980762113533156, -0.03],
    [0.045, -0.025980762113533156, -0.03], [0, 0.05196152422706631, 0.03],
    [-0.045, -0.025980762113533156, 0.03], [0.045, -0.025980762113533156, 0.03],
]
PRISM_FACES = [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
               [2, 0, 3], [2, 3, 5]]


def collision_chain(dt: float = 0.0005, duration: float = 1.0) -> Scene:
    """A rolling sphere strikes a cube, which strikes a triangular prism."""
    c = 'friction="0.3 0 0" condim="3" solref=".01 1" solimp=".99 .999 .001 .5 2"'
    vertex = " ".join(str(v) for vert in PRISM_VERTICES for v in vert)
    face = " ".join(str(i) for f in PRISM_FACES for i in f)

    def body(name, pos, mass, inertia, geom):
        return (f'<body name="{name}" pos="{pos}"><freejoint name="{name}_free" />'
                f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{inertia}" />'
                f'<geom name="{name}_geom" {geom} {c} /></body>')

    mjcf = (
        f'<mujoco model="collision_chain"><option timestep="{dt}" gravity="0 0 -9.81" '
        f'{OPTION} /><asset><mesh name="triangle" vertex="{vertex}" face="{face}" /></asset>'
        f'<worldbody><geom name="floor" type="plane" size="2 2 .05" {c} />'
        + body("sphere", "-0.18 -0.012 0.03", 0.15, "5.4e-05 5.4e-05 5.4e-05",
               'type="sphere" size="0.03"')
        + body("cube", "0.0 0.0 0.03", 0.1, "6e-05 6e-05 6e-05", 'type="box" size=".03 .03 .03"')
        + body("prism", "0.1 0.02 0.03", 0.1,
               "6.374999999999999e-05 6.374999999999999e-05 6.75e-05",
               'type="mesh" mesh="triangle"')
        + "</worldbody></mujoco>"
    )
    return Scene(
        name="collision-chain", mjcf=mjcf, dt=dt, duration=duration,
        bodies=["sphere", "cube", "prism"],
        free_init={"sphere": FreeInit(linear=(1.4, 0.0, 0.0), angular=(0.0, 1.4 / 0.03, 0.0)),
                   "cube": FreeInit(), "prism": FreeInit()},
        params={"task": "collision_chain", "dt": dt, "duration": duration, "friction": 0.3},
    )
