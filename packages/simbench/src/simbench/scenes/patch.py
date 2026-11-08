"""Flat-on-flat contact patch: a part pinched between two pads and twisted until it slips.

twist   A disc (or square) puck is clamped between two flat pads on slide joints, each
        pushed in with a constant force N, then twisted about the pinch axis by a slow
        torque ramp. With uniform pressure over the contact face the slip torque has a
        closed form, two faces of
          disc   (2/3) mu N R
          square mu N a (sqrt(2) + ln(1 + sqrt(2))) / 6
        A point-contact model resists twist only through torsional friction
        (condim 4/6, a length that has to be tuned to the patch) or through several
        contact points spread over the face (multiccd, or Newton's hydroelastic patch).
"""

import math

import numpy as np

from simbench.scene import SOLIMP, SOLREF, Scene

TWIST = {"mu": 0.5, "clamp_n": 10.0, "radius": 0.02, "side": 0.03, "thickness": 0.02,
         "puck_mass": 0.1, "pad_half": (0.03, 0.004, 0.03), "pad_mass": 0.5,
         "settle": 0.2, "rate": 0.2, "duration": 2.2}


def slip_torque(shape: str, mu: float, clamp_n: float, radius: float, side: float) -> float:
    """Uniform-pressure Coulomb limit for two contact faces [N m]."""
    if shape == "disc":
        per_face = 2 / 3 * mu * clamp_n * radius
    else:
        per_face = mu * clamp_n * side * (math.sqrt(2) + math.log(1 + math.sqrt(2))) / 6
    return 2 * per_face


def twist(dt: float = 0.001, shape: str = "disc", condim: int = 3,
          torsion: float | None = None, multiccd: bool = False) -> Scene:
    """`torsion` [m] sets MuJoCo's torsional friction (used with condim 4/6)."""
    p = TWIST
    mu, n, t = p["mu"], p["clamp_n"], p["thickness"]
    px, py, pz = p["pad_half"]
    spin = torsion if torsion is not None else 0.005
    friction = f'friction="{mu} {spin} 0.0001" condim="{condim}"'
    c = f'{friction} solref="{SOLREF}" solimp="{SOLIMP}"'
    if shape == "disc":
        puck = (f'<geom name="puck_geom" type="cylinder" size="{p["radius"]} {t / 2}" '
                f'euler="90 0 0" mass="{p["puck_mass"]}" rgba="0.85 0.55 0.2 1" {c} />')
    else:
        a = p["side"] / 2
        puck = (f'<geom name="puck_geom" type="box" size="{a} {t / 2} {a}" '
                f'mass="{p["puck_mass"]}" rgba="0.85 0.55 0.2 1" {c} />')
    flag = '<flag multiccd="enable" />' if multiccd else ""
    gap = t / 2 + py

    def pad(name: str, y: float) -> str:
        return (f'<body name="{name}" pos="0 {y} 0.1"><joint name="{name}_slide" type="slide" '
                f'axis="0 1 0" /><geom name="{name}_geom" type="box" size="{px} {py} {pz}" '
                f'mass="{p["pad_mass"]}" rgba="0.55 0.6 0.65 1" {c} /></body>')

    mjcf = (
        f'<mujoco model="twist"><option timestep="{dt}" gravity="0 0 0" integrator="Euler" '
        f'solver="Newton" iterations="100" tolerance="1e-10" cone="elliptic">{flag}</option>'
        f'<worldbody><body name="puck" pos="0 0 0.1"><joint name="twist" type="hinge" '
        f'axis="0 1 0" />{puck}</body>{pad("pad_l", -gap)}{pad("pad_r", gap)}'
        f'</worldbody></mujoco>'
    )
    settle, rate = p["settle"], p["rate"]

    def effort(time: float) -> np.ndarray:
        ramp = rate * max(0.0, time - settle)
        return np.array([n, -n, ramp])

    ref = slip_torque(shape, mu, n, p["radius"], p["side"])
    return Scene(
        name="twist", mjcf=mjcf, dt=dt, duration=p["duration"], bodies=["puck", "pad_l", "pad_r"],
        joints=["pad_l_slide", "pad_r_slide", "twist"], effort=effort,
        params={"task": "twist", "dt": dt, "shape": shape, "condim": condim,
                "torsion": spin, "multiccd": multiccd, "slip_torque_ref": ref, **p},
    )
