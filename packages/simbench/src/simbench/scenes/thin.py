"""Scenes for two MuJoCo 4.0 changes: speculative contacts and fourth-order RK4 rotations.

card-catch  An 88 x 63 x 0.3 mm, 1.8 g playing card released flat 5 mm above a 1 mm
            catch plate with a downward speed (and optionally a spin about its long
            axis). The card is lost once its centre passes the plate's mid-plane, where
            the box-box contact normal flips. Two things get it there: a soft contact
            (time constant `contact_tc`) that lets it sink more than the combined half
            thickness (0.65 mm) before stopping it, and a step longer than that half
            thickness, which only continuous collision (`speculative`, 4.0) handles. The
            default time constant is two timesteps, the stiffest MuJoCo recommends.
tumble      A free T-handle spun about its intermediate principal axis with no gravity
            (the Dzhanibekov flip). 4.0 made RK4 fourth-order for free-body
            orientations; the metric compares against an independent rigid-body solve.
"""

from simbench.scene import SOLIMP, FreeInit, Scene

CARD_HALF = (0.044, 0.0315, 0.00015)
CARD_MASS = 0.0018
PLATE_HALF_XY, PLATE_Z, GAP = 0.15, 0.1, 0.005


def card_catch(dt: float = 0.001, speed: float = 1.0, spin: float = 0.0,
               plate_mm: float = 1.0, duration: float = 0.3,
               contact_tc: float | None = None) -> Scene:
    """Card dropped at `speed` [m/s] (spin [rad/s] about x) onto a `plate_mm` thick plate."""
    tc = 2 * dt if contact_tc is None else contact_tc
    half_t = plate_mm / 2e3
    top = PLATE_Z + half_t
    z0 = top + GAP + CARD_HALF[2]
    c = f'friction="0.4 0 0" condim="3" solref="{tc!r} 1" solimp="{SOLIMP}"'
    mjcf = (
        f'<mujoco model="card_catch"><option timestep="{dt}" gravity="0 0 -9.81" '
        f'integrator="Euler" solver="Newton" iterations="100" tolerance="1e-10" '
        f'cone="elliptic" /><worldbody>'
        f'<geom name="plate" type="box" pos="0 0 {PLATE_Z}" '
        f'size="{PLATE_HALF_XY} {PLATE_HALF_XY} {half_t!r}" rgba="0.55 0.6 0.65 1" {c} />'
        f'<body name="card" pos="0 0 {z0!r}"><freejoint name="card_free" />'
        f'<geom name="card_geom" type="box" size="{CARD_HALF[0]} {CARD_HALF[1]} '
        f'{CARD_HALF[2]}" mass="{CARD_MASS}" rgba="0.95 0.95 0.92 1" {c} /></body>'
        f'</worldbody></mujoco>'
    )
    return Scene(
        name="card-catch", mjcf=mjcf, dt=dt, duration=duration, bodies=["card"],
        free_init={"card": FreeInit(linear=(0.0, 0.0, -speed), angular=(spin, 0.0, 0.0))},
        params={"task": "card-catch", "dt": dt, "duration": duration, "speed": speed,
                "spin": spin, "plate_mm": plate_mm, "contact_tc": tc, "plate_top": top,
                "plate_bottom": PLATE_Z - half_t, "plate_half_xy": PLATE_HALF_XY,
                "card_half": list(CARD_HALF), "card_mass": CARD_MASS},
    )


TUMBLE = {"mass": 0.2, "inertia": (1e-4, 4e-4, 4.8e-4), "omega0": (0.05, 6.0, 0.0),
          "duration": 4.0, "pos": (0.0, 0.0, 0.3)}


def tumble(dt: float = 0.002, integrator: str = "RK4") -> Scene:
    """T-handle (stem along x, bar along z) spun about y, the intermediate axis."""
    p = TUMBLE
    ixx, iyy, izz = p["inertia"]
    x, y, z = p["pos"]
    vis = 'contype="0" conaffinity="0" rgba="0.85 0.55 0.2 1"'
    mjcf = (
        f'<mujoco model="tumble"><option timestep="{dt}" gravity="0 0 0" '
        f'integrator="{integrator}" /><worldbody>'
        f'<body name="handle" pos="{x} {y} {z}"><freejoint name="handle_free" />'
        f'<inertial pos="0 0 0" mass="{p["mass"]}" diaginertia="{ixx} {iyy} {izz}" />'
        f'<geom name="stem" type="capsule" fromto="-0.06 0 0 0.03 0 0" size="0.008" {vis} />'
        f'<geom name="bar" type="capsule" fromto="0.03 0 -0.05 0.03 0 0.05" size="0.008" '
        f'{vis} /></body></worldbody></mujoco>'
    )
    return Scene(
        name="tumble", mjcf=mjcf, dt=dt, duration=p["duration"], bodies=["handle"],
        free_init={"handle": FreeInit(angular=p["omega0"])},
        params={"task": "tumble", "dt": dt, "integrator": integrator, **p},
    )
