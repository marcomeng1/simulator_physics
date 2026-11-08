"""Phase 3 scenes: the Franka Panda from the pinned Menagerie revision.

panda-effort    open-loop constant-feedforward effort replay with two torque pulses (0.5 s)
panda-tracking  1 kHz PD + frozen inverse-dynamics feedforward along a quintic (2.5 s)
grasp           pad grasp and 8 cm lift of a 40 mm cube (3.5 s)
stack           pick, transfer 100 mm and place on a dynamic support block (6 s)
push            push three blocks through a fixed channel with the pads (5 s)

Timings, gains, pad geometry, masses and friction follow the article's audits. The
article does not publish its table/object layout or IK waypoints, so the contact
scenes use the layout defined here; every engine sees the same MJCF and tape.
"""

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
from simbench.scene import SCENE_DIR, Scene

CTRL_DT = 0.001
KP = np.array([80, 80, 60, 60, 20, 15, 8, 400, 400], dtype=float)
KD = np.array([18, 18, 14, 14, 5, 4, 2, 4, 4], dtype=float)
LIMIT = np.array([87, 87, 87, 87, 12, 12, 12, 4, 4], dtype=float)

CUBE = 0.04
CUBE_MASS = 0.1
OPEN, CLOSED = 0.04, 0.012
TRACKING_OFFSET = [0.3, 0.25, 0.2, 0.2, -0.25, 0.15, 0.3]
PULSES = [(1, 2.0, 0.05, 0.15), (3, -1.0, 0.15, 0.25)]  # (joint index, N m, t0, t1)


class TapeController:
    """u = clip(ff + kp (qref - q) + kd (qdref - qd)), sampled and held at 1 kHz."""

    def __init__(self, tape: dict, n: int):
        self.qref, self.qdref, self.ff = tape["qref"], tape["qdref"], tape["ff"]
        self.kp, self.kd, self.limit = KP[:n], KD[:n], LIMIT[:n]
        self.u = None

    def __call__(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        i = t / CTRL_DT
        k = int(round(i))
        if self.u is None or abs(i - k) < 1e-6:
            k = min(k, len(self.ff) - 1)
            u = self.ff[k] + self.kp * (self.qref[k] - q) + self.kd * (self.qdref[k] - qd)
            self.u = np.clip(u, -self.limit, self.limit)
        return self.u


def _mjcf(variant: str, dt: float, extra=()) -> str:
    tree = panda_tree(variant, SCENE_DIR)
    root = tree.getroot()
    root.find("option").set("timestep", repr(dt))
    world = root.find("worldbody")
    for build in extra:
        build(world)
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode")


def _static_box(name, pos, half):
    def build(world):
        ET.SubElement(world, "geom", name=name, type="box", pos=_fmt(pos), size=_fmt(half),
                      rgba="0.55 0.5 0.45 1", **contact_attrs("static"))
    return build


def _cube(name, pos, rgba="0.2 0.5 0.9 1"):
    def build(world):
        body = ET.SubElement(world, "body", name=name, pos=_fmt(pos))
        ET.SubElement(body, "freejoint", name=f"{name}_free")
        inertia = CUBE_MASS * CUBE**2 / 6
        ET.SubElement(body, "inertial", pos="0 0 0", mass=repr(CUBE_MASS),
                      diaginertia=_fmt([inertia] * 3))
        ET.SubElement(body, "geom", name=f"{name}_geom", type="box", size=_fmt([CUBE / 2] * 3),
                      rgba=rgba, **contact_attrs("object"))
    return build


def _hold(tape: dict) -> dict[str, float]:
    return dict(zip(ARM_JOINTS + FINGER_JOINTS, tape["qref"][0].tolist(), strict=False))


# --- free-space ----------------------------------------------------------------------


BIAS_SPEC = {"name": "panda-bias", "kind": "bias", "model": "free", "q0": Q_READY}
TRACKING_SPEC = {"name": "panda-tracking", "kind": "joint", "model": "free", "q0": Q_READY,
                 "offset": TRACKING_OFFSET, "duration": 2.5, "ctrl_dt": CTRL_DT}


def effort(dt: float = 0.001, duration: float = 0.5) -> Scene:
    """Article Scenario 3a: constant q0 gravity feedforward plus two torque pulses."""
    spec = BIAS_SPEC
    bias = tapes.load(spec)["bias"]

    def tau(t: float) -> np.ndarray:
        out = bias.copy()
        for j, value, t0, t1 in PULSES:
            if t0 - 1e-9 <= t < t1 - 1e-9:
                out[j] += value
        return out

    return Scene(
        name="panda-effort", mjcf=_mjcf("free", dt), dt=dt, duration=duration,
        bodies=PANDA_BODIES, joints=ARM_JOINTS, joint_init=dict(zip(ARM_JOINTS, Q_READY,
                                                                    strict=True)),
        effort=tau,
        params={"task": "panda-effort", "dt": dt, "duration": duration, "q0": Q_READY,
                "pulses": PULSES, "tape": tapes.spec_sha(spec), "bias": bias.tolist()},
    )


def tracking(dt: float = 0.001) -> Scene:
    spec = TRACKING_SPEC
    duration = spec["duration"]
    tape = tapes.load(spec)
    return Scene(
        name="panda-tracking", mjcf=_mjcf("free", dt), dt=dt, duration=duration,
        bodies=PANDA_BODIES, joints=ARM_JOINTS, joint_init=_hold(tape),
        controller=TapeController(tape, 7),
        params={"task": "panda-tracking", "dt": dt, "duration": duration,
                "tape": tapes.spec_sha(spec), "kp": KP[:7].tolist(), "kd": KD[:7].tolist(),
                "limit": LIMIT[:7].tolist(), "ctrl_dt": CTRL_DT},
    )


# --- contact tasks -------------------------------------------------------------------


def _contact_scene(name, dt, spec, objects, statics, params, object_names):
    tape = tapes.load(spec)
    return Scene(
        name=name, mjcf=_mjcf("pads", dt, [*statics, *objects]), dt=dt,
        duration=spec["duration"], bodies=PANDA_BODIES + object_names,
        joints=ARM_JOINTS + FINGER_JOINTS, joint_init=_hold(tape),
        controller=TapeController(tape, 9),
        params={"task": name, "dt": dt, "duration": spec["duration"], "tape_spec": spec,
                "tape": tapes.spec_sha(spec), "kp": KP.tolist(), "kd": KD.tolist(),
                "limit": LIMIT.tolist(), "ctrl_dt": CTRL_DT, "friction": 0.5, **params},
    )


GRASP_AT = (0.5, 0.0)


def grasp_spec() -> dict:
    x, y = GRASP_AT
    z = CUBE / 2
    return {"name": "grasp", "kind": "cartesian", "duration": 3.5, "ctrl_dt": CTRL_DT,
            "q_seed": Q_READY, "tcp_start": [x, y, z],
            "tcp_segments": [[1.0, 1.8, [x, y, z + 0.08], "joint"]],
            "finger_start": [OPEN, OPEN],
            "finger_segments": [[0.3, 0.8, [CLOSED, CLOSED]], [2.3, 2.7, [OPEN, OPEN]]]}


def grasp(dt: float = 0.001, offset_mm: float = 0.0) -> Scene:
    """Centred, or offset along hand-local X (world X for the downward hand)."""
    x, y = GRASP_AT
    cube = (x + offset_mm * 1e-3, y, CUBE / 2)
    return _contact_scene(
        "grasp", dt, grasp_spec(), [_cube("cube", cube)],
        [_static_box("table", (x, y, -0.025), (0.15, 0.15, 0.025))],
        {"offset_mm": offset_mm, "cube": cube}, ["cube"],
    )


STACK_PICK, STACK_SUPPORT = (0.5, -0.05), (0.5, 0.05)
LIFT, CLEARANCE = 0.08, 0.008


def stack_spec(offset_mm: float) -> dict:
    (px, py), (sx, sy) = STACK_PICK, STACK_SUPPORT
    z = CUBE / 2
    place = [sx + offset_mm * 1e-3, sy, z + CUBE + CLEARANCE]
    return {"name": f"stack-{offset_mm:g}mm", "kind": "cartesian", "duration": 6.0,
            "ctrl_dt": CTRL_DT, "q_seed": Q_READY, "tcp_start": [px, py, z],
            "tcp_segments": [[1.0, 1.8, [px, py, z + LIFT], "joint"],
                             [1.8, 2.6, [place[0], place[1], z + LIFT], "joint"],
                             [2.6, 3.4, place, "joint"],
                             [4.2, 4.8, [place[0], place[1], place[2] + LIFT], "joint"]],
            "finger_start": [OPEN, OPEN],
            "finger_segments": [[0.3, 0.8, [CLOSED, CLOSED]], [3.6, 4.0, [OPEN, OPEN]]]}


def stack(dt: float = 0.001, offset_mm: float = 0.0,
          perturb_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Scene:
    """Place a cube on an unanchored support with a nominal X offset from its centre."""
    (px, py), (sx, sy) = STACK_PICK, STACK_SUPPORT
    upper = (px + perturb_mm[0] * 1e-3, py + perturb_mm[1] * 1e-3,
             CUBE / 2 + perturb_mm[2] * 1e-3)
    return _contact_scene(
        "stack", dt, stack_spec(offset_mm),
        [_cube("upper", upper), _cube("support", (sx, sy, CUBE / 2), "0.9 0.6 0.2 1")],
        [_static_box("table", (0.5, 0.0, -0.025), (0.4, 0.3, 0.025))],
        {"offset_mm": offset_mm, "perturb_mm": list(perturb_mm)}, ["upper", "support"],
    )


PUSH_REAR_X, PUSH_ENTRANCE_X, PUSH_TRAVEL = 0.40, 0.47, 0.24
PUSH_FRONT_GAP, PUSH_FRONT_Y = 0.001, 0.022
PUSH_PASS_MM = 60.0
WALL_HALF = (0.125, 0.01, 0.02)


def push_spec() -> dict:
    start = [PUSH_REAR_X - CUBE / 2 - 0.012 - 0.003, 0.0, CUBE / 2]
    end = [start[0] + PUSH_TRAVEL, 0.0, CUBE / 2]
    return {"name": "push", "kind": "cartesian", "duration": 5.0, "ctrl_dt": CTRL_DT,
            "q_seed": Q_READY, "tcp_start": start,
            "tcp_segments": [[0.5, 3.5, end, "line"]],
            "finger_start": [CLOSED, CLOSED], "finger_segments": []}


def push(dt: float = 0.001, width_mm: float = 90.0, layout_mm: float = 0.0) -> Scene:
    """Rear cube pushes two side-by-side blocks (84 mm span) into a fixed channel.

    `layout_mm` shifts the front pair laterally (the article's +-1 mm layouts).
    """
    front_x = PUSH_REAR_X + CUBE + PUSH_FRONT_GAP
    dy = layout_mm * 1e-3
    wall_x = PUSH_ENTRANCE_X + WALL_HALF[0]
    wall_y = width_mm * 1e-3 / 2 + WALL_HALF[1]
    return _contact_scene(
        "push", dt, push_spec(),
        [_cube("cube", (PUSH_REAR_X, 0.0, CUBE / 2)),
         _cube("block_left", (front_x, -PUSH_FRONT_Y + dy, CUBE / 2), "0.9 0.6 0.2 1"),
         _cube("block_right", (front_x, PUSH_FRONT_Y + dy, CUBE / 2), "0.3 0.8 0.4 1")],
        [_static_box("table", (0.5, 0.0, -0.025), (0.4, 0.3, 0.025)),
         _static_box("wall_left", (wall_x, -wall_y, WALL_HALF[2]), WALL_HALF),
         _static_box("wall_right", (wall_x, wall_y, WALL_HALF[2]), WALL_HALF)],
        {"width_mm": width_mm, "layout_mm": layout_mm, "entrance_x": PUSH_ENTRANCE_X,
         "pass_mm": PUSH_PASS_MM}, ["cube", "block_left", "block_right"],
    )


def tape_specs() -> list[dict]:
    """Every tape the Phase 3 plan needs (generated by `simbench make-tapes`)."""
    specs = [BIAS_SPEC, TRACKING_SPEC, grasp_spec(), push_spec()]
    # 24 and 26 mm extend the article's offsets: under load the hand lags its unloaded
    # reference by ~4 mm in -X with this layout, so these reach past the static edge.
    specs += [stack_spec(o) for o in (0.0, 16.0, 20.0, 22.0, 24.0, 26.0)]
    return specs
