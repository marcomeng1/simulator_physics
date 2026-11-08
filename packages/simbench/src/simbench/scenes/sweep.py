"""Phase 6 crowded sweep: a Panda paddle gathers 15 objects and pushes them into a tray.

Rebuilt from the article's published sweep bundle (`make fetch`): the static geometry,
object shapes and initial positions come from its recorded scene, and the 12.5 s joint
reference is recovered from its MuJoCo CPU recording (`tapes.recorded_joints`), since the
frozen pilot-07 reference arrays are not published. Protocol values follow the article:
density 500 kg/m^3, friction 0.5, analytic cylinder mass and inertia on shared 32-sided
convex meshes, a 150 g paddle that touches objects and statics, arm KP x4 / KD x2, 1 ms.
"""

import xml.etree.ElementTree as ET
from functools import cache

import numpy as np

from simbench import tapes
from simbench.models import (
    ARM_JOINTS,
    FINGER_JOINTS,
    PANDA_BODIES,
    _fmt,
    contact_attrs,
    panda_tree,
)
from simbench.reference import BUNDLES, load_bundle
from simbench.scene import SCENE_DIR, Scene
from simbench.scenes.panda import CTRL_DT, KD, KP, LIMIT, TapeController, _hold

DENSITY = 500.0
DURATION = 12.5
FINGER_HOLD = 0.012
GAIN_KP, GAIN_KD = 4.0, 2.0
COLORS = {"sphere": "0.85 0.35 0.3 1", "box": "0.25 0.55 0.9 1", "mesh": "0.95 0.75 0.25 1"}

SPEC = {"name": "sweep", "kind": "samples", "model": "paddle", "bundle": BUNDLES["sweep"],
        "engine": "mujoco_cpu", "finger_hold": FINGER_HOLD, "duration": DURATION,
        "ctrl_dt": CTRL_DT}


@cache
def layout() -> dict:
    """Static geometry and object shapes/positions from the article's recorded scene."""
    trace = next(t for t in load_bundle(BUNDLES["sweep"])["traces"]
                 if t["engine"] == "mujoco_cpu")
    pose = {b["id"]: b for b in trace["frames"][0]["bodies"]}
    shapes = {b["id"]: b for b in trace["scene"]["bodies"]}
    table = shapes["table"]
    top = pose["table"]["position_m"][2] + table["size"][2] / 2
    statics, objects = [], []
    for name, b in shapes.items():
        if name.startswith(("table", "tray", "wall", "guide", "ramp")):
            statics.append({"name": name, "shape": b["shape"], "size": b.get("size"),
                            "vertices": b.get("vertices"),
                            "pos": pose[name]["position_m"]})
        elif name.startswith("object_"):
            v = np.asarray(b.get("vertices", []))
            half = (b["radius"] if b["shape"] == "sphere" else
                    b["size"][2] / 2 if b["shape"] == "box" else v[:, 2].max())
            xy = np.round(pose[name]["position_m"][:2], 6)
            objects.append({"name": name, "shape": b["shape"], "radius": b.get("radius"),
                            "size": b.get("size"), "vertices": b.get("vertices"),
                            "pos": [*xy.tolist(), top + half]})
    tray = next(s for s in statics if s["name"] == "tray")
    return {"statics": statics, "objects": objects, "table_top": top,
            "tray": {"pos": tray["pos"], "size": tray["size"]}}


def _inertial(obj: dict) -> tuple[float, list[float]]:
    if obj["shape"] == "sphere":
        r = obj["radius"]
        m = DENSITY * 4 / 3 * np.pi * r**3
        return m, [0.4 * m * r**2] * 3
    if obj["shape"] == "box":
        s = obj["size"]
        m = DENSITY * np.prod(s)
        return m, [m / 12 * (s[(i + 1) % 3] ** 2 + s[(i + 2) % 3] ** 2) for i in range(3)]
    v = np.asarray(obj["vertices"])  # analytic cylinder about z, not the polygon prism
    r, h = np.linalg.norm(v[:, :2], axis=1).max(), np.ptp(v[:, 2])
    m = DENSITY * np.pi * r**2 * h
    return m, [m * (3 * r**2 + h**2) / 12] * 2 + [m * r**2 / 2]


def _mjcf(dt: float) -> str:
    lay = layout()
    tree = panda_tree("paddle", SCENE_DIR)
    root = tree.getroot()
    root.find("option").set("timestep", repr(dt))
    asset, world = root.find("asset"), root.find("worldbody")
    for s in lay["statics"]:
        attrs = {"name": s["name"], "rgba": "0.55 0.5 0.45 1", **contact_attrs("static")}
        if s["shape"] == "box":
            ET.SubElement(world, "geom", type="box", pos=_fmt(s["pos"]),
                          size=_fmt(np.asarray(s["size"]) / 2), **attrs)
        else:  # convex polyhedron given in world coordinates
            ET.SubElement(asset, "mesh", name=f"{s['name']}_mesh",
                          vertex=_fmt(np.ravel(s["vertices"])))
            ET.SubElement(world, "geom", type="mesh", mesh=f"{s['name']}_mesh", **attrs)
    for o in lay["objects"]:
        body = ET.SubElement(world, "body", name=o["name"], pos=_fmt(o["pos"]))
        ET.SubElement(body, "freejoint", name=f"{o['name']}_free")
        mass, inertia = _inertial(o)
        ET.SubElement(body, "inertial", pos="0 0 0", mass=repr(float(mass)),
                      diaginertia=_fmt(inertia))
        attrs = {"name": f"{o['name']}_geom", "rgba": COLORS[o["shape"]],
                 **contact_attrs("object")}
        if o["shape"] == "sphere":
            ET.SubElement(body, "geom", type="sphere", size=repr(o["radius"]), **attrs)
        elif o["shape"] == "box":
            ET.SubElement(body, "geom", type="box", size=_fmt(np.asarray(o["size"]) / 2),
                          **attrs)
        else:
            ET.SubElement(asset, "mesh", name=f"{o['name']}_mesh",
                          vertex=_fmt(np.ravel(o["vertices"])))
            ET.SubElement(body, "geom", type="mesh", mesh=f"{o['name']}_mesh", **attrs)
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode")


def sweep(dt: float = 0.001) -> Scene:
    tape = tapes.load(SPEC)
    lay = layout()
    names = [o["name"] for o in lay["objects"]]
    kp, kd = KP.copy(), KD.copy()
    kp[:7] *= GAIN_KP
    kd[:7] *= GAIN_KD
    mjcf = _mjcf(dt)
    return Scene(
        name="sweep", mjcf=mjcf, dt=dt, duration=DURATION,
        bodies=[*PANDA_BODIES, "paddle", *names], joints=ARM_JOINTS + FINGER_JOINTS,
        joint_init=_hold(tape), controller=TapeController(tape, 9, kp=kp, kd=kd),
        params={"task": "sweep", "dt": dt, "duration": DURATION, "tape_spec": SPEC,
                "tape": tapes.spec_sha(SPEC), "kp": kp.tolist(), "kd": kd.tolist(),
                "limit": LIMIT.tolist(), "ctrl_dt": CTRL_DT, "friction": 0.5,
                "density": DENSITY, "table_top": lay["table_top"], "tray": lay["tray"],
                "objects": names},
    )


def tape_specs() -> list[dict]:
    return [SPEC]
