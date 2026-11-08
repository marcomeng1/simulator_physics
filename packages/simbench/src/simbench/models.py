"""Derived MJCF models generated from the pinned Menagerie assets.

Generation uses only the standard library, so every route compiles byte-identical
XML regardless of which MuJoCo version (if any) it has installed.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from simbench.paths import ASSETS, PANDA_DIR, PANDA_XML

GENERATED = ASSETS / "generated"
FINGER_OPENING = 0.04
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def _quat_rotate(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz / np.linalg.norm(quat_wxyz)
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def build_panda_article() -> Path:
    """The article's simplified free-space Panda.

    Seven dynamic arm joints on a fixed base; fingers welded open at 4 cm each;
    no tendon, finger equality, actuators or keyframe; all collisions disabled;
    armature, damping and frictionloss zero; Euler integrator.
    """
    tree = ET.parse(PANDA_XML)
    root = tree.getroot()
    root.set("model", "panda_article")

    compiler = root.find("compiler")
    meshdir = Path("..") / PANDA_DIR.relative_to(ASSETS) / compiler.get("meshdir")
    compiler.set("meshdir", meshdir.as_posix())
    root.find("option").set("integrator", "Euler")

    top_default = root.find("default")
    ET.SubElement(top_default, "geom", contype="0", conaffinity="0")
    panda_joint = top_default.find("default[@class='panda']/joint")
    panda_joint.set("armature", "0")
    panda_joint.set("damping", "0")
    panda_joint.set("frictionloss", "0")

    for tag in ("tendon", "equality", "actuator", "keyframe", "contact"):
        for el in root.findall(tag):
            root.remove(el)

    for name in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{name}']")
        joint = body.find("joint")
        body.remove(joint)
        pos = np.array([float(v) for v in body.get("pos", "0 0 0").split()])
        quat = np.array([float(v) for v in body.get("quat", "1 0 0 0").split()])
        body.set("pos", _fmt(pos + _quat_rotate(quat, np.array([0.0, FINGER_OPENING, 0.0]))))

    GENERATED.mkdir(parents=True, exist_ok=True)
    out = GENERATED / "panda_article.xml"
    ET.indent(tree)
    tree.write(out, encoding="unicode")
    return out


MODELS = {
    "panda": lambda: PANDA_XML,
    "panda_article": build_panda_article,
}


def model_path(name: str) -> Path:
    return MODELS[name]()
