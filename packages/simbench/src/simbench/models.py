"""Derived MJCF models generated from the pinned Menagerie assets.

Generation uses only the standard library, so every route compiles byte-identical
XML regardless of which MuJoCo version (if any) it has installed.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from simbench.paths import ASSETS, PANDA_DIR, PANDA_XML
from simbench.scene import OPTION, SOLIMP, SOLREF

GENERATED = ASSETS / "generated"
FINGER_OPENING = 0.04
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
PANDA_BODIES = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand",
                "left_finger", "right_finger"]
Q_READY = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]

# Fingertip pad (article grasp audit): one box per finger, in the finger frame.
PAD_HALF = (0.012, 0.004, 0.016)
PAD_POS = (0.0, 0.004, 0.0445)
TCP_POS = (0.0, 0.0, 0.0584 + PAD_POS[2])
FRICTION = 0.5

# Sweep paddle (article sweep protocol): rigid 150 g box, 220 mm wide, on the hand.
# The mount is the article's recorded paddle pose in the hand frame.
PADDLE_HALF = (0.006, 0.11, 0.026)
PADDLE_POS = (0.0071094334, 0.0, 0.0869)
PADDLE_MASS = 0.15
PEG_GRIP = 0.0101  # finger opening that clears the 10 mm peg radius

# Collision bits: pads touch objects; objects touch everything; statics touch objects.
# The sweep paddle uses the pad bit but also touches statics.
PAD, OBJECT, STATIC = 1, 2, 4
MASKS = {"pad": (PAD, OBJECT), "object": (OBJECT, PAD | OBJECT | STATIC),
         "static": (STATIC, OBJECT), "paddle": (PAD, OBJECT | STATIC)}


def contact_attrs(kind: str, friction: float = FRICTION) -> dict[str, str]:
    contype, conaffinity = MASKS[kind]
    return {"contype": str(contype), "conaffinity": str(conaffinity),
            "friction": f"{friction} 0 0", "condim": "3", "solref": SOLREF, "solimp": SOLIMP}


def _fmt(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def panda_tree(variant: str, out_dir: Path) -> ET.ElementTree:
    """Menagerie Panda simplified for the benchmark.

    Common to both variants: no tendon, finger equality, actuators, keyframe or contact
    excludes; armature, damping and frictionloss zero; Euler integrator; every source
    geom collision-free.

    free: fingers welded open at 4 cm each (the article's free-space Panda).
    pads: fingers are independent slide joints (0-4 cm) with one analytic box pad each,
          the only robot geoms that collide; a `tcp` site sits between the pads.
    paddle: finger slide joints without pads; a rigid paddle on the hand is the only
          robot geom that collides (the article's sweep robot).
    peg: fingers welded closed on a 20 mm peg; the insertion scene adds the peg body.
    """
    tree = ET.parse(PANDA_XML)
    root = tree.getroot()
    root.set("model", f"panda_{variant}")

    compiler = root.find("compiler")
    meshdir = os.path.relpath(PANDA_DIR / compiler.get("meshdir"), out_dir)
    compiler.set("meshdir", Path(meshdir).as_posix())
    option = root.find("option")
    for key, value in (part.split("=") for part in OPTION.split()):
        option.set(key, value.strip('"'))

    top_default = root.find("default")
    ET.SubElement(top_default, "geom", contype="0", conaffinity="0")
    panda_joint = top_default.find("default[@class='panda']/joint")
    for key in ("armature", "damping", "frictionloss"):
        panda_joint.set(key, "0")

    for tag in ("tendon", "equality", "actuator", "keyframe", "contact"):
        for el in root.findall(tag):
            root.remove(el)

    for name in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{name}']")
        if variant == "free":
            # The article welds both fingers at +4 cm along the *hand* Y axis (its recorded
            # poses put the right finger on the same side as the left), not along each
            # finger's own slide axis. Kept so the effort replay matches its traces exactly.
            body.remove(body.find("joint"))
            pos = np.array([float(v) for v in body.get("pos", "0 0 0").split()])
            body.set("pos", _fmt(pos + np.array([0.0, FINGER_OPENING, 0.0])))
        elif variant == "peg":
            body.remove(body.find("joint"))
            pos = np.array([float(v) for v in body.get("pos", "0 0 0").split()])
            side = 1.0 if name == "left_finger" else -1.0
            body.set("pos", _fmt(pos + np.array([0.0, side * PEG_GRIP, 0.0])))
        else:
            for geom in body.findall("geom"):
                if geom.get("class", "").startswith("fingertip_pad_collision"):
                    body.remove(geom)
            if variant == "pads":
                ET.SubElement(body, "geom", name=f"{name}_pad", type="box",
                              size=_fmt(PAD_HALF), pos=_fmt(PAD_POS), rgba="0.9 0.5 0.1 1",
                              **contact_attrs("pad"))
    hand = root.find(".//body[@name='hand']")
    if variant == "pads":
        ET.SubElement(hand, "site", name="tcp", pos=_fmt(TCP_POS), size="0.005")
    if variant == "paddle":
        paddle = ET.SubElement(hand, "body", name="paddle", pos=_fmt(PADDLE_POS))
        inertia = [PADDLE_MASS / 3 * (PADDLE_HALF[(i + 1) % 3] ** 2 + PADDLE_HALF[(i + 2) % 3] ** 2)
                   for i in range(3)]
        ET.SubElement(paddle, "inertial", pos="0 0 0", mass=repr(PADDLE_MASS),
                      diaginertia=_fmt(inertia))
        ET.SubElement(paddle, "geom", name="paddle_geom", type="box", size=_fmt(PADDLE_HALF),
                      rgba="0.25 0.3 0.35 1", **contact_attrs("paddle"))
    return tree


def write(tree: ET.ElementTree, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(path, encoding="unicode")
    return path


def build_panda_article() -> Path:
    """The article's simplified free-space Panda as a standalone file (for the audit)."""
    tree = panda_tree("free", GENERATED)
    tree.getroot().set("model", "panda_article")
    return write(tree, GENERATED / "panda_article.xml")


MODELS = {
    "panda": lambda: PANDA_XML,
    "panda_article": build_panda_article,
}


def model_path(name: str) -> Path:
    return MODELS[name]()
