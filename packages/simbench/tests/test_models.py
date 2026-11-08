import xml.etree.ElementTree as ET

import pytest
from simbench.models import ARM_JOINTS, FINGER_JOINTS, MASKS, build_panda_article, panda_tree
from simbench.paths import PANDA_XML
from simbench.scene import SCENE_DIR

pytestmark = pytest.mark.skipif(not PANDA_XML.exists(), reason="run `make fetch` first")


def test_panda_article_is_simplified():
    root = ET.parse(build_panda_article()).getroot()
    assert root.find("option").get("integrator") == "Euler"
    for tag in ("tendon", "equality", "actuator", "keyframe", "contact"):
        assert root.find(tag) is None
    joints = [j.get("name") for j in root.iter("joint") if j.get("name")]
    assert joints == [f"joint{i}" for i in range(1, 8)]
    panda_joint = root.find("default/default[@class='panda']/joint")
    assert panda_joint.get("armature") == panda_joint.get("damping") == "0"


def test_fingers_welded_as_in_article():
    # Both offsets along hand +Y: the article's recorded right-finger pose, not a mirror.
    root = ET.parse(build_panda_article()).getroot()
    left = [float(v) for v in root.find(".//body[@name='left_finger']").get("pos").split()]
    right = [float(v) for v in root.find(".//body[@name='right_finger']").get("pos").split()]
    assert left == pytest.approx([0, 0.04, 0.0584])
    assert right == pytest.approx([0, 0.04, 0.0584])


def test_pads_variant():
    root = panda_tree("pads", SCENE_DIR).getroot()
    joints = [j.get("name") for j in root.iter("joint") if j.get("name")]
    assert joints == ARM_JOINTS + FINGER_JOINTS
    colliding = [g for g in root.iter("geom") if g.get("contype", "0") != "0"]
    assert [g.get("name") for g in colliding] == ["left_finger_pad", "right_finger_pad"]
    assert not [g for g in root.iter("geom") if "fingertip" in g.get("class", "")]
    assert root.find(".//body[@name='hand']/site[@name='tcp']") is not None


def _collides(a: str, b: str) -> bool:
    (ta, ca), (tb, cb) = MASKS[a], MASKS[b]
    return bool(ta & cb or tb & ca)


def test_collision_masks():
    assert _collides("pad", "object") and _collides("object", "object")
    assert _collides("object", "static")
    assert not _collides("pad", "static") and not _collides("pad", "pad")
    assert not _collides("static", "static")
