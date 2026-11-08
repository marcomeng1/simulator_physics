import xml.etree.ElementTree as ET

import pytest
from simbench.models import build_panda_article
from simbench.paths import PANDA_XML

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


def test_fingers_welded_open():
    root = ET.parse(build_panda_article()).getroot()
    left = [float(v) for v in root.find(".//body[@name='left_finger']").get("pos").split()]
    right = [float(v) for v in root.find(".//body[@name='right_finger']").get("pos").split()]
    assert left == pytest.approx([0, 0.04, 0.0584])
    assert right == pytest.approx([0, -0.04, 0.0584], abs=1e-12)
