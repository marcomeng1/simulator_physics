import xml.etree.ElementTree as ET

import numpy as np
import pytest
from simbench.metrics import tumble_reference
from simbench.scenes import deform, thin


def test_card_starts_above_plate_and_is_thin():
    scene = thin.card_catch(speed=1.0, plate_mm=1.0)
    p = scene.params
    card = ET.fromstring(scene.mjcf).find(".//body[@name='card']")
    z0 = float(card.get("pos").split()[2])
    assert z0 - p["card_half"][2] - p["plate_top"] == pytest.approx(0.005)
    assert p["plate_top"] - p["plate_bottom"] == pytest.approx(0.001)
    assert 2 * p["card_half"][2] == pytest.approx(0.0003)


def test_tumble_inertia_is_physical_and_spins_about_middle_axis():
    a, b, c = thin.TUMBLE["inertia"]
    assert a + b > c and b + c > a and a + c > b
    assert a < b < c
    assert np.argmax(np.abs(thin.TUMBLE["omega0"])) == 1


def test_tumble_reference_flips_over():
    p = thin.TUMBLE
    q = tumble_reference(p, np.linspace(0.0, p["duration"], 81))
    assert np.linalg.norm(q, axis=1) == pytest.approx(1.0)
    y_axis_y = 1 - 2 * (q[:, 1] ** 2 + q[:, 3] ** 2)
    assert y_axis_y.min() < -0.9


def test_sheet_clamps_two_columns_and_names_flex_bodies():
    scene = deform.sheet()
    root = ET.fromstring(scene.mjcf)
    nx = deform.SHEET["nx"]
    free = {b.get("name") for b in root.iter("body") if b.find("joint") is not None}
    clamped = [f"sheet_{k}" for k in range(deform.SHEET["nx"] * deform.SHEET["ny"])
               if k % nx <= 1]
    assert not free & set(clamped)
    assert set(scene.params["tip_bodies"]) <= free
    assert scene.params["flex_prefix"] == "sheet_"
    assert root.find(".//option").get("integrator") == "discrete"
