import xml.etree.ElementTree as ET

import numpy as np
import pytest
from simbench.paths import CARTWHEEL_G1
from simbench.scenes import patch
from simbench.web import decimate

needs_cartwheel = pytest.mark.skipif(
    not (CARTWHEEL_G1 / "policy.safetensors").exists(),
    reason="run `simbench fetch` for the Cartwheel G1",
)


@pytest.mark.parametrize("shape", ["disc", "square"])
def test_slip_torque_matches_uniform_pressure_integral(shape):
    mu, n, radius, side = 0.5, 10.0, 0.02, 0.03
    half = radius if shape == "disc" else side / 2
    u = (np.arange(2000) + 0.5) / 2000 * 2 * half - half
    x, y = np.meshgrid(u, u)
    r = np.hypot(x, y)
    inside = r <= radius if shape == "disc" else np.ones_like(r, bool)
    per_face = mu * n * r[inside].mean()
    assert patch.slip_torque(shape, mu, n, radius, side) == pytest.approx(2 * per_face, rel=1e-4)


def test_twist_clamps_then_ramps_torque_about_pinch_axis():
    scene = patch.twist()
    p = scene.params
    assert scene.joints == ["pad_l_slide", "pad_r_slide", "twist"]
    before, after = scene.effort(p["settle"] / 2), scene.effort(p["settle"] + 1.0)
    assert before[:2] == pytest.approx([p["clamp_n"], -p["clamp_n"]])
    assert before[2] == 0 and after[2] == pytest.approx(p["rate"])
    assert scene.effort(scene.duration)[2] > 2 * p["slip_torque_ref"]


def test_decimate_keeps_valid_faces_under_budget():
    n = 60
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, n, endpoint=False))
    vert = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    vert = vert.reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n), indexing="ij")
    a, b = j * n + i, ((j + 1) % n) * n + i
    face = np.concatenate([np.stack([a, b, b + 1], -1), np.stack([a, b + 1, a + 1], -1)])
    face = face.reshape(-1, 3)
    v, f = decimate(vert, face, max_faces=len(face) // 4)
    assert 0 < len(f) <= len(face) // 4
    assert f.max() < len(v)
    assert ((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])).all()
    assert np.abs(np.linalg.norm(v, axis=1) - 1).max() < 0.2


@needs_cartwheel
def test_g1_policy_and_motion_shapes():
    from simbench.scenes import g1

    cfg = g1.config()
    assert len(cfg["joints"]) == 29 and cfg["hist"] == 10 and cfg["goal_frames"] == 10
    q = g1.load_motion("basketball_shot")
    assert q.shape[1] == 7 + 29 and q.shape[0] > 50
    policy, _ = g1.load_policy()
    obs_dim = policy.obs_norm.mean.size // cfg["hist"]
    obs = np.zeros((cfg["hist"], obs_dim), np.float64)
    goal = np.zeros(policy.goal_norm.mean.size, np.float64)
    a = policy(obs, goal)
    assert a.shape == (29,) and np.isfinite(a).all()


@needs_cartwheel
@pytest.mark.parametrize("terrain", ["flat", "glass", "sand", "mud", "sponge"])
def test_g1_terrain_gives_feet_and_ground_the_same_contact(terrain):
    from simbench.scenes import g1

    scene = g1.walk(terrain=terrain, duration=0.1)
    root = ET.fromstring(scene.mjcf)
    spec = g1.TERRAINS[terrain]
    colliding = [g for g in root.iter("geom") if g.get("contype", "1") != "0"]
    feet = [g for b in root.iter("body") if b.get("name") in g1.FEET for g in b.findall("geom")
            if g.get("contype", "1") != "0"]
    assert len(feet) >= 8
    for g in colliding:
        assert float(g.get("friction").split()[0]) == spec["friction"]
        assert (g.get("solref"), g.get("solimp")) == (spec["solref"], spec["solimp"])
    assert float(root.find("option").get("timestep")) == 0.002
    assert scene.controller.decimation == 10
    assert scene.joints == g1.joints() and len(scene.joint_init) == 29
