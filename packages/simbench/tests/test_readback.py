import numpy as np
from simbench.readback import compare, inertia_tensor, probe_configurations, quat_to_mat


def test_quat_to_mat_is_rotation():
    q = np.array([0.9238795, 0.0, 0.0, -0.3826834])
    rot = quat_to_mat(q)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rot @ [1, 0, 0], [np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0],
                               atol=1e-7)


def test_inertia_tensor_roundtrip():
    full = np.array([[0.7, -1e-4, 6e-3], [-1e-4, 0.7, 2e-2], [6e-3, 2e-2, 9e-3]])
    principal, axes = np.linalg.eigh(full)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    w = np.sqrt(max(0.0, 1 + np.trace(axes))) / 2
    q = np.array([w, (axes[2, 1] - axes[1, 2]) / (4 * w), (axes[0, 2] - axes[2, 0]) / (4 * w),
                  (axes[1, 0] - axes[0, 1]) / (4 * w)])
    np.testing.assert_allclose(inertia_tensor(principal, q), full, atol=1e-12)


def test_probe_configurations_within_limits_and_seeded():
    ranges = {"a": (-1.0, 1.0), "b": (0.0, 0.04)}
    configs = probe_configurations(ranges)
    assert configs == probe_configurations(ranges)
    assert configs[0] == {"a": 0.0, "b": 0.02}
    for cfg in configs:
        for name, value in cfg.items():
            assert ranges[name][0] <= value <= ranges[name][1]


def _readback(route, mass=1.0, armature=0.0, z=0.0):
    return {
        "route": route,
        "bodies": {"link": {"mass": mass, "com": [0, 0, 0], "inertia": np.eye(3).tolist()}},
        "joints": {"j": {"type": "hinge", "axis": [0, 0, 1], "range": [-1, 1],
                         "armature": armature, "damping": 0.0, "frictionloss": 0.0}},
        "fk_poses": [{"link": [0, 0, z, 1, 0, 0, 0]}],
    }


def test_compare_reports_worst_differences_and_missing():
    report = compare(_readback("ref"), _readback("other", mass=1.5, armature=0.1, z=2e-3))
    assert report["max_abs_diff"]["mass"] == 0.5
    assert report["max_abs_diff"]["armature"] == 0.1
    assert abs(report["max_abs_diff"]["fk_pos"] - 2e-3) < 1e-15
    assert report["worst_at"]["armature"] == "j"

    other = _readback("other")
    other["bodies"] = {}
    assert compare(_readback("ref"), other)["missing"] == ["body:link"]
