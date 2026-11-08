"""Engine-neutral readback of an imported model, and comparison between routes.

Conventions shared by every adapter:
- Quaternions are (w, x, y, z).
- Body COM and inertia are expressed in the body frame; inertia is the full 3x3
  tensor about the COM.
- Joint axes are expressed in the child body frame.
- Bodies and joints are keyed by their MJCF names (engine prefixes stripped).
"""

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class BodyReadback:
    mass: float
    com: list[float]
    inertia: list[list[float]]


@dataclass
class JointReadback:
    type: str
    axis: list[float]
    range: list[float]
    armature: float
    damping: float
    frictionloss: float


@dataclass
class ModelReadback:
    route: str
    model: str
    bodies: dict[str, BodyReadback] = field(default_factory=dict)
    joints: dict[str, JointReadback] = field(default_factory=dict)
    # One entry per probe configuration: joint name -> angle, body name -> (pos, quat).
    fk_qpos: list[dict[str, float]] = field(default_factory=list)
    fk_poses: list[dict[str, list[float]]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=float) / np.linalg.norm(quat_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def inertia_tensor(principal: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Full inertia tensor from principal moments and the principal-axes orientation."""
    rot = quat_to_mat(quat_wxyz)
    return rot @ np.diag(principal) @ rot.T


def probe_configurations(ranges: dict[str, tuple[float, float]], n_random: int = 4,
                         seed: int = 0) -> list[dict[str, float]]:
    """Midpoint of every joint range plus seeded random configurations inside the limits."""
    rng = np.random.default_rng(seed)
    names = sorted(ranges)
    configs = [{n: 0.5 * (ranges[n][0] + ranges[n][1]) for n in names}]
    for _ in range(n_random):
        configs.append({n: float(rng.uniform(*ranges[n])) for n in names})
    return configs


def _rotation_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius norm of the rotation-matrix difference (the article's FK metric).

    Unlike arccos(|a.b|), this stays accurate for float32 quaternions.
    """
    return float(np.linalg.norm(quat_to_mat(a) - quat_to_mat(b)))


def compare(ref: dict, other: dict) -> dict:
    """Worst-case differences of `other` against `ref` (both ModelReadback dicts)."""
    report: dict = {"route": other["route"], "reference": ref["route"], "missing": []}
    worst = dict.fromkeys(
        ["mass", "com", "inertia", "joint_axis", "joint_range", "armature", "damping",
         "frictionloss", "fk_pos", "fk_rot"], 0.0)
    where: dict[str, str] = {}

    def track(key: str, value: float, label: str) -> None:
        if value > worst[key]:
            worst[key] = value
            where[key] = label

    for name, rb in ref["bodies"].items():
        ob = other["bodies"].get(name)
        if ob is None:
            report["missing"].append(f"body:{name}")
            continue
        track("mass", abs(ob["mass"] - rb["mass"]), name)
        track("com", float(np.max(np.abs(np.subtract(ob["com"], rb["com"])))), name)
        track("inertia", float(np.max(np.abs(np.subtract(ob["inertia"], rb["inertia"])))), name)

    for name, rj in ref["joints"].items():
        oj = other["joints"].get(name)
        if oj is None:
            report["missing"].append(f"joint:{name}")
            continue
        track("joint_axis", float(np.max(np.abs(np.subtract(oj["axis"], rj["axis"])))), name)
        track("joint_range", float(np.max(np.abs(np.subtract(oj["range"], rj["range"])))), name)
        for key in ("armature", "damping", "frictionloss"):
            track(key, abs(oj[key] - rj[key]), name)

    for i, (rp, op) in enumerate(zip(ref["fk_poses"], other["fk_poses"], strict=True)):
        for name, rpose in rp.items():
            opose = op.get(name)
            if opose is None:
                continue
            track("fk_pos", float(np.linalg.norm(np.subtract(opose[:3], rpose[:3]))),
                  f"config {i} {name}")
            track("fk_rot", _rotation_difference(np.array(opose[3:]), np.array(rpose[3:])),
                  f"config {i} {name}")

    report["max_abs_diff"] = worst
    report["worst_at"] = where
    return report
