"""Unitree G1 (29-DOF) tracking Cartwheel's basketball_shot motion on contact terrains.

Policy, MJCF and motion come from Hugging Face `andrew-cartwheel/cartwheel-control`
(g1/), Cartwheel Open Model License for weights/motions; G1 model is BSD-3. The loop
mirrors cartwheel-control (Apache-2.0): 50 Hz motion-tracking MLP, 500 Hz PD on all
29 hinges, observation history and goal frames as trained. Simulation runs through
simbench's five engines — not mjlab / cartwheel-play.

Terrains are contact-parameter models: every colliding geom and the ground get the same
friction, solref and solimp.

flat    MuJoCo default contact on a plane
glass   flat, friction 0.1
sand    soft, well-damped contact on 1.5 cm ripples every 0.3 m
mud     very soft, heavily overdamped contact, high friction
sponge  soft, underdamped contact on random bumps up to 2.5 cm
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from safetensors import safe_open

from simbench.models import _fmt
from simbench.paths import CARTWHEEL_G1
from simbench.scene import SCENE_DIR, FreeInit, Scene

POLICY_HZ = 50.0
DAMPING_RATIO = 2.0
PD_FREQ_HZ = 10.0
FALL_HEIGHT = 0.25
FALL_TILT = 0.8
FALL_HOLD_S = 1.0
MAX_SECONDS = 20.0
# Terrain features span a wide patch so the viewer reads the material immediately.
TRACK_X = (-2.0, 18.0)
TRACK_Y = (-4.0, 4.0)
FLOOR_HALF = (22.0, 10.0)  # plane half-extents [m] for the viewer

# Cartwheel G1 default pose (robots.py); joint order matches config.json.
DEFAULT_JOINT_POS = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0, 0, 0, 0,
    0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0,
], dtype=np.float64)

# Contact params + viewer look (rgba). Uneven features are denser and cover TRACK_*.
TERRAINS = {
    "flat": {"friction": 1.0, "solref": "0.02 1", "solimp": "0.9 0.95 0.001 0.5 2",
             "rgba": "0.55 0.56 0.58 1", "look": "concrete"},
    "glass": {"friction": 0.1, "solref": "0.02 1", "solimp": "0.9 0.95 0.001 0.5 2",
              "rgba": "0.55 0.75 0.90 0.55", "look": "glass",
              "tiles": (1.0, 0.012)},
    "sand": {"friction": 0.6, "solref": "0.05 1.2", "solimp": "0.3 0.95 0.02 0.5 2",
             "rgba": "0.86 0.70 0.42 1", "look": "sand",
             "ripples": (0.4, 0.025)},
    "mud": {"friction": 1.2, "solref": "0.05 2", "solimp": "0.6 0.95 0.03 0.5 2",
            "rgba": "0.28 0.18 0.10 1", "look": "mud",
            "puddles": (0.9, 0.04)},
    "sponge": {"friction": 0.9, "solref": "0.05 0.15", "solimp": "0.3 0.95 0.02 0.5 2",
               "rgba": "0.92 0.45 0.55 1", "look": "sponge",
               "bumps": (0.45, 0.045)},
}

FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def g1_dir() -> Path:
    return CARTWHEEL_G1


def config() -> dict:
    return json.loads((g1_dir() / "config.json").read_text())


def joints() -> list[str]:
    return list(config()["joints"])


def motion_path(name: str = "basketball_shot") -> Path:
    return g1_dir() / "motions" / f"{name}.npz"


def load_motion(name: str = "basketball_shot") -> np.ndarray:
    """Reference qpos (frames, 7 + nj) at 50 Hz."""
    z = np.load(motion_path(name))
    q = np.asarray(z["qpos"], np.float64)
    fps = float(z["fps"]) if "fps" in z.files else POLICY_HZ
    if abs(fps - POLICY_HZ) > 1e-6:
        raise ValueError(f"{name}: expected {POLICY_HZ:g} Hz, got {fps:g}")
    nj = len(joints())
    if q.ndim != 2 or q.shape[1] != 7 + nj:
        raise ValueError(f"{name}: qpos shape {q.shape}, expected (frames, {7 + nj})")
    q = q.copy()
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True).clip(1e-8)
    return q


# --- quaternions (wxyz) --------------------------------------------------------------

def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qapply(q, v):
    w, xyz = q[0], q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _qapply_inv(q, v):
    return _qapply(_qconj(q), v)


def _qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _q6d(q):
    return _qmat(q)[:, :2].reshape(6)


def _qyaw(q):
    w, x, y, z = q
    a = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def _qrotvec(q):
    q = q * (-1.0 if q[0] < 0 else 1.0)
    xyz, s = q[1:], np.linalg.norm(q[1:])
    angle = 2.0 * math.atan2(s, q[0])
    k = (2.0 + angle ** 2 / 12.0) if s < 1e-6 else angle / s
    return xyz * k


def _motion_vel(qpos: np.ndarray, i: int) -> np.ndarray:
    """qvel (6 + nj): world linear, root-frame angular, joint velocities."""
    n = len(qpos)
    lo, hi = max(0, i - 1), min(n - 1, i + 1)
    span = max(hi - lo, 1) / POLICY_HZ
    qp, qn = qpos[lo], qpos[hi]
    lin = (qn[:3] - qp[:3]) / span
    ang = _qrotvec(_qmul(_qconj(qp[3:7]), qn[3:7])) / span
    jv = (qn[7:] - qp[7:]) / span
    return np.concatenate([lin, ang, jv])


# --- policy (numpy MLP, Cartwheel architecture) --------------------------------------

def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class RunningNorm:
    def __init__(self, mean, var, eps=1e-2):
        self.mean, self.var, self.eps = mean.astype(np.float64), var.astype(np.float64), eps

    def __call__(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.eps ** 2)


class TrackingPolicy:
    """Cartwheel Policy: goal encoder -> latent, decoder(latent + obs history) -> action."""

    def __init__(self, weights: dict[str, np.ndarray], num_joints: int):
        self.w = weights
        self.goal_norm = RunningNorm(weights["goal_norm.mean"], weights["goal_norm.var"])
        self.obs_norm = RunningNorm(weights["obs_norm.mean"], weights["obs_norm.var"])
        self.num_joints = num_joints

    def _mlp(self, prefix: str, x: np.ndarray) -> np.ndarray:
        # encoder/decoder: Linear-ELU × 4 then Linear
        for i in range(0, 8, 2):
            x = _elu(self.w[f"{prefix}.{i}.weight"] @ x + self.w[f"{prefix}.{i}.bias"])
        return self.w[f"{prefix}.8.weight"] @ x + self.w[f"{prefix}.8.bias"]

    def __call__(self, obs_hist: np.ndarray, goal: np.ndarray) -> np.ndarray:
        z = np.tanh(self._mlp("encoder", self.goal_norm(goal)))
        flat = self.obs_norm(obs_hist.reshape(-1))
        return self._mlp("decoder", np.concatenate([z, flat]))


def load_policy() -> tuple[TrackingPolicy, dict]:
    cfg = config()
    path = g1_dir() / "policy.safetensors"
    weights = {}
    with safe_open(str(path), framework="np") as f:
        for k in f.keys():  # noqa: SIM118 — safetensors SafeOpen is not a Mapping
            weights[k] = np.array(f.get_tensor(k), dtype=np.float64)
    return TrackingPolicy(weights, len(cfg["joints"])), cfg


def pd_gains(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """kp, kd, effort, action_scale from MJCF armature / actuatorfrcrange (Cartwheel RP1 rule)."""
    tree = ET.parse(g1_dir() / "model" / "g1.xml")
    by_name = {j.get("name"): j for j in tree.iter("joint") if j.get("name")}
    w = PD_FREQ_HZ * 2.0 * math.pi
    kp, kd, effort, scale = [], [], [], []
    for name in joint_names:
        j = by_name[name]
        arm = float(j.get("armature", "0.01"))
        lo, hi = (float(v) for v in j.get("actuatorfrcrange", "-1 1").split())
        eff = max(abs(lo), abs(hi))
        k = arm * w ** 2
        kp.append(k)
        kd.append(2.0 * DAMPING_RATIO * arm * w)
        effort.append(eff)
        scale.append(0.25 * eff / k)
    return (np.asarray(kp), np.asarray(kd), np.asarray(effort), np.asarray(scale))


# --- controller ----------------------------------------------------------------------

class Tracker:
    """Cartwheel 50 Hz tracking policy + PD on the 29 actuated hinges."""

    uses_state = True

    def __init__(self, motion: str, dt: float, joint_names: list[str]):
        self.policy, cfg = load_policy()
        if cfg["joints"] != joint_names:
            raise ValueError("policy joint order differs from scene joints")
        self.hist = int(cfg["hist"])
        self.goal_frames = int(cfg["goal_frames"])
        self.qpos = load_motion(motion)
        self.nj = len(joint_names)
        self.q0 = DEFAULT_JOINT_POS.copy()
        self.kp, self.kd, self.effort, self.scale = pd_gains(joint_names)
        self.dt = dt
        self.decimation = int(round(1.0 / (POLICY_HZ * dt)))
        self.n_frames = len(self.qpos)
        self.end_s = min((self.n_frames - 1) / POLICY_HZ, MAX_SECONDS)
        self.fell_at: float | None = None
        self.reset()

    def reset(self) -> None:
        self.action = np.zeros(self.nj, np.float64)
        self.target = self.q0.copy()
        self.ref_i = 0
        self.fell_at = None
        proprio0 = self._proprio(self.qpos[0], _motion_vel(self.qpos, 0), self.action, 0)
        self.obs_hist = np.tile(proprio0, (self.hist, 1))

    def _clamp_ref(self, i: int) -> int:
        return int(np.clip(i, 0, self.n_frames - 1))

    def _proprio(self, root_q, root_qd, a_prev, ref_i) -> np.ndarray:
        q_root, w = root_q[3:7], root_qd[3:6]
        grav = _qapply_inv(q_root, np.array([0.0, 0.0, -1.0]))
        ref_root = self.qpos[self._clamp_ref(ref_i)][3:7]
        rel = _q6d(_qmul(_qconj(q_root), ref_root))
        return np.concatenate([root_q[7:] - self.q0, root_qd[6:], w, grav, a_prev, rel])

    def _goal(self, ref_i: int) -> np.ndarray:
        cq = self.qpos[self._clamp_ref(ref_i)]
        heading = _qyaw(cq[3:7])
        feats = []
        for k in range(1, self.goal_frames + 1):
            i = self._clamp_ref(ref_i + k)
            rq, rv = self.qpos[i], _motion_vel(self.qpos, i)
            root = rq[3:7]
            feats.append(np.concatenate([
                _qapply_inv(heading, rq[:3] - cq[:3]),
                _q6d(_qmul(_qconj(heading), root)),
                _qapply_inv(root, rv[:3]),
                rv[3:6],
                rq[7:],
                0.1 * rv[6:],
            ]))
        return np.concatenate(feats)

    def _state_qpos_qvel(self, q, qd, state) -> tuple[np.ndarray, np.ndarray]:
        """Full free+hinge qpos/qvel; angvel from world body rate into root frame."""
        pos, quat = state["pos"][0], state["quat"][0]
        ang_w, lin = state["angvel"][0], state["linvel"][0]
        ang_b = _qmat(quat).T @ ang_w
        root_q = np.concatenate([pos, quat, q])
        root_qd = np.concatenate([lin, ang_b, qd])
        return root_q, root_qd

    def _fell(self, root_q: np.ndarray, ref_i: int) -> bool:
        rq = self.qpos[self._clamp_ref(ref_i)]
        g = np.array([0.0, 0.0, -1.0])
        e_tilt = abs(_qapply_inv(rq[3:7], g)[2] - _qapply_inv(root_q[3:7], g)[2])
        return abs(rq[2] - root_q[2]) > FALL_HEIGHT or e_tilt > FALL_TILT

    def __call__(self, t, q, qd, state) -> np.ndarray:
        k = int(round(t / self.dt))
        if k == 0:
            self.reset()
        root_q, root_qd = self._state_qpos_qvel(q, qd, state)
        if self.fell_at is None and self._fell(root_q, self.ref_i):
            self.fell_at = t
        if k > 0 and k % self.decimation == 0 and t <= self.end_s:
            self.ref_i = self._clamp_ref(int(round(t * POLICY_HZ)))
            self.action = self.policy(self.obs_hist, self._goal(self.ref_i))
            self.target = self.q0 + self.scale * self.action
            proprio = self._proprio(root_q, root_qd, self.action, self.ref_i)
            self.obs_hist = np.vstack([proprio, self.obs_hist[:-1]])
        tau = self.kp * (self.target - q) - self.kd * qd
        return np.clip(tau, -self.effort, self.effort)

    def observe(self, t, contacts, names):
        if t >= self.end_s + (FALL_HOLD_S if self.fell_at is not None else 0.0):
            return ("ok", None)
        if self.fell_at is not None and t >= self.fell_at + FALL_HOLD_S:
            return ("ok", None)
        return None


# --- scene ---------------------------------------------------------------------------

def _terrain(world, name: str, attrs: dict) -> None:
    """Large, visually distinct ground + uneven coverage for the named material."""
    spec = TERRAINS[name]
    rgba = spec["rgba"]
    look = spec.get("look", name)
    fx, fy = FLOOR_HALF
    ET.SubElement(world, "geom", name=f"floor_{look}", type="plane",
                  size=_fmt([fx, fy, 0.05]), rgba=rgba, **attrs)
    x0, x1 = TRACK_X
    y0, y1 = TRACK_Y
    if "tiles" in spec:  # glass: raised wet tiles over a wide strip
        spacing, lift = spec["tiles"]
        half = spacing * 0.42
        for i, x in enumerate(np.arange(x0 + spacing, x1, spacing)):
            for j, y in enumerate(np.arange(y0 + spacing, y1, spacing)):
                ET.SubElement(world, "geom", name=f"tile{i}_{j}", type="box",
                              pos=_fmt([x, y, lift]), size=_fmt([half, half, lift]),
                              rgba="0.65 0.82 0.95 0.7", **attrs)
    if "ripples" in spec:  # sand dunes: long capsules across the full width
        spacing, height = spec["ripples"]
        radius = max(0.05, height * 2.2)
        for i, x in enumerate(np.arange(x0 + spacing * 0.5, x1, spacing)):
            ET.SubElement(world, "geom", name=f"ripple{i}", type="capsule",
                          fromto=_fmt([x, y0 + 0.2, height - radius,
                                       x, y1 - 0.2, height - radius]),
                          size=f"{radius}", rgba="0.78 0.60 0.32 1", **attrs)
    if "puddles" in spec:  # mud: irregular low boxes + soft mounds
        spacing, height = spec["puddles"]
        rng = np.random.default_rng(1)
        for i, x in enumerate(np.arange(x0 + 0.3, x1, spacing)):
            for j, y in enumerate(np.arange(y0 + 0.3, y1, spacing)):
                sx = rng.uniform(0.18, 0.38)
                sy = rng.uniform(0.18, 0.38)
                h = rng.uniform(0.4, 1.0) * height
                ET.SubElement(world, "geom", name=f"puddle{i}_{j}", type="box",
                              pos=_fmt([x + rng.uniform(-0.08, 0.08),
                                        y + rng.uniform(-0.08, 0.08), h]),
                              size=_fmt([sx, sy, h]), rgba="0.22 0.14 0.08 1", **attrs)
                if (i + j) % 3 == 0:
                    r = rng.uniform(0.08, 0.14)
                    ET.SubElement(world, "geom", name=f"mound{i}_{j}", type="sphere",
                                  pos=_fmt([x + 0.15, y - 0.1, r * 0.6]),
                                  size=f"{r}", rgba="0.34 0.22 0.12 1", **attrs)
    if "bumps" in spec:  # sponge: dense soft spheres
        spacing, height = spec["bumps"]
        radius = 0.12
        rng = np.random.default_rng(0)
        for i, x in enumerate(np.arange(x0 + 0.3, x1, spacing)):
            for j, y in enumerate(np.arange(y0 + 0.3, y1, spacing)):
                h = rng.uniform(0.35, 1.0) * height
                ET.SubElement(world, "geom", name=f"bump{i}_{j}", type="sphere",
                              pos=_fmt([x + rng.uniform(-0.06, 0.06),
                                        y + rng.uniform(-0.06, 0.06), h - radius * 0.35]),
                              size=f"{radius * rng.uniform(0.85, 1.2)}",
                              rgba="0.95 0.55 0.62 1", **attrs)


def _body_names(root) -> list[str]:
    return [b.get("name") for b in root.iter("body") if b.get("name")]


def mjcf(terrain: str, dt: float, q0: np.ndarray) -> str:
    """Cartwheel g1.xml with terrain contact and free-joint pose from motion frame 0."""
    spec = TERRAINS[terrain]
    attrs = {"friction": f"{spec['friction']} 0.005 0.0001", "solref": spec["solref"],
             "solimp": spec["solimp"]}
    tree = ET.parse(g1_dir() / "model" / "g1.xml")
    root = tree.getroot()
    root.set("model", f"g1_{terrain}")
    meshdir = os.path.relpath(g1_dir() / "model" / "assets", SCENE_DIR)
    compiler = root.find("compiler")
    compiler.set("meshdir", Path(meshdir).as_posix() + "/")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", repr(dt))
    option.set("integrator", "Euler")
    # Keep Cartwheel visual STL meshes (group 2) for the viewer; collision capsules stay for
    # contact. Apply terrain contact params only to colliding geoms.
    for geom in root.iter("geom"):
        if geom.get("contype", "1") != "0":
            geom.attrib.update(attrs)
    world = root.find("worldbody")
    pelvis = world.find("body")  # floating base
    pelvis.set("pos", _fmt(q0[:3]))
    pelvis.set("quat", _fmt(q0[3:7]))
    for light in list(world.findall("light")):
        world.remove(light)
    ET.SubElement(world, "light", pos="1 0 3.5", dir="0 0 -1", directional="true")
    _terrain(world, terrain, attrs)
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode")


def robot_mass() -> float:
    tree = ET.parse(g1_dir() / "model" / "g1.xml")
    return round(sum(float(i.get("mass")) for i in tree.iter("inertial")), 4)


def walk(dt: float = 0.002, terrain: str = "flat", motion: str = "basketball_shot",
         duration: float | None = None) -> Scene:
    names = joints()
    qpos = load_motion(motion)
    q0 = qpos[0]
    v0 = _motion_vel(qpos, 0)
    tracker = Tracker(motion, dt, names)
    if duration is None:
        duration = tracker.end_s + FALL_HOLD_S
    # World-frame angular for FreeInit (adapters convert as needed).
    ang_world = _qmat(q0[3:7]) @ v0[3:6]
    bodies = _body_names(ET.parse(g1_dir() / "model" / "g1.xml").getroot())
    return Scene(
        name="g1-walk", mjcf=mjcf(terrain, dt, q0), dt=dt, duration=duration,
        bodies=bodies, joints=names,
        joint_init=dict(zip(names, q0[7:].tolist(), strict=True)),
        free_init={"pelvis": FreeInit(linear=tuple(v0[:3]), angular=tuple(ang_world))},
        controller=tracker, record_dt=0.02,
        params={"terrain": terrain, "motion": motion, "mass": robot_mass(),
                "policy": "cartwheel-control g1/policy.safetensors",
                "policy_credit": (
                    "Cartwheel Control — https://github.com/Cartwhl/cartwheel-control ; "
                    "weights https://huggingface.co/andrew-cartwheel/cartwheel-control ; "
                    "Cartwheel Open Model License (NOTICE: Licensed by Cartwheel under the "
                    "Cartwheel Open Model License)."
                ),
                "policy_hz": POLICY_HZ, "n_frames": len(qpos),
                "ref_qpos": qpos[:: max(1, len(qpos) // 250)].round(5).tolist(),
                **{k: v for k, v in TERRAINS[terrain].items()
                   if k not in ("rgba", "look")}},
    )
