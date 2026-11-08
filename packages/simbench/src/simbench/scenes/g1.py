"""Unitree G1 (12-DOF legs) walking on different terrains with Unitree's pretrained policy.

The policy (`deploy/pre_train/g1/motion.pt` of unitree_rl_gym, BSD-3) is a 47-input
LSTM(64) + MLP(32, ELU) trained in Isaac Gym. It runs here exactly as in Unitree's
MuJoCo deploy script: 50 Hz policy, 500 Hz PD on the 12 leg joints, a 0.8 s gait clock
and a constant velocity command. The weights are read from the TorchScript archive
with numpy, so no route needs torch.

Each foot touches the ground through four 5 mm spheres; the pelvis and leg links keep
Unitree's collision meshes, which reach the ground only when the feet sink deep.
Terrains are contact-parameter models: every colliding geom and the ground get the same
friction, solref and solimp, so every engine's combination rule yields the same pair
values.

flat    MuJoCo default contact on a plane
glass   flat, friction 0.1
sand    soft, well-damped contact on 1.5 cm ripples every 0.3 m
mud     very soft, heavily overdamped contact, high friction (sinks, little rebound)
sponge  soft, underdamped contact on random bumps up to 2.5 cm (springy and uneven)
"""

import json
import os
import pickle
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from simbench.models import _fmt
from simbench.paths import UNITREE
from simbench.scene import SCENE_DIR, Scene

JOINTS = [f"{side}_{name}_joint" for side in ("left", "right")
          for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
BODIES = ["pelvis"] + [f"{side}_{name}_link" for side in ("left", "right")
                       for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch",
                                    "ankle_roll")]
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
TORQUE_LIMIT = np.array([88, 139, 88, 139, 50, 50] * 2, dtype=float)
GAIT_PERIOD = 0.8
TRACK = (0.0, 8.0)  # terrain features span x in this range [m]

TERRAINS = {
    "flat": {"friction": 1.0, "solref": "0.02 1", "solimp": "0.9 0.95 0.001 0.5 2"},
    "glass": {"friction": 0.1, "solref": "0.02 1", "solimp": "0.9 0.95 0.001 0.5 2"},
    "sand": {"friction": 0.6, "solref": "0.05 1.2", "solimp": "0.3 0.95 0.02 0.5 2",
             "ripples": (0.3, 0.015)},
    "mud": {"friction": 1.2, "solref": "0.05 2", "solimp": "0.6 0.95 0.03 0.5 2"},
    "sponge": {"friction": 0.9, "solref": "0.05 0.15", "solimp": "0.3 0.95 0.02 0.5 2",
               "bumps": (0.25, 0.025)},
}


# --- policy --------------------------------------------------------------------------

def load_torchscript_weights(path) -> dict[str, np.ndarray]:
    """Tensors of a TorchScript archive by attribute path, read without torch."""
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    dtypes = {"FloatStorage": np.float32, "DoubleStorage": np.float64}

    class Module:  # also stands in for TorchScript helpers such as restore_type_tag
        def __init__(self, *args, **kwargs):
            self.state = None

        def __setstate__(self, state):
            self.state = state

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return lambda st, off, size, stride, *_: np.lib.stride_tricks.as_strided(
                    st[off:], size, [s * st.itemsize for s in stride]).copy()
            if name.endswith("Storage"):
                return name
            return type(name, (Module,), {})

        def persistent_load(self, pid):
            _, kind, key, _, _ = pid
            return np.frombuffer(z.read(f"{root}/data/{key}"), dtypes[kind])

    out = {}

    def walk(obj, prefix):
        state = getattr(obj, "state", obj)
        if isinstance(state, np.ndarray):
            out[prefix] = state
        elif isinstance(state, dict):
            for k, v in state.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))

    walk(Unpickler(z.open(f"{root}/data.pkl")).load(), "")
    return out


class LstmPolicy:
    """PyTorch nn.LSTM (gates i, f, g, o) followed by Linear-ELU-Linear."""

    def __init__(self, w: dict[str, np.ndarray]):
        self.w_ih, self.w_hh = w["memory.weight_ih_l0"], w["memory.weight_hh_l0"]
        self.b = w["memory.bias_ih_l0"] + w["memory.bias_hh_l0"]
        self.l0 = (w["actor.0.weight"], w["actor.0.bias"])
        self.l2 = (w["actor.2.weight"], w["actor.2.bias"])
        self.hidden = self.w_hh.shape[1]
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros(self.hidden, np.float32)
        self.c = np.zeros(self.hidden, np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        def sig(x):
            return 1 / (1 + np.exp(-x))

        gates = self.w_ih @ obs + self.w_hh @ self.h + self.b
        i, f, g, o = np.split(gates, 4)
        self.c = sig(f) * self.c + sig(i) * np.tanh(g)
        self.h = sig(o) * np.tanh(self.c)
        x = self.l0[0] @ self.h + self.l0[1]
        x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return self.l2[0] @ x + self.l2[1]


def deploy_config() -> dict:
    """The flat `key: value` deploy YAML (numbers, lists, strings) without a YAML library."""
    cfg, pending = {}, ""
    for line in (UNITREE / "g1.yaml").read_text().splitlines():
        line = (pending + " " + line.split("#")[0]).strip()
        if not line or line.count("[") > line.count("]"):
            pending = line
            continue
        pending = ""
        key, value = (part.strip() for part in line.split(":", 1))
        cfg[key] = json.loads(value)
    return cfg


def _rot(quat) -> np.ndarray:
    w, x, y, z = quat
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Walker:
    """Unitree's deploy loop as a state-feedback controller on the 12 leg joints."""

    uses_state = True

    def __init__(self, command, dt: float):
        cfg = deploy_config()
        self.policy = LstmPolicy(load_torchscript_weights(UNITREE / "motion.pt"))
        self.kp, self.kd = np.array(cfg["kps"], float), np.array(cfg["kds"], float)
        self.default = np.array(cfg["default_angles"], float)
        self.cfg, self.dt = cfg, dt
        self.decimation = int(round(cfg["simulation_dt"] * cfg["control_decimation"] / dt))
        self.command = np.asarray(command, np.float32)
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.action = np.zeros(12, np.float32)
        self.target = self.default.copy()

    def observation(self, t, q, qd, quat, angvel_world) -> np.ndarray:
        c = self.cfg
        rot = _rot(quat)
        phase = (t % GAIT_PERIOD) / GAIT_PERIOD
        return np.concatenate([
            rot.T @ angvel_world * c["ang_vel_scale"],
            rot.T @ np.array([0.0, 0.0, -1.0]),
            self.command * np.array(c["cmd_scale"]),
            (q - self.default) * c["dof_pos_scale"],
            qd * c["dof_vel_scale"],
            self.action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        ]).astype(np.float32)

    def __call__(self, t, q, qd, state) -> np.ndarray:
        k = int(round(t / self.dt))
        if k == 0:
            self.reset()
        elif k % self.decimation == 0:
            obs = self.observation(t, q, qd, state["quat"][0], state["angvel"][0])
            self.action = self.policy(obs)
            self.target = self.default + self.cfg["action_scale"] * self.action
        tau = self.kp * (self.target - q) - self.kd * qd
        return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


# --- scene ---------------------------------------------------------------------------

def _terrain(world, name: str, attrs: dict) -> None:
    spec = TERRAINS[name]
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  rgba="0.5 0.5 0.5 1", **attrs)
    x0, x1 = TRACK
    if "ripples" in spec:
        spacing, height = spec["ripples"]
        radius = 0.06
        for i, x in enumerate(np.arange(x0 + spacing, x1, spacing)):
            ET.SubElement(world, "geom", name=f"ripple{i}", type="capsule",
                          fromto=_fmt([x, -1, height - radius, x, 1, height - radius]),
                          size=f"{radius}", rgba="0.85 0.75 0.5 1", **attrs)
    if "bumps" in spec:
        spacing, height = spec["bumps"]
        radius = 0.15
        rng = np.random.default_rng(0)
        for i, x in enumerate(np.arange(x0 + 0.4, x1, spacing)):
            for j, y in enumerate(np.arange(-0.5, 0.51, spacing)):
                h = rng.uniform(0.3, 1.0) * height
                ET.SubElement(world, "geom", name=f"bump{i}_{j}", type="sphere",
                              pos=_fmt([x + rng.uniform(-0.05, 0.05), y, h - radius]),
                              size=f"{radius}", rgba="0.9 0.55 0.6 1", **attrs)


def mjcf(terrain: str, dt: float) -> str:
    spec = TERRAINS[terrain]
    attrs = {"friction": f"{spec['friction']} 0.005 0.0001", "solref": spec["solref"],
             "solimp": spec["solimp"]}
    tree = ET.parse(UNITREE / "g1_12dof.xml")
    root = tree.getroot()
    root.set("model", f"g1_{terrain}")
    meshdir = os.path.relpath(UNITREE / "meshes", SCENE_DIR)
    root.find("compiler").set("meshdir", Path(meshdir).as_posix() + "/")
    ET.SubElement(root, "option", timestep=repr(dt))
    for geom in root.iter("geom"):
        if geom.get("contype") != "0":
            geom.attrib.update(attrs)
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="1 0 3.5", dir="0 0 -1", directional="true")
    _terrain(world, terrain, attrs)
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode")


def robot_mass() -> float:
    tree = ET.parse(UNITREE / "g1_12dof.xml")
    return round(sum(float(i.get("mass")) for i in tree.iter("inertial")), 4)


def walk(dt: float = 0.002, terrain: str = "flat", speed: float = 0.5,
         duration: float = 10.0) -> Scene:
    cfg = deploy_config()
    walker = Walker([speed, 0.0, 0.0], dt)
    return Scene(
        name="g1-walk", mjcf=mjcf(terrain, dt), dt=dt, duration=duration,
        bodies=BODIES, joints=JOINTS, joint_init=dict(zip(JOINTS, cfg["default_angles"],
                                                          strict=True)),
        controller=walker, record_dt=0.02,
        params={"terrain": terrain, "speed": speed, **TERRAINS[terrain], "mass": robot_mass(),
                "policy": "unitree_rl_gym deploy/pre_train/g1/motion.pt"},
    )
