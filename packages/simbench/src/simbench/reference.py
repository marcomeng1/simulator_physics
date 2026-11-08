"""Convert the article's published replay bundles into Traces for like-for-like metrics."""

import gzip
import json

import numpy as np

from simbench.paths import REFERENCE
from simbench.trace import Trace

BUNDLES = {
    "slide": "sliding-block-20260921.json.gz",
    "drop": "drop-20260921.json.gz",
    "hinge": "hinge-20260921.json.gz",
    "collision-chain": "collision-chain-20260921.json.gz",
    "panda-effort": "panda-20260921.json.gz",
}
ENGINE_LABELS = {"physx": "PhysX (article)", "newton_mjwarp": "Newton 1.5.2 (article)",
                 "mujoco_cpu": "MuJoCo 3.11 (article)", "genesis": "Genesis 1.4.1 (article)"}


def _frame_bodies(frame: dict) -> list[dict]:
    if "bodies" in frame:
        return frame["bodies"]
    return [{"id": "cube", **{k: frame[k] for k in frame if k != "time_s"}}]


def article_traces(scene: str) -> dict[str, Trace]:
    """One Trace per engine in the article bundle (poses, contact net forces, torques).

    `position_m` is the body origin; the gravity force point, where recorded, is the COM.
    """
    path = REFERENCE / "article" / "replay" / BUNDLES[scene]
    with gzip.open(path) as f:
        bundle = json.load(f)
    out = {}
    for tr in bundle["traces"]:
        frames = tr["frames"]
        names = [b["id"] for b in _frame_bodies(frames[0])]
        n, nb = len(frames), len(names)
        pos, com, quat = np.zeros((n, nb, 3)), np.zeros((n, nb, 3)), np.zeros((n, nb, 4))
        lin, ang, net = np.zeros((n, nb, 3)), np.zeros((n, nb, 3)), np.zeros((n, nb, 3))
        q = np.zeros((n, 1 if scene == "hinge" else 0))
        tau = np.array([fr.get("joint_torque_Nm", []) for fr in frames])
        for k, fr in enumerate(frames):
            for i, b in enumerate(_frame_bodies(fr)):
                pos[k, i] = com[k, i] = b["position_m"]
                quat[k, i] = b["quaternion_wxyz"]
                lin[k, i] = b.get("velocity_m_s", [0, 0, 0])
                ang[k, i] = b.get("angular_velocity_rad_s", [0, 0, 0])
                for f in b.get("forces", []):
                    if f["kind"] == "contact":
                        net[k, i] += f["vector_N"]
                    elif f["kind"] == "gravity":
                        com[k, i] = f["point_m"]
            if scene == "hinge":
                qw, _, qy, _ = fr["bodies"][0]["quaternion_wxyz"]
                q[k, 0] = 2 * np.arctan2(qy, qw)
        t = np.array([fr["time_s"] for fr in frames])
        params = dict(tr["scene"])
        if scene == "hinge":
            pm = params["physical_model"]
            params = {**pm, "torque": 0.2, "pulse": (0.1, 0.2)}
        meta = {"route": f"article:{tr['engine']}", "label": ENGINE_LABELS[tr["engine"]],
                "engine_version": tr.get("engine_version"), "scene": scene,
                "dt": float(t[1] - t[0]), "bodies": names, "params": params}
        out[tr["engine"]] = Trace(meta, {"t": t, "pos": pos, "com": com, "quat": quat,
                                         "linvel": lin, "angvel": ang, "q": q, "tau": tau,
                                         "contact_net": net})
    return out
