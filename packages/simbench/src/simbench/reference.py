"""Convert the article's published replay bundles into Traces for like-for-like metrics."""

import gzip
import json
from functools import cache

import numpy as np

from simbench.paths import REFERENCE
from simbench.trace import Trace

# Paths under reference/article/.
BUNDLES = {
    "slide": "replay/sliding-block-20260921.json.gz",
    "drop": "replay/drop-20260921.json.gz",
    "hinge": "replay/hinge-20260921.json.gz",
    "collision-chain": "replay/collision-chain-20260921.json.gz",
    "panda-effort": "replay/panda-20260921.json.gz",
    "sweep": "assets/sweep-comparison/repeat-1.json.gz",
    "insertion": "assets/insertion/replay-c100-tilted-r0.json.gz",
}
INSERTION_GUIDED = "assets/insertion/replay-guided-tilted-contact_guided.json.gz"
DEFORM = {"compression": "assets/deformable/compression.json.gz",
          "grasp-1N": "assets/deformable/grasp-1N.json.gz",
          "grasp-2N": "assets/deformable/grasp-2N.json.gz"}
# The article's sweep "Repeat 1" and "Repeat 2"; Repeat 3 is not published.
SWEEP_REPEATS = {1: "assets/sweep-comparison/repeat-1.json.gz",
                 2: "assets/sweep-comparison/repeat-2.json.gz"}


@cache
def load_bundle(rel: str) -> dict:
    with gzip.open(REFERENCE / "article" / rel) as f:
        return json.load(f)
ENGINE_LABELS = {"physx": "PhysX (article)", "newton_mjwarp": "Newton 1.5.2 (article)",
                 "mujoco_cpu": "MuJoCo 3.11 (article)", "genesis": "Genesis 1.4.1 (article)",
                 "mujoco_warp": "MuJoCo-Warp (article)"}


def _frame_bodies(frame: dict) -> list[dict]:
    if "bodies" in frame:
        return frame["bodies"]
    return [{"id": "cube", **{k: frame[k] for k in frame if k != "time_s"}}]


def _insertion_meta(tr: dict, path: str, dt: float) -> dict:
    """Our scene parameters for the bundle's case; the peg is the article's `paddle` body.

    Stopped runs hold their last recorded pose to 4 s, so the status carries the stop.
    """
    from simbench.scenes.insertion import DEPTH_PASS, DURATION, PASS_WINDOW, peg_tip_hand

    case = "tilted" if "tilted" in path else "offset" if "offset" in path else "aligned"
    names = ["peg" if b["id"] == "paddle" else b["id"] for b in tr["frames"][0]["bodies"]]
    return {"bodies": names, "status": "ok" if tr["status"] == "complete" else tr["status"],
            "params": {"task": "insertion", "dt": dt, "duration": DURATION, "case": case,
                       "clearance_mm": 0.10, "guided": "guided" in path,
                       "bore_origin": tr["frames"][0]["bodies"][names.index("bore_000")][
                           "position_m"],
                       "peg_tip_hand": peg_tip_hand(case).tolist(),
                       "depth_pass_m": DEPTH_PASS, "pass_window_s": PASS_WINDOW}}


def article_traces(scene: str, bundle_path: str | None = None) -> dict[str, Trace]:
    """One Trace per engine in the article bundle (poses, contact net forces, torques).

    `position_m` is the body origin; the gravity force point, where recorded, is the COM.
    """
    bundle = load_bundle(bundle_path or BUNDLES[scene])
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
        params = {k: v for k, v in tr["scene"].items() if k != "bodies"}
        if scene == "hinge":
            pm = params["physical_model"]
            params = {**pm, "torque": 0.2, "pulse": (0.1, 0.2)}
        meta = {"route": f"article:{tr['engine']}", "label": ENGINE_LABELS[tr["engine"]],
                "engine_version": tr.get("engine_version"), "scene": scene,
                "dt": float(t[1] - t[0]), "bodies": names, "params": params}
        if scene == "insertion":
            meta |= _insertion_meta(tr, bundle_path or BUNDLES[scene], meta["dt"])
        if scene == "sweep":
            from simbench.scenes.sweep import layout

            meta["params"] |= {"tray": layout()["tray"], "table_top": layout()["table_top"]}
        out[tr["engine"]] = Trace(meta, {"t": t, "pos": pos, "com": com, "quat": quat,
                                         "linvel": lin, "angvel": ang, "q": q, "tau": tau,
                                         "contact_net": net})
    return out


DEFORM_ENGINES = {"mujoco_cpu": "mujoco_cpu", "mujoco_warp_native": "mujoco_warp"}


def article_deform(key: str) -> dict[str, dict]:
    """Headline metrics of the article's MuJoCo flex traces, on our metric windows.

    The bundles hold vertices and plate forces rather than bodies, so they are scored here
    instead of being converted to Traces. Other backends use different solvers (FEM, VBD).
    """
    from simbench.scenes.deform import HOLD_WINDOW, ball_mesh

    out = {}
    for tr in load_bundle(DEFORM[key])["traces"]:
        if tr["backend"] not in DEFORM_ENGINES:
            continue
        frames = tr["frames"]
        t = np.array([f["t"] for f in frames])
        v = np.array([f["v"] for f in frames])
        fail = tr["failure"]
        m = {"completed": fail is None, "invalid_at_s": fail and fail["time_s"]}
        if key == "compression":
            hold = (t >= 0.6 - 1e-9) & (t <= 0.8 + 1e-9)
            load = np.array([abs(f["forces"][1][2]) for f in frames])
            extent = np.ptp(v[:, :, 2], axis=1)
            m |= {"hold_plate_load_N": float(load[hold].mean()),
                  "hold_extent_mm": float(extent[hold].mean() * 1e3),
                  "final_extent_mm": float(extent[-1] * 1e3)}
        else:
            mass = ball_mesh()["mass"]
            lift = (v[:, :, 2] - v[0, :, 2]) @ mass / mass.sum() * 1e3
            t0, t1 = HOLD_WINDOW
            hold = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
            valid = hold & (t < (fail["time_s"] if fail else np.inf))
            m |= {"hold_lift_mm": float(lift[hold].mean()) if valid.sum() == hold.sum()
                  and hold.any() else None}
        engine = DEFORM_ENGINES[tr["backend"]]
        out[f"article:{engine}"] = {"status": "published", "versions": {engine: None},
                                    "label": ENGINE_LABELS[engine], "metrics": m}
    return out
