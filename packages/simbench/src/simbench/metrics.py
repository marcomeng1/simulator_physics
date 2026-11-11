"""Scene metrics computed from Traces. All values SI unless the key says otherwise."""

import numpy as np

from simbench.readback import quat_to_mat
from simbench.trace import Trace

GRAVITY = np.array([0.0, 0.0, -9.81])


def momentum_residual(tr: Trace, body: str, mass: float) -> np.ndarray:
    """Per-step |m (v_{k+1} - v_k) - (F_contact,k + m g) dt| [N s] for a free body.

    Checks that the recorded contact forces explain the recorded motion; it
    validates the instrumentation, not the physics.
    """
    i = tr.body(body)
    dv = np.diff(tr.linvel[:, i], axis=0)
    impulse = (tr.contact_net[:-1, i] + mass * GRAVITY) * tr.meta["dt"]
    return np.linalg.norm(mass * dv - impulse, axis=1)


def settle_time(tr: Trace, body: str, speed: float = 1e-3) -> float | None:
    """First time after which the COM speed stays below `speed` for the rest of the run."""
    v = np.linalg.norm(tr.linvel[:, tr.body(body)], axis=1)
    moving = np.nonzero(v >= speed)[0]
    if len(moving) == 0:
        return 0.0
    last = moving[-1] + 1
    return float(tr.t[last]) if last < len(tr.t) else None


def lowest_vertex_overlap(tr: Trace, body: str, side: float, floor_z: float = 0.0) -> np.ndarray:
    """Depth of a cube's lowest vertex below the floor plane [m] (positive = overlap)."""
    i = tr.body(body)
    z = tr.com[:, i, 2]
    rz = np.array([np.abs(quat_to_mat(q)[2]).sum() for q in tr.quat[:, i]])
    return floor_z - (z - side / 2 * rz)


def cube_metrics(tr: Trace, mass: float, side: float) -> dict:
    i = tr.body("cube")
    force = np.linalg.norm(tr.contact_net[:, i], axis=1)
    resid = momentum_residual(tr, "cube", mass)
    contact = np.nonzero(force > 0)[0]
    first = int(contact[0]) if len(contact) else None
    out = {
        "frames": len(tr.t),
        "travel_x_m": float(tr.com[-1, i, 0] - tr.com[0, i, 0]),
        "peak_contact_force_N": float(force.max()),
        "mean_final_support_N": float(tr.contact_net[-50:, i, 2].mean()),
        "min_com_z_m": float(tr.com[:, i, 2].min()),
        "max_com_z_m": float(tr.com[:, i, 2].max()),
        "settled_at_s": settle_time(tr, "cube"),
        "max_overlap_m": float(lowest_vertex_overlap(tr, "cube", side).max()),
        "max_momentum_residual_Ns": float(resid.max()),
        "first_contact_s": float(tr.t[first]) if first is not None else None,
    }
    if first is not None:
        window = slice(first, first + int(round(0.04 / tr.meta["dt"])))
        out["impulse_first_40ms_Ns"] = float(tr.contact_net[window, i, 2].sum() * tr.meta["dt"])
    return out


def pendulum_reference(params: dict, t: np.ndarray, substeps: int = 200) -> np.ndarray:
    """Independent nonlinear pendulum solution (RK4) for the hinge scene [rad]."""
    m, length, g = params["mass"], params["length"], params["gravity"]
    lc = length / 2
    i_pivot = params["inertia_com"][1] + m * lc**2
    t0, t1 = params["pulse"]

    def acc(time, th):
        tau = params["torque"] if t0 <= time < t1 else 0.0
        return (tau - m * g * lc * np.sin(th)) / i_pivot

    dt = (t[1] - t[0]) / substeps
    th, om, out, time = params["q0"], 0.0, np.zeros(len(t)), 0.0
    for k in range(len(t)):
        out[k] = th
        for _ in range(substeps):
            k1t, k1o = om, acc(time, th)
            k2t, k2o = om + dt / 2 * k1o, acc(time + dt / 2, th + dt / 2 * k1t)
            k3t, k3o = om + dt / 2 * k2o, acc(time + dt / 2, th + dt / 2 * k2t)
            k4t, k4o = om + dt * k3o, acc(time + dt, th + dt * k3t)
            th += dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
            om += dt / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
            time += dt
    return out


def hinge_metrics(tr: Trace) -> dict:
    ref = pendulum_reference(tr.meta["params"], tr.t)
    err = np.degrees(np.abs(tr.q[:, 0] - ref))
    return {"frames": len(tr.t), "max_reference_error_deg": float(err.max()),
            "final_angle_rad": float(tr.q[-1, 0])}


def hinge_plastic_metrics(tr: Trace) -> dict:
    """Ideal elasto-plasticity: the spring rest angle ends at peak - yield (0 if elastic)."""
    y = tr.meta["params"]["yield_rad"]
    q = tr.q[:, 0]
    expected = max(q.max() - y, 0.0) if y else 0.0
    return {"frames": len(tr.t), "peak_angle_deg": float(np.degrees(q.max())),
            "final_angle_deg": float(np.degrees(q[-1])),
            "expected_final_deg": float(np.degrees(expected)),
            "final_error_deg": float(np.degrees(q[-1] - expected))}


def chain_metrics(tr: Trace) -> dict:
    out = {"frames": len(tr.t)}
    for name in tr.bodies:
        i = tr.body(name)
        out[f"{name}_final_xy_mm"] = (tr.com[-1, i, :2] * 1e3).tolist()
        r = quat_to_mat(tr.quat[-1, i])
        out[f"{name}_final_heading_deg"] = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    out["peak_horizontal_force_N"] = float(np.linalg.norm(tr.contact_net[:, :, :2], axis=2).max())
    return out


# --- Panda scenes --------------------------------------------------------------------

# Original Menagerie arm ranges: monitored, as in the article, not used as a pass gate.
ARM_RANGES = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                       [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                       [-2.8973, 2.8973]])
FINGERS = ("left_finger", "right_finger")


def _window(tr: Trace, t0: float, t1: float) -> np.ndarray:
    return (tr.t >= t0 - 1e-9) & (tr.t < t1 - 1e-9)


def _tilt_deg(quat: np.ndarray) -> np.ndarray:
    """Angle between each body's z axis and world z (cubes: 0 when upright)."""
    zz = np.array([abs(quat_to_mat(q)[2, 2]) for q in np.atleast_2d(quat)])
    return np.degrees(np.arccos(np.clip(zz, -1.0, 1.0)))


def contact_force_on(tr: Trace, body: str, by: tuple[str, ...]) -> np.ndarray:
    """Per-frame summed native contact force on `body` from the bodies in `by` [N]."""
    names = tr.meta["contact_names"]
    a = np.array([names[i] for i in tr.c_a]) if len(tr.c_a) else np.zeros(0, str)
    b = np.array([names[i] for i in tr.c_b]) if len(tr.c_b) else np.zeros(0, str)
    out = np.zeros((len(tr.t), 3))
    on_b = (b == body) & np.isin(a, by)
    on_a = (a == body) & np.isin(b, by)
    np.add.at(out, tr.c_frame[on_b], tr.c_force[on_b])
    np.add.at(out, tr.c_frame[on_a], -tr.c_force[on_a])
    return out


def _arm_limit_violation(tr: Trace) -> float | None:
    if tr.q.shape[1] < 7:  # published bundles carry poses, not joint positions
        return None
    q = tr.q[:, :7]
    return float(np.maximum(ARM_RANGES[:, 0] - q, q - ARM_RANGES[:, 1]).max().clip(0))


def _arm_metrics(tr: Trace) -> dict:
    h = tr.body("hand")
    excursion = np.linalg.norm(tr.pos[:, h] - tr.pos[0, h], axis=1)
    return {"frames": len(tr.t), "hand_excursion_cm": float(excursion.max() * 100),
            "final_hand_pos_m": tr.pos[-1, h].tolist(),
            "max_arm_limit_violation_rad": _arm_limit_violation(tr),
            "max_abs_effort": np.abs(tr.tau).max(axis=0).tolist()}


def _tape(tr: Trace) -> dict:
    from simbench import tapes
    from simbench.scenes import panda

    spec = tr.meta["params"].get("tape_spec") or panda.TRACKING_SPEC
    return tapes.load(spec)


def tracking_metrics(tr: Trace) -> dict:
    qref = _tape(tr)["qref"]
    k = np.minimum(np.round(tr.t / 0.001).astype(int), len(qref) - 1)
    err = np.degrees(np.abs(tr.q - qref[k, :7]))
    return {**_arm_metrics(tr), "max_tracking_error_deg": float(err.max())}


def grasp_metrics(tr: Trace) -> dict:
    i = tr.body("cube")
    z0, xy0 = tr.com[0, i, 2], tr.com[0, i, :2]
    lift = tr.com[:, i, 2] - z0
    lateral = np.linalg.norm(tr.com[:, i, :2] - xy0, axis=1)
    hold = _window(tr, 1.8, 2.3)
    normals = []
    for finger in FINGERS:
        f = contact_force_on(tr, finger, ("cube",))
        y = np.array([quat_to_mat(q)[:, 1] for q in tr.quat[:, tr.body("hand")]])
        normals.append(np.abs(np.einsum("ij,ij->i", f, y)))
    normal = np.mean(normals, axis=0)
    # Cube COM in the hand frame: slip of the held cube during the hold window.
    h = tr.body("hand")
    rel = np.array([quat_to_mat(q).T @ (c - p) for q, c, p in
                    zip(tr.quat[:, h], tr.com[:, i], tr.pos[:, h], strict=True)])
    slip = np.linalg.norm(rel[hold] - rel[hold][0], axis=1)
    return {"frames": len(tr.t), "max_lift_cm": float(lift.max() * 100),
            "hold_lift_cm": float(lift[hold].mean() * 100),
            "final_height_error_mm": float(lift[-1] * 1e3),
            "max_lateral_mm": float(lateral.max() * 1e3),
            "final_lateral_mm": float(lateral[-1] * 1e3),
            "hold_slip_mm": float(slip.max() * 1e3),
            "mean_hold_normal_force_per_finger_N": float(normal[hold].mean()),
            "peak_normal_force_per_finger_N": float(normal.max()),
            "max_arm_limit_violation_rad": _arm_limit_violation(tr)}


def stack_metrics(tr: Trace, release: float = 4.0, tail: float = 0.5) -> dict:
    u, s = tr.body("upper"), tr.body("support")
    rel = (tr.com[:, u] - tr.com[:, s]) * 1e3
    tilt = _tilt_deg(tr.quat[:, u])
    end = tr.t >= tr.t[-1] - tail + 1e-9
    speed = np.linalg.norm(tr.linvel[:, [u, s]], axis=2).max(axis=1)
    on_support = np.linalg.norm(contact_force_on(tr, "upper", ("support",)), axis=1) > 0
    after = tr.t >= release - 1e-9
    k_rel = int(np.argmax(after))
    survives = bool(abs(rel[-1, 2] - 40) < 5 and np.all(np.abs(rel[-1, :2]) < 20)
                    and tilt[-1] < 10)
    window_range = np.ptp(tr.com[end][:, [u, s]], axis=0).max(axis=0) * 1e3
    return {"frames": len(tr.t), "stack_survives": survives,
            "sustained_stack": bool(survives and on_support[end].all()
                                    and speed[end].max() < 1e-3),
            "final_relative_position_mm": rel[-1].tolist(),
            "release_relative_position_mm": rel[k_rel].tolist(),
            "final_tilt_deg": float(tilt[-1]),
            "max_post_release_tilt_deg": float(tilt[after].max()),
            "final_base_displacement_mm": float(
                np.linalg.norm(tr.com[-1, s] - tr.com[0, s]) * 1e3),
            "max_final_speed_m_s": float(speed[end].max()),
            "tail_support_contact_fraction": float(on_support[end].mean()),
            "final_window_position_range_mm": window_range.tolist(),
            "max_arm_limit_violation_rad": _arm_limit_violation(tr)}


def push_metrics(tr: Trace) -> dict:
    p = tr.meta["params"]
    objects = ["cube", "block_left", "block_right"]
    target = p["entrance_x"] + p["pass_mm"] * 1e-3
    final = {o: tr.com[-1, tr.body(o)].tolist() for o in objects}
    passed = [o for o in objects if final[o][0] >= target]
    pad = sum(np.linalg.norm(contact_force_on(tr, o, FINGERS), axis=1) for o in objects)
    h = tr.body("hand")
    start = {o: tr.com[0, tr.body(o), 0] for o in objects}
    progress = min(min((final[o][0] - start[o]) / (target - start[o]), 1.0) for o in objects)
    tuning = {}
    if "limit" in p and tr.tau.shape[1] >= 7:
        # Selection score: squared arm effort normalised by the limits, episode mean.
        effort = (tr.tau[:, :7] / np.asarray(p["limit"][:7])) ** 2
        tuning = {"mean_norm_sq_effort": float(effort.mean())}
    return tuning | {"frames": len(tr.t), "all_passed": len(passed) == 3,
            "passed_count": len(passed), "min_progress_capped": float(progress),
            "final_positions_m": final,
            "max_pad_push_N": float(pad.max()),
            "final_hand_travel_mm": float((tr.pos[-1, h, 0] - tr.pos[0, h, 0]) * 1e3),
            "max_hand_height_change_mm": float(np.abs(tr.pos[:, h, 2] - tr.pos[0, h, 2]).max()
                                               * 1e3),
            "max_finger_opening_mm": float(tr.q[:, 7:9].sum(axis=1).max() * 1e3),
            "max_object_tilt_deg": float(max(_tilt_deg(tr.quat[:, tr.body(o)]).max()
                                             for o in objects)),
            "max_arm_limit_violation_rad": _arm_limit_violation(tr)}


def sweep_metrics(tr: Trace) -> dict:
    """Article criterion: object centre inside the tray footprint at the end of the run."""
    p = tr.meta["params"]
    objects = [b for b in tr.bodies if b.startswith("object_")]
    (tx, ty, tz), (sx, sy, sz) = p["tray"]["pos"], p["tray"]["size"]
    final = np.array([tr.com[-1, tr.body(o)] for o in objects])
    in_tray = ((np.abs(final[:, 0] - tx) <= sx / 2) & (np.abs(final[:, 1] - ty) <= sy / 2)
               & (final[:, 2] >= tz + sz / 2 - 0.005))
    fallen = final[:, 2] < p["table_top"] - 0.05
    out = {"frames": len(tr.t), "collected": int(in_tray.sum()),
           "collected_ids": [o for o, c in zip(objects, in_tray, strict=True) if c],
           "fallen": int(fallen.sum()),
           "left_outside": int((~in_tray & ~fallen).sum()),
           "max_arm_limit_violation_rad": _arm_limit_violation(tr)}
    if "contact_names" in tr.meta:
        paddle = np.linalg.norm(contact_force_on(tr, "paddle", tuple(objects)), axis=1)
        static = np.linalg.norm(contact_force_on(tr, "paddle", ("world",)), axis=1)
        out |= {"peak_paddle_contact_N": float(paddle.max()),
                "paddle_static_contact_frames": int((static > 0).sum())}
    return out


def insertion_depth(tr: Trace) -> tuple[np.ndarray, np.ndarray]:
    """Peg tip depth below the bore top and its distance from the bore axis [m]."""
    p = tr.meta["params"]
    hand = tr.body("hand")
    rot = np.array([quat_to_mat(q) for q in tr.quat[:, hand]])
    tip = tr.pos[:, hand] + rot @ np.asarray(p["peg_tip_hand"])
    origin = np.asarray(p["bore_origin"])
    return origin[2] - tip[:, 2], np.linalg.norm(tip[:, :2] - origin[:2], axis=1)


def _hand_wrench(tr: Trace, peg_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame contact force and moment about the hand from contacts on peg_ids."""
    hand = tr.body("hand")
    force = np.zeros((len(tr.t), 3))
    moment = np.zeros((len(tr.t), 3))
    if not peg_ids or "c_frame" not in tr.arrays or not len(tr.c_frame):
        return force, moment, np.zeros(0)
    sel = np.isin(tr.c_a, peg_ids) | np.isin(tr.c_b, peg_ids)
    if not sel.any():
        return force, moment, np.zeros(0)
    sign = np.where(np.isin(tr.c_b[sel], peg_ids), 1.0, -1.0)[:, None]
    f = tr.c_force[sel] * sign
    frames = tr.c_frame[sel]
    np.add.at(force, frames, f)
    np.add.at(moment, frames, np.cross(tr.c_pos[sel] - tr.pos[frames, hand], f))
    return force, moment, frames


def insertion_metrics(tr: Trace) -> dict:
    """Article criterion: depth >= 19 mm throughout the last 0.25 s of a completed run."""
    p = tr.meta["params"]
    depth, lateral = insertion_depth(tr)
    steps = int(round(p["duration"] / p["dt"]))
    completed = tr.meta.get("status") == "ok" and len(tr.t) == steps
    tail = tr.t >= p["duration"] - p["pass_window_s"] - 1e-9
    out = {"frames": len(tr.t), "completed": completed,
           "depth_reached": bool(completed and depth[tail].min() >= p["depth_pass_m"]),
           "final_depth_mm": float(depth[-1] * 1e3), "max_depth_mm": float(depth.max() * 1e3),
           "final_lateral_mm": float(lateral[-1] * 1e3),
           "stop_time_s": None if completed else float(tr.t[-1])}
    if "contact_names" in tr.meta:
        names = tr.meta["contact_names"]
        peg_ids = [i for i, n in enumerate(names) if n == "peg" or n.startswith("peg_")]
        force, moment, frames = _hand_wrench(tr, peg_ids)
        sel = np.isin(tr.c_a, peg_ids) | np.isin(tr.c_b, peg_ids) if peg_ids else slice(0, 0)
        penet = 0.0
        if peg_ids and "c_dist" in tr.arrays and np.shape(tr.c_dist)[0]:
            mask = np.isin(tr.c_a, peg_ids) | np.isin(tr.c_b, peg_ids)
            if mask.any():
                penet = float(max(0.0, -tr.c_dist[mask].min()) * 1e6)
        out |= {"peak_contact_N": float(np.linalg.norm(force, axis=1).max()),
                "peak_hand_force_N": float(np.linalg.norm(force, axis=1).max()),
                "peak_hand_force_xy_N": float(np.linalg.norm(force[:, :2], axis=1).max()),
                "peak_hand_force_z_N": float(np.max(np.abs(force[:, 2]))),
                "peak_hand_moment_Nm": float(np.linalg.norm(moment, axis=1).max()),
                "peak_penetration_um": penet,
                "contact_frames": int(len(np.unique(frames))) if len(frames) else 0}
    else:  # published bundles: net peg contact force sampled every 10 ms
        out["peak_contact_N"] = float(np.linalg.norm(tr.contact_net[:, tr.body("peg")],
                                                     axis=1).max())
        out["peak_hand_force_N"] = out["peak_contact_N"]
    return out


def soft_insertion_metrics(tr: Trace) -> dict:
    """Soft-insertion: depth + hand wrench + continuum extras (strain / permanent set)."""
    out = insertion_metrics(tr)
    p = tr.meta.get("params") or {}
    sk = tr.meta.get("sim_kwargs") or {}
    st = tr.meta.get("settings") or {}
    out["kind"] = p.get("kind", sk.get("fem_mode", ""))
    if st.get("peak_vertex_disp_m") is not None:
        out["peak_vertex_disp_mm"] = float(st["peak_vertex_disp_m"]) * 1e3
    fem = st.get("fem") or p.get("polymer") or {}
    if fem:
        out["fem_E_Pa"] = float(sk.get("E", fem.get("E", p.get("flex_peg", {}).get("young", 0))))
        out["fem_nu"] = float(sk.get("nu", fem.get("nu", p.get("flex_peg", {}).get("poisson", 0))))
    yield_strain = float(p.get("yield_strain") or p.get("flex_peg", {}).get("yield") or 0.0)
    out["yield_strain"] = yield_strain

    flex_idxs = [i for i, b in enumerate(tr.bodies) if b.startswith("peg_")]
    if flex_idxs:
        # Continuum tip = free (non-mount) vertex centroid; ghost "peg" follows the hand
        # rigidly and must not decide depth_reached when the mesh has exploded.
        mount = set((p.get("flex_peg") or {}).get("mount") or [])
        free = [i for i, b in enumerate(tr.bodies)
                if b.startswith("peg_") and int(b.split("_", 1)[1]) not in mount] or flex_idxs
        tip = tr.pos[:, free].mean(axis=1)
        origin = np.asarray(p["bore_origin"])
        depth = origin[2] - tip[:, 2]
        lateral = np.linalg.norm(tip[:, :2] - origin[:2], axis=1)
        disp = np.linalg.norm(tr.pos[:, flex_idxs] - tr.pos[0:1, flex_idxs], axis=2)
        peak_disp = float(disp.max())
        out["peak_vertex_disp_mm"] = peak_disp * 1e3
        # >50 mm of vertex travel from rest is a continuum blow-up, not insertion.
        bad = ~np.isfinite(tr.pos[:, flex_idxs]).all(axis=(1, 2)) | (disp.max(axis=1) > 0.05)
        out["continuum_valid"] = bool(not bad.any())
        out["final_depth_mm"] = float(depth[-1] * 1e3)
        out["max_depth_mm"] = float(depth.max() * 1e3)
        out["final_lateral_mm"] = float(lateral[-1] * 1e3)
        steps = int(round(p["duration"] / p["dt"]))
        completed = tr.meta.get("status") == "ok" and len(tr.t) == steps
        tail = tr.t >= p["duration"] - p["pass_window_s"] - 1e-9
        out["depth_reached"] = bool(
            completed and out["continuum_valid"] and depth[tail].min() >= p["depth_pass_m"])
        if "peak_hand_force_N" in out and not out["continuum_valid"]:
            # Report force on valid prefix only so a blow-up spike does not dominate.
            n_ok = int(np.argmax(bad)) if bad.any() else len(tr.t)
            n_ok = max(n_ok, 1)
            names = tr.meta.get("contact_names") or []
            peg_ids = [i for i, n in enumerate(names) if n == "peg" or n.startswith("peg_")]
            force, moment, _ = _hand_wrench(tr, peg_ids)
            force, moment = force[:n_ok], moment[:n_ok]
            out["peak_hand_force_N"] = float(np.linalg.norm(force, axis=1).max())
            out["peak_hand_moment_Nm"] = float(np.linalg.norm(moment, axis=1).max())
            out["peak_contact_N"] = out["peak_hand_force_N"]
        if yield_strain:
            mid = len(tr.t) // 2
            out["permanent_set_mm"] = float(np.linalg.norm(tip[-1] - tip[mid]) * 1e3)
    return out


def ball_vertices(tr: Trace) -> np.ndarray:
    return tr.pos[:, [i for i, b in enumerate(tr.bodies) if b.startswith("ball_")]]


def invalid_frames(tr: Trace) -> np.ndarray:
    """Article geometry gate per frame: a non-finite vertex or an inverted tet."""
    v = ball_vertices(tr)
    tets = np.asarray(tr.meta["params"]["tets"])
    a, b, c, d = (v[:, tets[:, i]] for i in range(4))
    vol = np.einsum("nti,nti->nt", np.cross(b - a, c - a), d - a) / 6
    return ~np.isfinite(v).all(axis=(1, 2)) | (vol * np.sign(vol[0]) <= 0).any(axis=1)


def _invalid_meta(tr: Trace) -> tuple[int, dict]:
    """Frames up to the first invalid one; the article stops runs there."""
    bad = invalid_frames(tr)
    hit = np.nonzero(bad)[0]
    n = int(hit[0]) if len(hit) else len(tr.t)
    stop = float(tr.t[n]) if len(hit) else None
    if stop is None and tr.meta.get("status") != "ok":
        stop = float(tr.t[-1])
    return n, {"invalid_at_s": stop, "completed": stop is None and len(tr.t) > 0,
               "invalid_frames": int(bad.sum())}


def ball_compress_metrics(tr: Trace) -> dict:
    n, out = _invalid_meta(tr)
    v = ball_vertices(tr)[:n]
    t = tr.t[:n]
    load = np.abs(tr.contact_net[:n, tr.body("plate_hi"), 2])
    hold = (t >= 0.6 - 1e-9) & (t <= 0.8 + 1e-9)
    extent = np.ptp(v[:, :, 2], axis=1)
    p = tr.meta["params"]
    gap = 2 * (p["plate_gap0"] - p["squeeze"] - p["plate_half"][2])
    return out | {"frames": len(tr.t),
                  "hold_plate_load_N": float(load[hold].mean()) if hold.any() else None,
                  "hold_extent_mm": float(extent[hold].mean() * 1e3) if hold.any() else None,
                  "plate_gap_mm": gap * 1e3,
                  "final_extent_mm": float(extent[-1] * 1e3),
                  "permanent_set_mm": float((extent[0] - extent[-1]) * 1e3)
                  if out["completed"] else None,
                  "peak_plate_load_N": float(load.max())}


def ball_grasp_metrics(tr: Trace) -> dict:
    """Lift is the mass-weighted COM rise; the run is judged up to its first invalid frame."""
    p = tr.meta["params"]
    n, out = _invalid_meta(tr)
    mass = np.asarray(p["node_mass"])
    v_all = ball_vertices(tr)
    lift_all = (v_all[:, :, 2] - v_all[0, :, 2]) @ mass / mass.sum() * 1e3
    v, t, lift = v_all[:n], tr.t[:n], lift_all[:n]
    centroid = (v[:, :, 2].mean(axis=1) - v[0, :, 2].mean()) * 1e3
    t0, t1 = p["hold_window"]
    hold = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    hold_all = (tr.t >= t0 - 1e-9) & (tr.t <= t1 + 1e-9)
    if hold_all.any() and np.isfinite(lift_all[hold_all]).all():
        out["ungated_hold_lift_mm"] = float(lift_all[hold_all].mean())
    # Width lost across the fingers (hand y) after release: the plastic dent.
    width = np.ptp(v_all[:, :, 1], axis=1)
    out["dent_mm"] = (float((width[0] - width[-1]) * 1e3)
                      if tr.meta["status"] == "ok" and np.isfinite(width).all() else None)
    full_hold = len(t) and t[-1] >= t1 - 1e-9
    stop = out["invalid_at_s"]
    phase = (None if stop is None else "before hold" if stop < t0 else
             "during hold" if stop <= 2.3 else "during lowering" if stop <= 3.1 else
             "during release")
    hold_lift = float(lift[hold].mean()) if full_hold and hold.any() else None
    out |= {"frames": len(tr.t), "failure_phase": phase,
            "hold_lift_mm": hold_lift,
            "hold_centroid_lift_mm": float(centroid[hold].mean()) if full_hold and hold.any()
            else None,
            "sustained_lift": bool(hold_lift is not None and hold_lift >= 40.0),
            "max_lift_mm": float(lift.max()) if len(lift) else None,
            "min_lift_mm": float(lift.min()) if len(lift) else None,
            "max_arm_limit_violation_rad": _arm_limit_violation(tr)}
    if "contact_names" in tr.meta and "ball" in tr.meta["contact_names"]:
        pads = np.linalg.norm(contact_force_on(tr, "left_finger", ("ball",)), axis=1)[:n]
        out["hold_pad_force_N"] = float(pads[hold].mean()) if hold.any() else None
    return out


def card_metrics(tr: Trace) -> dict:
    """Outcome of the thin card: caught on the plate, tunneled through it, or off its edge."""
    p = tr.meta["params"]
    pos = tr.pos[:, tr.body("card")]
    inside = (np.abs(pos[:, :2]) < p["plate_half_xy"]).all(axis=1)
    below = (pos[:, 2] < p["plate_bottom"]) & inside
    rest = pos[-1, 2] - p["plate_top"] - p["card_half"][2]
    if below.any():
        outcome = "tunneled"
    elif inside[-1] and abs(rest) < 2e-3:
        outcome = "caught"
    else:
        outcome = "off plate"
    k = int(np.argmax(below)) if below.any() else len(tr.t)
    pen = p["plate_top"] + p["card_half"][2] - pos[:k, 2]
    return {"frames": len(tr.t), "outcome": outcome, "caught": outcome == "caught",
            "tunnel_at_s": float(tr.t[k]) if below.any() else None,
            "rest_gap_mm": float(rest * 1e3) if outcome == "caught" else None,
            "max_penetration_mm": float(pen.max() * 1e3) if k else None,
            "impact_speed": float(np.sqrt(p["speed"]**2 + 2 * 9.81 * 0.005)),
            "height_mm": ((pos[tr.t <= 0.06, 2] - p["plate_top"]) * 1e3).round(3).tolist()}


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def tumble_reference(params: dict, t: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Independent torque-free rigid-body solve (RK4 on Euler's equations and q) [w x y z]."""
    inertia = np.array(params["inertia"])

    def f(state):
        q, w = state[:4], state[4:]
        return np.concatenate([0.5 * _qmul(q, np.r_[0.0, w]),
                               np.cross(inertia * w, w) / inertia])

    state = np.r_[1.0, 0.0, 0.0, 0.0, params["omega0"]]
    out, now = np.zeros((len(t), 4)), 0.0
    for i, target in enumerate(t):
        while now < target - 1e-12:
            step = min(h, target - now)
            k1 = f(state)
            k2 = f(state + step / 2 * k1)
            k3 = f(state + step / 2 * k2)
            k4 = f(state + step * k3)
            state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            state[:4] /= np.linalg.norm(state[:4])
            now += step
        out[i] = state[:4]
    return out


def tumble_metrics(tr: Trace) -> dict:
    """Orientation error vs the reference, and drift of world angular momentum and energy."""
    p = tr.meta["params"]
    b = tr.body("handle")
    quat, w = tr.quat[:, b], tr.angvel[:, b]
    ref = tumble_reference(p, tr.t)
    err = np.degrees(2 * np.arccos(np.clip(np.abs((quat * ref).sum(axis=1)), 0, 1)))
    inertia = np.array(p["inertia"])
    rot = np.stack([quat_to_mat(q) for q in quat])
    w_body = np.einsum("nji,nj->ni", rot, w)
    ang_mom = np.einsum("nij,nj->ni", rot, inertia * w_body)
    energy = 0.5 * (inertia * w_body**2).sum(axis=1)
    return {"frames": len(tr.t), "final_error_deg": float(err[-1]),
            "max_error_deg": float(err.max()),
            "momentum_drift": float(np.linalg.norm(ang_mom - ang_mom[0], axis=1).max()
                                    / np.linalg.norm(ang_mom[0])),
            "energy_drift": float(np.abs(energy / energy[0] - 1).max()),
            "error_deg": err[:: max(1, len(err) // 200)].round(6).tolist()}


def sheet_metrics(tr: Trace) -> dict:
    """Tip deflection of the strip under the press and the permanent set after release."""
    p = tr.meta["params"]
    tip = tr.pos[:, [tr.body(n) for n in p["tip_bodies"]], 2].mean(axis=1)
    if not np.isfinite(tip).all():
        bad = int(np.argmax(~np.isfinite(tip)))
        return {"frames": len(tr.t), "invalid_at_s": float(tr.t[bad]), "held_tip_mm": None,
                "permanent_set_mm": None}
    hold = int(np.argmin(np.abs(tr.t - p["hold_t"])))
    tail = tr.t >= tr.t[-1] - 0.2
    return {"frames": len(tr.t), "invalid_at_s": None,
            "held_tip_mm": float(-tip[hold] * 1e3), "peak_tip_mm": float(-tip.min() * 1e3),
            "permanent_set_mm": float(-tip[tail].mean() * 1e3),
            "tip_mm": (-tip[:: max(1, len(tip) // 200)] * 1e3).round(3).tolist()}


def twist_metrics(tr: Trace) -> dict:
    """Slip torque: the applied twist torque when the puck has turned 1 degree."""
    p = tr.meta["params"]
    angle, torque = np.degrees(tr.q[:, 2]), tr.tau[:, 2]
    if not np.isfinite(angle).all():
        return {"frames": len(tr.t), "slip_torque_Nm": None, "slip_ratio": None}
    held = tr.t >= p["settle"]
    slipped = np.flatnonzero(held & (np.abs(angle) > 1.0))
    slip = float(torque[slipped[0]]) if len(slipped) else None
    puck = tr.meta["contact_names"].index("puck")
    on = (tr.c_a == puck) | (tr.c_b == puck)
    frames = tr.c_frame[on & (tr.c_frame < int(np.argmax(held)) + 50)]
    per_frame = np.bincount(frames, minlength=len(tr.t))[held][:50] if len(frames) else [0]
    squeeze = np.abs(tr.qd[:, :2]).max(axis=1)
    if len(slipped):
        held = held & (tr.t < tr.t[slipped[0]])
    return {"frames": len(tr.t), "slip_torque_Nm": slip,
            "slip_torque_ref_Nm": p["slip_torque_ref"],
            "slip_ratio": slip / p["slip_torque_ref"] if slip is not None else None,
            "slip_at_s": float(tr.t[slipped[0]]) if len(slipped) else None,
            "contacts_per_step": float(np.mean(per_frame)),
            "pad_speed_max": float(squeeze[held].max()),
            "angle_deg": angle[:: max(1, len(angle) // 200)].round(4).tolist(),
            "torque_Nm": torque[:: max(1, len(torque) // 200)].round(5).tolist()}


def _grav_z_body(quat_wxyz) -> float:
    """z-component of gravity (0,0,-1) in the body frame."""
    w, x, y, z = quat_wxyz
    # R^T @ (0,0,-1) z-component = -(1 - 2(x^2+y^2)) = 2(x^2+y^2) - 1
    return 2 * (x * x + y * y) - 1


def walk_metrics(tr: Trace) -> dict:
    """G1 Cartwheel tracking: fall time, joint/root error vs reference, foot sinkage."""
    p = tr.meta["params"]
    base = tr.pos[:, tr.body("pelvis")]
    if not len(tr.t) or not np.isfinite(base).all():
        return {"frames": len(tr.t), "fell": None, "completed": False}
    hz = float(p.get("policy_hz") or 50.0)
    ref = None
    try:
        from simbench.scenes.g1 import load_motion
        ref = load_motion(p.get("motion") or "basketball_shot")
    except Exception:
        ref = np.asarray(p.get("ref_qpos") or [], float)
        if ref.ndim != 2 or ref.shape[0] == 0:
            ref = None
    fell_at = None
    if ref is not None:
        for i, t in enumerate(tr.t):
            fi = min(int(round(t * hz)), len(ref) - 1)
            if abs(ref[fi, 2] - base[i, 2]) > 0.25:
                fell_at = float(t)
                break
            if abs(_grav_z_body(ref[fi, 3:7]) - _grav_z_body(tr.quat[i, tr.body("pelvis")])) > 0.8:
                fell_at = float(t)
                break
    if fell_at is None:
        down = np.flatnonzero(base[:, 2] < 0.35)
        fell_at = float(tr.t[down[0]]) if len(down) else None
    up = tr.t < (fell_at if fell_at is not None else np.inf)
    x = base[up]
    frames = np.flatnonzero(up)
    on = np.isin(tr.c_frame, frames) if len(tr.c_frame) else np.zeros(0, bool)
    pen = np.zeros(len(tr.t))
    if len(tr.c_frame):
        np.maximum.at(pen, tr.c_frame[on], -tr.c_dist[on])
    joint_mae = root_err = None
    if ref is not None and tr.q.shape[1] > 0:
        errs, roots = [], []
        for i in np.flatnonzero(up):
            fi = min(int(round(tr.t[i] * hz)), len(ref) - 1)
            n = min(tr.q.shape[1], ref.shape[1] - 7)
            errs.append(np.abs(tr.q[i, :n] - ref[fi, 7:7 + n]).mean())
            roots.append(np.linalg.norm(base[i] - ref[fi, :3]))
        joint_mae = float(np.mean(errs)) if errs else None
        root_err = float(np.mean(roots)) if roots else None
    completed = fell_at is None and tr.meta.get("status") == "ok"
    step = max(1, len(tr.t) // 250)
    return {"frames": len(tr.t), "fell": fell_at is not None, "fell_at_s": fell_at,
            "completed": bool(completed),
            "joint_mae_deg": float(np.degrees(joint_mae)) if joint_mae is not None else None,
            "root_err_mm": float(root_err * 1e3) if root_err is not None else None,
            "distance_m": float(x[-1, 0] - x[0, 0]) if len(x) else None,
            "base_height_m": float(x[:, 2].mean()) if len(x) else None,
            "sink_mm": float(np.median(pen[frames]) * 1e3) if len(frames) else None,
            "x_m": base[::step, 0].round(4).tolist(), "z_m": base[::step, 2].round(4).tolist()}


FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def cloth_fold_metrics(tr: Trace) -> dict:
    """Half-plane coverage after the fold + grasp slip of pinched cloth patches."""
    p = tr.meta["params"]
    cloth = [i for i, b in enumerate(tr.bodies) if b.startswith(p.get("flex_prefix", "cloth_"))]
    out = {"frames": len(tr.t), "completed": tr.meta.get("status") == "ok"}
    if not cloth or not len(tr.t):
        return out | {"coverage": None, "fold_ok": False, "grasp_slip_mm": None}
    pos = tr.pos[:, cloth]
    if not np.isfinite(pos).all():
        bad = int(np.argmax(~np.isfinite(pos).all(axis=(1, 2))))
        return out | {"invalid_at_s": float(tr.t[bad]), "coverage": None, "fold_ok": False,
                      "grasp_slip_mm": None}
    # Target half-plane: y <= 0 (fold left half onto the right / negative-y side).
    final = pos[-1]
    coverage = float((final[:, 1] <= 0.0).mean())
    out["coverage"] = coverage
    out["fold_ok"] = bool(coverage >= float(p.get("coverage_pass", 0.8)))
    # Grasp slip: patch-centroid drift in the hand's local frame during attach.
    # World-frame (cent - hand) grows under wrist rotation even with a perfect weld.
    slip = []
    t0, t1 = p.get("attach_window") or [0.65, 2.60]
    closed = (tr.t >= t0) & (tr.t <= t1)
    for hand_name, ids_key in (("L_hand", "pinch_left"), ("R_hand", "pinch_right")):
        ids = [i for i in (p.get(ids_key) or []) if i < len(cloth)]
        if not ids or hand_name not in tr.bodies or not closed.any():
            continue
        hand_i = tr.body(hand_name)
        hand = tr.pos[closed, hand_i]
        quat = tr.quat[closed, hand_i]  # wxyz
        cent = pos[closed][:, ids].mean(axis=1)
        # qinv * (cent - hand) for each frame.
        w, xyz = quat[:, 0], quat[:, 1:]
        d = cent - hand
        # R^T v = v + 2*(-u)×((-u)×v + w v) with u = xyz → use conjugate (-xyz, w)
        u = -xyz
        local = d + 2 * np.cross(u, np.cross(u, d) + w[:, None] * d)
        slip.append(float(np.linalg.norm(local - local[0], axis=1).max() * 1e3))
    out["grasp_slip_mm"] = float(max(slip)) if slip else None
    out["slip_ok"] = bool(out["grasp_slip_mm"] is not None
                          and out["grasp_slip_mm"] <= float(p.get("slip_pass_mm", 25.0)))
    out["passed"] = bool(out.get("completed") and out["fold_ok"] and out.get("slip_ok"))
    return out


METRICS = {
    "g1-walk": walk_metrics,
    "twist": twist_metrics,
    "card-catch": card_metrics,
    "tumble": tumble_metrics,
    "sheet": sheet_metrics,
    "slide": lambda tr: cube_metrics(tr, 0.1, 0.04),
    "drop": lambda tr: cube_metrics(tr, 0.1, 0.04),
    "hinge": hinge_metrics,
    "hinge-plastic": hinge_plastic_metrics,
    "collision-chain": chain_metrics,
    "panda-effort": _arm_metrics,
    "panda-tracking": tracking_metrics,
    "grasp": grasp_metrics,
    "stack": stack_metrics,
    "push": push_metrics,
    "sweep": sweep_metrics,
    "insertion": insertion_metrics,
    "soft-insertion": soft_insertion_metrics,
    "cloth-fold": cloth_fold_metrics,
    "ball-compress": ball_compress_metrics,
    "ball-grasp": ball_grasp_metrics,
}
