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
    return {"frames": len(tr.t), "all_passed": len(passed) == 3, "passed_count": len(passed),
            "final_positions_m": final,
            "max_pad_push_N": float(pad.max()),
            "final_hand_travel_mm": float((tr.pos[-1, h, 0] - tr.pos[0, h, 0]) * 1e3),
            "max_hand_height_change_mm": float(np.abs(tr.pos[:, h, 2] - tr.pos[0, h, 2]).max()
                                               * 1e3),
            "max_finger_opening_mm": float(tr.q[:, 7:9].sum(axis=1).max() * 1e3),
            "max_object_tilt_deg": float(max(_tilt_deg(tr.quat[:, tr.body(o)]).max()
                                             for o in objects)),
            "max_arm_limit_violation_rad": _arm_limit_violation(tr)}


METRICS = {
    "slide": lambda tr: cube_metrics(tr, 0.1, 0.04),
    "drop": lambda tr: cube_metrics(tr, 0.1, 0.04),
    "hinge": hinge_metrics,
    "collision-chain": chain_metrics,
    "panda-effort": _arm_metrics,
    "panda-tracking": tracking_metrics,
    "grasp": grasp_metrics,
    "stack": stack_metrics,
    "push": push_metrics,
}
