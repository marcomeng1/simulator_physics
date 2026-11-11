"""Per-plan metric summaries: every route and seed, plus the article's own traces."""

import json
from collections import defaultdict

import numpy as np

from simbench.metrics import METRICS
from simbench.paths import RAW, SUMMARY
from simbench.plans import PLANS, PRIMARY
from simbench.reference import (
    BUNDLES,
    INSERTION_GUIDED,
    SWEEP_REPEATS,
    article_deform,
    article_traces,
)
from simbench.trace import Trace

# Published article bundles for variants other than a scene's PRIMARY one.
ARTICLE_VARIANTS = {"insertion/dt1ms-c100-tilted-guided": INSERTION_GUIDED}
ROUTE_ORDER = ["mujoco311", "mujoco4", "mjwarp4", "newton", "genesis"]
HEADLINE = {
    "slide": ["travel_x_m", "peak_contact_force_N", "max_com_z_m", "settled_at_s",
              "max_momentum_residual_Ns"],
    "drop": ["peak_contact_force_N", "max_overlap_m", "settled_at_s", "impulse_first_40ms_Ns",
             "max_momentum_residual_Ns"],
    "hinge": ["max_reference_error_deg", "final_angle_rad"],
    "hinge-plastic": ["peak_angle_deg", "final_angle_deg", "expected_final_deg"],
    "collision-chain": ["cube_final_xy_mm", "prism_final_heading_deg", "sphere_final_xy_mm",
                        "peak_horizontal_force_N"],
    "panda-effort": ["hand_excursion_cm", "max_arm_limit_violation_rad"],
    "panda-tracking": ["max_tracking_error_deg", "hand_excursion_cm"],
    "grasp": ["max_lift_cm", "hold_lift_cm", "hold_slip_mm", "final_lateral_mm",
              "mean_hold_normal_force_per_finger_N"],
    "stack": ["stack_survives", "sustained_stack", "final_relative_position_mm",
              "final_tilt_deg", "max_final_speed_m_s"],
    "push": ["all_passed", "passed_count", "max_pad_push_N", "max_hand_height_change_mm",
             "max_object_tilt_deg"],
    "sweep": ["collected", "fallen", "left_outside", "peak_paddle_contact_N"],
    "insertion": ["depth_reached", "final_depth_mm", "peak_contact_N", "peak_penetration_um"],
    "soft-insertion": ["depth_reached", "final_depth_mm", "peak_hand_force_N",
                       "peak_hand_moment_Nm", "peak_vertex_disp_mm", "permanent_set_mm"],
    "cloth-fold": ["passed", "coverage", "grasp_slip_mm", "fold_ok", "slip_ok"],
    "ball-compress": ["completed", "hold_plate_load_N", "hold_extent_mm", "permanent_set_mm"],
    "ball-grasp": ["completed", "invalid_at_s", "hold_lift_mm", "ungated_hold_lift_mm",
                   "invalid_frames", "dent_mm"],
    "card-catch": ["outcome", "tunnel_at_s", "rest_gap_mm", "max_penetration_mm"],
    "tumble": ["final_error_deg", "max_error_deg", "momentum_drift", "energy_drift"],
    "sheet": ["held_tip_mm", "peak_tip_mm", "permanent_set_mm"],
    "twist": ["slip_torque_Nm", "slip_ratio", "contacts_per_step", "pad_speed_max"],
    "g1-walk": ["fell_at_s", "completed", "joint_mae_deg", "root_err_mm", "sink_mm",
                "distance_m"],
}


def _clean(value):
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def seed_spread(traces: list[Trace]) -> float | None:
    """Largest COM distance [m] between any seed's trajectory and seed 0's."""
    if len(traces) < 2:
        return None
    base = traces[0].com
    spread = 0.0
    for tr in traces[1:]:
        n = min(len(base), len(tr.com))
        spread = max(spread, float(np.linalg.norm(tr.com[:n] - base[:n], axis=2).max()))
    return spread


def path_difference(tr: Trace, base: Trace | None) -> float | None:
    """Largest COM distance [m] from `base` over matching frames and bodies."""
    if (base is None or tr is base or not tr.meta.get("frames", 1)
            or not base.meta.get("frames", 1)):
        return None
    n = min(len(tr.t), len(base.t))
    if not np.allclose(tr.t[:n], base.t[:n]):
        return None
    shared = [b for b in tr.bodies if b in base.bodies]
    a = tr.com[:n, [tr.body(b) for b in shared]]
    b = base.com[:n, [base.body(b) for b in shared]]
    return float(np.linalg.norm(a - b, axis=2).max())


def variant_summary(scene: str, variant: str, baseline: str = "mujoco4") -> dict:
    folder = RAW / scene / variant
    by_route: dict[str, list[Trace]] = defaultdict(list)
    for meta in sorted(folder.glob("*.json")):
        tr = Trace.load(meta.with_suffix(""))
        by_route[tr.meta["route"]].append(tr)
    for traces in by_route.values():
        traces.sort(key=lambda t: t.meta["seed"])
    base = by_route[baseline][0] if baseline in by_route else None
    rows = {}
    for route in sorted(by_route, key=lambda r: ROUTE_ORDER.index(r) if r in ROUTE_ORDER else 99):
        traces = by_route[route]
        tr = traces[0]
        rows[route] = {
            "status": tr.meta["status"],
            "versions": {k: v for k, v in tr.meta["environment"]["packages"].items()
                         if k not in ("simbench", "numpy")},
            "seeds": [t.meta["seed"] for t in traces],
            "seed_spread_m": seed_spread(traces),
            f"path_diff_vs_{baseline}_m": _clean(path_difference(tr, base)),
            "notes": tr.meta.get("notes", []),
            "error": tr.meta.get("error"),
            "metrics": ({k: _clean(v) for k, v in METRICS[scene](tr).items()}
                        if tr.meta["frames"] else {}),
        }
        if len(traces) > 1:
            rows[route]["seed_metrics"] = [
                {"status": t.meta["status"],
                 **({k: _clean(METRICS[scene](t).get(k)) for k in HEADLINE[scene]}
                    if t.meta["frames"] else {})} for t in traces]
    bundle = (BUNDLES.get(scene) if PRIMARY.get(scene) == variant
              else ARTICLE_VARIANTS.get(f"{scene}/{variant}"))
    bundles = [(bundle, "")] if bundle else []
    if scene == "sweep" and bundle:
        bundles.append((SWEEP_REPEATS[2], "#2"))
    for path, suffix in bundles:
        for engine, tr in article_traces(scene, path).items():
            rows[tr.meta["route"] + suffix] = {
                "status": "published", "versions": {engine: tr.meta["engine_version"]},
                f"path_diff_vs_{baseline}_m": _clean(path_difference(tr, base)),
                "metrics": {k: _clean(v) for k, v in METRICS[scene](tr).items()},
            }
    deform = ("compression" if (scene, variant) == ("ball-compress", PRIMARY["ball-compress"])
              else {"dt0.05ms-e2N": "grasp-2N", "dt0.05ms-e1N": "grasp-1N"}.get(variant)
              if scene == "ball-grasp" else None)
    if deform:
        for route, row in article_deform(deform).items():
            rows[route] = row | {"metrics": {k: _clean(v) for k, v in row["metrics"].items()}}
    return rows


def _load(scene: str, variant: str, route: str, seed: int = 0) -> Trace | None:
    stem = RAW / scene / variant / f"{route}-seed{seed}"
    return Trace.load(stem) if stem.with_suffix(".json").exists() else None


def contact_counts(tr: Trace) -> dict[str, float]:
    """Mean number of native contacts per frame for each body pair (frames in contact)."""
    names = tr.meta["contact_names"]
    out = {}
    pairs = defaultdict(list)
    for k, a, b in zip(tr.c_frame, tr.c_a, tr.c_b, strict=True):
        pairs["-".join(sorted((names[a], names[b])))].append(k)
    for pair, frames in sorted(pairs.items()):
        out[pair] = len(frames) / len(set(frames))
    return out


def first_divergence(a: Trace, b: Trace, threshold: float = 1e-6) -> float | None:
    n = min(len(a.t), len(b.t))
    d = np.linalg.norm(a.com[:n] - b.com[:n], axis=2).max(axis=1)
    hit = np.nonzero(d > threshold)[0]
    return float(a.t[hit[0]]) if len(hit) else None


def version_report(plans=("phase1", "phase2", "phase3")) -> dict:
    """MuJoCo 3.11 vs 4.0 on identical MJCF and options, for every recorded variant."""
    rows = {}
    for scene, variant in sorted({(r.scene, r.variant) for p in plans for r in PLANS[p]}):
        old, new = _load(scene, variant, "mujoco311"), _load(scene, variant, "mujoco4")
        if old is None or new is None:
            continue
        metrics = METRICS[scene]
        rows[f"{scene}/{variant}"] = {
            "path_diff_m": _clean(path_difference(new, old)),
            "first_divergence_1um_s": first_divergence(new, old),
            "contacts_per_frame": {"mujoco311": _clean(list(contact_counts(old).items())),
                                   "mujoco4": _clean(list(contact_counts(new).items()))},
            "metrics": {"mujoco311": {k: _clean(v) for k, v in metrics(old).items()},
                        "mujoco4": {k: _clean(v) for k, v in metrics(new).items()}},
        }
    out = SUMMARY / "mujoco4-vs-311.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{'scene/variant':<36}{'path diff':>12}{'diverges at':>14}   contacts/frame 3.11 -> 4.0")
    for key, row in rows.items():
        diff = row["path_diff_m"] * 1e3
        div = row["first_divergence_1um_s"]
        old = dict(row["contacts_per_frame"]["mujoco311"])
        new = dict(row["contacts_per_frame"]["mujoco4"])
        changed = [f"{p} {old.get(p, 0):.2f}->{new.get(p, 0):.2f}" for p in sorted({*old, *new})
                   if abs(old.get(p, 0) - new.get(p, 0)) > 0.005]
        print(f"{key:<36}{diff:>9.3g} mm{'' if div is None else f'{div:.3f} s':>14}   "
              + "; ".join(changed))
    print(f"\nwrote {out.relative_to(SUMMARY.parent.parent)}")
    return rows


def sensitivity_report(plan: str = "phase4") -> dict:
    """Each option/determinism variant against its baseline on the same route."""
    rows = {}
    for run in PLANS[plan]:
        keys = HEADLINE[run.scene]
        for route in run.routes or ROUTE_ORDER:
            traces = [t for s in run.seeds if (t := _load(run.scene, run.variant, route, s))]
            bases = [t for s in run.seeds if (t := _load(run.scene, run.base, route, s))]
            if not traces or not bases:
                continue
            tr, base = traces[0], bases[0]
            ok = tr.meta["status"] == "ok"
            traces = [t for t in traces if t.meta["status"] == "ok"]
            rows[f"{run.scene}/{run.variant}/{route}"] = {
                "status": tr.meta["status"], "error": tr.meta.get("error"),
                "sim": run.sim, "base": run.base,
                "path_diff_vs_base_m": _clean(path_difference(tr, base)) if ok else None,
                "seed_spread_m": _clean(seed_spread(traces)),
                "base_seed_spread_m": _clean(seed_spread(bases)),
                "metrics": {k: _clean(METRICS[run.scene](tr).get(k)) for k in keys} if ok
                else None,
                "base_metrics": {k: _clean(METRICS[run.scene](base).get(k)) for k in keys},
            }
    out = SUMMARY / f"{plan}.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{'scene/variant/route':<52}{'status':>10}"
          + "".join(f"{h:>12}" for h in ("vs base", "seeds", "base seeds")))
    for key, row in rows.items():
        cells = [row["path_diff_vs_base_m"], row["seed_spread_m"], row["base_seed_spread_m"]]
        cells = ["" if v is None else f"{v * 1e3:.3g} mm" for v in cells]
        print(f"{key:<52}{row['status']:>10}" + "".join(f"{c:>12}" for c in cells))
        if row["metrics"]:
            changed = {k: (row["base_metrics"][k], v) for k, v in row["metrics"].items()
                       if v != row["base_metrics"][k]}
            for k, (b, v) in changed.items():
                print(f"{'':<8}{k}: {json.dumps(b)} -> {json.dumps(v)}")
        elif row["error"]:
            print(f"{'':<8}{row['error'][:150]}")
    print(f"\nwrote {out.relative_to(SUMMARY.parent.parent)}")
    return rows


def phase_report(plan: str) -> dict:
    variants = sorted({(r.scene, r.variant) for r in PLANS[plan]})
    report = {scene: {} for scene, _ in variants}
    for scene, variant in variants:
        report[scene][variant] = variant_summary(scene, variant)
    if plan == "terrain":
        # Keep Cartwheel attribution next to the numbers that depend on their release.
        report["_credits"] = {
            "legged_locomotion": (
                "Policy, basketball_shot motion and G1 packaging from Cartwheel Control "
                "(https://github.com/Cartwhl/cartwheel-control; "
                "https://huggingface.co/andrew-cartwheel/cartwheel-control). "
                "Licensed by Cartwheel under the Cartwheel Open Model License. "
                "Unitree G1 MJCF/meshes: BSD-3-Clause. Inference reimplemented in simbench "
                "(numpy); not cartwheel-play / mjlab."
            ),
        }
        print("\nCredit: Cartwheel Control — G1 policy + basketball_shot "
              "(https://github.com/Cartwhl/cartwheel-control; "
              "Cartwheel Open Model License).")
    out = SUMMARY / f"{plan}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    for scene, variants_ in report.items():
        if scene.startswith("_"):
            continue
        for variant, rows in variants_.items():
            keys = HEADLINE[scene]
            print(f"\n{scene} / {variant}")
            print(f"  {'route':<26}" + "".join(f"{k:>26}" for k in keys)
                  + "   vs mujoco4   seed spread")
            for route, row in rows.items():
                cells = "".join(f"{json.dumps(row['metrics'].get(k)):>26}" for k in keys)
                mm = [row.get(k) for k in ("path_diff_vs_mujoco4_m", "seed_spread_m")]
                mm = ["" if v is None else f"{v * 1e3:.3g} mm" for v in mm]
                print(f"  {route:<26}{cells}   {mm[0]:>10}   {mm[1]}")
    print(f"\nwrote {out.relative_to(SUMMARY.parent.parent)}")
    return report
