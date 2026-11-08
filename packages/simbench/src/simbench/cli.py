"""Command-line entry point: `simbench <command> ...`, run inside a route's uv environment."""

import argparse
import json
import sys
from pathlib import Path

from simbench.paths import SUMMARY

# The article's acceptance tolerances (panda/audit.md). fk_rot is the Frobenius norm of the
# rotation-matrix difference; joint properties allow for float32 rounding (0.1f - 0.1 ~ 1.5e-9).
TOLERANCES = {
    "mass": 1e-6, "com": 1e-6, "inertia": 1e-6,
    "joint_axis": 1e-6, "joint_range": 1e-6,
    "armature": 1e-7, "damping": 1e-7, "frictionloss": 1e-7,
    "fk_pos": 1e-5, "fk_rot": 1e-4,
}
MODEL_NAMES = ["panda", "panda_article"]


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def cmd_fetch(args) -> None:
    from simbench.fetch import fetch_menagerie, fetch_reference, fetch_unitree

    fetch_menagerie(update=args.update)
    fetch_unitree(update=args.update)
    fetch_reference(update=args.update)


def cmd_env_report(args) -> None:
    from simbench import adapters
    from simbench.envinfo import environment
    from simbench.models import model_path

    report = environment(args.route)
    adapter = adapters.load(args.route)
    report["smoke"] = {name: adapter.smoke(model_path(name)) for name in MODEL_NAMES}
    _write(SUMMARY / "env" / f"{args.route}.json", report)
    print(json.dumps({"route": args.route, "packages": report["packages"],
                      "smoke": report["smoke"]}, indent=2))


def cmd_audit(args) -> None:
    from simbench import adapters
    from simbench.adapters.mujoco_cpu import joint_ranges, load_model
    from simbench.models import model_path
    from simbench.readback import probe_configurations

    path = model_path(args.model)
    configs = probe_configurations(joint_ranges(load_model(path)))
    result = adapters.load(args.route).readback(path, args.route, args.model, configs)
    for rb in result if isinstance(result, list) else [result]:
        _write(SUMMARY / "audit" / args.model / f"{rb.route}.json", rb.to_dict())
        print(f"wrote audit/{args.model}/{rb.route}.json "
              f"({len(rb.bodies)} bodies, {len(rb.joints)} joints, {len(rb.fk_poses)} configs)")


def cmd_audit_compare(args) -> None:
    from simbench.readback import compare

    folder = SUMMARY / "audit" / args.model
    ref = json.loads((folder / f"{args.reference}.json").read_text())
    rows = []
    for path in sorted(folder.glob("*.json")):
        if path.stem in (args.reference, "compare"):
            continue
        report = compare(ref, json.loads(path.read_text()))
        report["fail"] = sorted(k for k, v in report["max_abs_diff"].items()
                                if v > TOLERANCES[k]) + (["missing"] if report["missing"] else [])
        rows.append(report)
    _write(folder / "compare.json", {"tolerances": TOLERANCES, "routes": rows})

    keys = list(TOLERANCES)
    print(f"model {args.model}, reference {args.reference}")
    print(f"{'route':<18}" + "".join(f"{k:>13}" for k in keys) + "  status")
    for r in rows:
        cells = "".join(f"{r['max_abs_diff'][k]:>13.2e}" for k in keys)
        status = "ok" if not r["fail"] else "FAIL " + ",".join(r["fail"])
        print(f"{r['route']:<18}{cells}  {status}")
        for k in r["fail"]:
            if k in r["worst_at"]:
                print(f"{'':<18}  {k}: worst at {r['worst_at'][k]}")
        if r["missing"]:
            print(f"{'':<18}  missing: {', '.join(r['missing'][:6])}")


def cmd_run_plan(args) -> None:
    from simbench.plans import PLANS
    from simbench.run import simulate, trace_stem
    from simbench.scenes import make

    for run in PLANS[args.plan]:
        if args.scene and run.scene not in args.scene:
            continue
        if run.routes is not None and args.route not in run.routes:
            continue
        for seed in run.seeds:
            scene = make(run.scene, **run.overrides)
            tr = simulate(args.route, scene, seed=seed, tag=run.tag, sim_kwargs=run.sim)
            stem = trace_stem(scene, args.route, seed, run.tag)
            tr.save(stem)
            print(f"{args.route:<10} {stem.parent.parent.name}/{stem.parent.name} seed {seed}: "
                  f"{tr.meta['status']} ({tr.meta['frames']} frames, "
                  f"{tr.meta.get('timing', {}).get('loop_s', 0.0):.1f} s)"
                  + (f" {tr.meta['error']}" if tr.meta["error"] else ""), flush=True)


def cmd_make_tapes(args) -> None:
    from simbench import tapes
    from simbench.scenes import tape_specs

    for spec in tape_specs():
        path = tapes.tape_path(spec)
        if path.exists() and not args.force:
            print(f"kept {path.name}")
            continue
        print(f"wrote {tapes.generate(spec).name}", flush=True)


def cmd_bench(args) -> None:
    from simbench import bench
    from simbench.envinfo import environment

    path = SUMMARY / "bench" / f"{args.route}.json"
    out = json.loads(path.read_text()) if path.exists() else {"workloads": {}}
    out["environment"] = environment(args.route)
    worlds = [n for n in bench.WORLDS if n <= args.max_worlds]
    for workload in args.workloads:
        kwargs = {} if args.route.startswith("mujoco") else {"worlds": worlds}
        runs, equiv = bench.BENCH[args.route](workload, **kwargs)
        out["workloads"][workload] = {"runs": runs, "equivalence": equiv}
        _write(path, out)
        for r in runs:
            if "error" in r:
                print(f"{args.route:<10} {workload:<6} n={r['worlds']:<7} ERROR {r['error']}")
                continue
            mem = r.get("gpu_memory_mib")
            print(f"{args.route:<10} {workload:<6} {r['mode']:<10} n={r['worlds']:<7} "
                  f"threads={r.get('threads', '-'):<3} {r['world_steps_per_s']:>14,.0f} "
                  f"world-steps/s  rtf/world {r['rtf_per_world']:>9.3f}"
                  + (f"  {mem:,.0f} MiB" if mem else ""), flush=True)
        if equiv:
            print(f"{'':<10} {workload:<6} equivalence {json.dumps(equiv)}", flush=True)


def cmd_export_web(args) -> None:
    from simbench.web import export

    export()


def cmd_tune(args) -> None:
    from simbench.tuning import tune

    tune(args.route)


def cmd_report(args) -> None:
    from simbench import report
    from simbench.plans import PLANS

    if args.plan == "versions":
        report.version_report()
    elif all(r.base for r in PLANS[args.plan]):
        report.sensitivity_report(args.plan)
    else:
        report.phase_report(args.plan)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="simbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="fetch the pinned Panda, G1 and article reference files")
    p.add_argument("--update", action="store_true", help="rewrite the lock files")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("env-report", help="record versions and smoke-step every model")
    p.add_argument("--route", required=True)
    p.set_defaults(func=cmd_env_report)

    p = sub.add_parser("audit", help="read back an imported model after loading")
    p.add_argument("--route", required=True)
    p.add_argument("--model", choices=MODEL_NAMES, default="panda_article")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("audit-compare", help="compare all route readbacks against a reference")
    p.add_argument("--model", choices=MODEL_NAMES, default="panda_article")
    p.add_argument("--reference", default="mujoco4")
    p.set_defaults(func=cmd_audit_compare)

    p = sub.add_parser("run-plan", help="run every scene/timestep/seed of a plan on one route")
    p.add_argument("--route", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--scene", nargs="*", help="restrict to these scenes")
    p.set_defaults(func=cmd_run_plan)

    p = sub.add_parser("tune", help="Phase 6 push controller tuning and held-out evaluation")
    p.add_argument("--route", required=True)
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("make-tapes", help="generate the frozen Panda reference/feedforward tapes")
    p.add_argument("--force", action="store_true", help="regenerate existing tapes")
    p.set_defaults(func=cmd_make_tapes)

    p = sub.add_parser("bench", help="Phase 5 batched runtime benchmark on one route")
    p.add_argument("--route", required=True)
    p.add_argument("--workloads", nargs="*", default=["slide", "drop", "panda", "push"])
    p.add_argument("--max-worlds", type=int, default=131072)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("export-web", help="write replay data and summaries for web/")
    p.set_defaults(func=cmd_export_web)

    p = sub.add_parser("report", help="metrics for every recorded trace of a plan")
    p.add_argument("--plan", required=True)
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
