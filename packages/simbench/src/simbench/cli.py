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
    from simbench.fetch import fetch_menagerie, fetch_reference

    fetch_menagerie(update=args.update)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="simbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="fetch the pinned Menagerie Panda and article reference files")
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
