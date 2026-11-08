"""Frozen run plans: which scenes, timesteps and seeds each phase runs on every route."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Run:
    scene: str
    overrides: dict = field(default_factory=dict)
    seeds: tuple[int, ...] = (0,)
    tag: str = ""
    # Adapter keyword arguments (solver options, determinism) and the routes that take them.
    sim: dict = field(default_factory=dict)
    routes: tuple[str, ...] | None = None
    # Variant directory this run is compared against (sensitivity runs).
    base: str | None = None

    @property
    def variant(self) -> str:
        return f"dt{self.overrides.get('dt', 0) * 1e3:g}ms" + (f"-{self.tag}" if self.tag else "")


CPU = ("mujoco311", "mujoco4")
MJ4 = ("mujoco4",)


def _option_runs(scene, overrides, base_tag, variants):
    """One run per (name, options, routes), tagged `<base_tag>-<name>`."""
    dt = f"dt{overrides['dt'] * 1e3:g}ms"
    base = dt + (f"-{base_tag}" if base_tag else "")
    return [Run(scene, overrides, tag="-".join(filter(None, [base_tag, name])),
                sim={"options": options}, routes=routes, base=base)
            for name, options, routes in variants]


CONTACT_OPTIONS = [
    ("pyramidal", {"cone": "PYRAMIDAL"}, CPU),
    ("nonativeccd", {"disable": ["NATIVECCD"]}, CPU),
    ("speculative", {"enable": ["SPECULATIVE"]}, MJ4),
    ("discrete", {"integrator": "DISCRETE"}, MJ4),
    ("ipc", {"integrator": "DISCRETE", "enable": ["IPC"]}, MJ4),  # IPC requires DISCRETE
]
JACOBIAN = [("jac-dense", {"jacobian": "DENSE"}, CPU), ("jac-sparse", {"jacobian": "SPARSE"}, CPU)]
DETERMINISTIC = {"mjwarp4": {"deterministic": True}, "newton": {"deterministic": "RUN_TO_RUN"}}

PLANS: dict[str, list[Run]] = {
    "phase1": [
        Run("slide", {"dt": 0.002}, seeds=(0, 1)),
        Run("slide", {"dt": 0.001}),
        Run("drop", {"dt": 0.001}, seeds=(0, 1)),
        Run("drop", {"dt": 0.002}),
        Run("hinge", {"dt": 0.001}, seeds=(0, 1, 2)),
        Run("hinge", {"dt": 0.0005}),
    ],
    "phase2": [
        Run("collision-chain", {"dt": 0.0005}, seeds=(0, 1, 2)),
        Run("collision-chain", {"dt": 0.001}),
    ],
    "phase3": [
        Run("panda-effort", {"dt": 0.001}, seeds=(0, 1)),
        Run("panda-effort", {"dt": 0.0005}),
        Run("panda-tracking", {"dt": 0.001}, seeds=(0, 1)),
        Run("panda-tracking", {"dt": 0.0005}),
        Run("grasp", {"dt": 0.001}, seeds=(0, 1)),
        Run("grasp", {"dt": 0.001, "offset_mm": 8.0}, tag="offset8mm"),
        *[Run("stack", {"dt": 0.001, "offset_mm": o}, tag=f"offset{o:g}mm")
          for o in (0.0, 16.0, 22.0, 24.0, 26.0)],
        Run("stack", {"dt": 0.001, "offset_mm": 20.0}, seeds=(0, 1), tag="offset20mm"),
        Run("stack", {"dt": 0.001, "offset_mm": 20.0, "perturb_mm": (1.0, -1.0, 0.0)},
            seeds=(1,), tag="offset20mm-p1"),
        Run("stack", {"dt": 0.001, "offset_mm": 20.0, "perturb_mm": (-1.0, 1.0, 0.0)},
            seeds=(2,), tag="offset20mm-p2"),
        Run("stack", {"dt": 0.0005, "offset_mm": 20.0}, tag="offset20mm"),
        *[Run("push", {"dt": 0.001, "width_mm": w}, tag=f"w{w:g}mm") for w in (100.0, 80.0)],
        Run("push", {"dt": 0.001, "width_mm": 90.0}, seeds=(0, 1), tag="w90mm"),
        *[Run("push", {"dt": 0.001, "width_mm": 90.0, "layout_mm": d}, tag=f"w90mm-l{d:+g}mm")
          for d in (1.0, -1.0)],
        Run("push", {"dt": 0.0005, "width_mm": 90.0}, tag="w90mm"),
    ],
    # What MuJoCo 4.0 changes: option sensitivity on the CPU routes (same MJCF), the
    # sparse-vs-dense Jacobian check for mujoco#3767, and GPU determinism switches.
    "phase4": [
        *_option_runs("slide", {"dt": 0.002}, "", [
            *CONTACT_OPTIONS, ("impratio10", {"impratio": 10.0}, CPU)]),
        *_option_runs("drop", {"dt": 0.001}, "", CONTACT_OPTIONS),
        *_option_runs("collision-chain", {"dt": 0.0005}, "", [
            *CONTACT_OPTIONS, ("nomulticcd", {"disable": ["MULTICCD"]}, CPU), *JACOBIAN]),
        *_option_runs("grasp", {"dt": 0.001}, "", JACOBIAN),
        *_option_runs("stack", {"dt": 0.001, "offset_mm": 20.0}, "offset20mm", JACOBIAN),
        *_option_runs("push", {"dt": 0.001, "width_mm": 90.0}, "w90mm", [
            *JACOBIAN, ("pyramidal", {"cone": "PYRAMIDAL"}, CPU),
            ("speculative", {"enable": ["SPECULATIVE"]}, MJ4)]),
        *[Run(scene, overrides, seeds=seeds, tag="-".join(filter(None, [tag, "det"])),
              sim=kwargs, routes=(route,), base=f"dt{overrides['dt'] * 1e3:g}ms"
              + (f"-{tag}" if tag else ""))
          for scene, overrides, tag, seeds in [
              ("collision-chain", {"dt": 0.0005}, "", (0, 1, 2)),
              ("push", {"dt": 0.001, "width_mm": 90.0}, "w90mm", (0, 1)),
              ("stack", {"dt": 0.001, "offset_mm": 20.0}, "offset20mm", (0, 1))]
          for route, kwargs in DETERMINISTIC.items()],
        # Article's Genesis drop (53.4 N, 7.57 mm) vs ours (= MuJoCo): default options?
        *[Run(scene, {"dt": dt}, tag="gsdefault", sim={"matched_options": False},
              routes=("genesis",), base=f"dt{dt * 1e3:g}ms")
          for scene, dt in (("drop", 0.001), ("slide", 0.002))],
    ],
}

# The variant each scene's headline comparison uses (the article's main figure).
PRIMARY = {"slide": "dt2ms", "drop": "dt1ms", "hinge": "dt1ms", "collision-chain": "dt0.5ms",
           "panda-effort": "dt1ms", "panda-tracking": "dt1ms", "grasp": "dt1ms",
           "stack": "dt1ms-offset20mm", "push": "dt1ms-w90mm"}
