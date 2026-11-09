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
FLEX = ("mujoco311", "mujoco4", "mjwarp4")


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

CARD_CASES = [(0.001, v, 0.0) for v in (0.3, 1.0, 3.0, 10.0)] + [(0.002, 1.0, 0.0),
                                                                (0.001, 1.0, 30.0)]


_IW, _HYDRO = {"mujoco_invweight": True}, {"contacts": "hydroelastic", "naconmax": 12000,
                                           "njmax": 40000}
# Genesis FEM soft-insertion (polymer E=1e6); lives under scene soft-insertion, not peg insertion.
_FEM_PEG = {"fem_mode": "peg", "E": 1e6, "nu": 0.3, "rho": 1000.0, "friction_mu": 0.5,
            "hydroelastic_modulus": 1e7, "attachment": "coupler",
            "n_mount": 48, "fem_damping": 20.0}
_FEM_BORE = {**_FEM_PEG, "fem_mode": "bore"}
NEWTON_CONTACTS = [
    ("iw", _IW),
    ("point-iw", {"contacts": "newton"} | _IW),
    ("hydro", _HYDRO),
    ("hydro-iw", _HYDRO | _IW),
    ("hydro-full-iw", _HYDRO | _IW | {"reduce": False, "kh": 1e10}),
]


def _card_tag(speed: float, spin: float) -> str:
    return f"v{speed:g}" + (f"-w{spin:g}" if spin else "")


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
    # Article section 4: the crowded sweep's three unchanged-input repeats.
    "phase6": [
        Run("sweep", {"dt": 0.001}, seeds=(0, 1, 2)),
        *_option_runs("sweep", {"dt": 0.001}, "", JACOBIAN),
        Run("sweep", {"dt": 0.001}, seeds=(0, 1), tag="det", sim=DETERMINISTIC["mjwarp4"],
            routes=("mjwarp4",), base="dt1ms"),
        # Precision insertion: 3 clearances x 3 approaches x 3 repeats, plus the
        # contact-guided reruns of the tight offset and tilted cases.
        *[Run("insertion", {"dt": 0.001, "clearance_mm": c, "case": case}, seeds=(0, 1, 2),
              tag=f"c{c * 1e3:03.0f}-{case}")
          for c in (0.10, 0.25, 0.50) for case in ("aligned", "offset", "tilted")],
        *[Run("insertion", {"dt": 0.001, "clearance_mm": 0.10, "case": case, "guided": True},
              seeds=(0, 1, 2), tag=f"c100-{case}-guided", base=f"dt1ms-c100-{case}")
          for case in ("offset", "tilted")],
        # Newton's SDF hydroelastic patches, stored as <variant>-hydro and shown in the
        # web replay as route "newton-hydro" beside the five point-contact engines.
        *[Run("insertion", {"dt": 0.001, "clearance_mm": 0.10, "case": case}, seeds=(0,),
              tag=f"c100-{case}-hydro", sim=_HYDRO | _IW, routes=("newton",),
              base=f"dt1ms-c100-{case}")
          for case in ("aligned", "offset", "tilted")],
        *[Run("insertion", {"dt": 0.001, "clearance_mm": 0.10, "case": case, "guided": True},
              seeds=(0,), tag=f"c100-{case}-guided-hydro", sim=_HYDRO | _IW, routes=("newton",),
              base=f"dt1ms-c100-{case}-guided")
          for case in ("offset", "tilted")],
        Run("insertion", {"dt": 0.001, "clearance_mm": 0.50, "case": "aligned"}, seeds=(0,),
            tag="c500-aligned-hydro", sim=_HYDRO | _IW, routes=("newton",),
            base="dt1ms-c500-aligned"),
        # Deformable soft-insertion (materialized peg/bore): continuum cases, not the
        # rigid peg-insertion matrix. Motion = c100-tilted open impedance.
        Run("soft-insertion", {"dt": 0.001, "clearance_mm": 0.10, "case": "tilted",
                               "kind": "fem-peg"},
            tag="fem-peg", sim=_FEM_PEG, routes=("genesis",)),
        Run("soft-insertion", {"dt": 0.001, "clearance_mm": 0.10, "case": "tilted",
                               "kind": "fem-bore"},
            tag="fem-bore", sim=_FEM_BORE, routes=("genesis",)),
        # MuJoCo flex peg: elastic (M1) and plastic yield (M2). Soft E=20 kPa card
        # (ball suite) with elevated flex damping + 2-layer hand mount; not the same
        # E=1e6 polymer as Genesis A1/A2.
        Run("soft-insertion", {"dt": 5e-5, "clearance_mm": 0.10, "case": "tilted",
                               "kind": "flex-peg", "yield_strain": 0.0},
            tag="flex-peg", routes=("mujoco4", "mjwarp4")),
        Run("soft-insertion", {"dt": 5e-5, "clearance_mm": 0.10, "case": "tilted",
                               "kind": "flex-peg", "yield_strain": 0.05},
            tag="flex-peg-y0.05", routes=("mujoco4", "mjwarp4")),
        # Dual-Franka scripted napkin fold (0.4×0.4 m, cotton-ish). MuJoCo 2D flex;
        # Newton VBD / Genesis PBD rebuild cloth from scene params (not MJCF flex).
        Run("cloth-fold", {"dt": 1e-4}, tag="square", routes=("mujoco4", "mjwarp4")),
        Run("cloth-fold", {"dt": 1e-4}, tag="square", routes=("newton",),
            sim={"cloth_vbd": True}),
        Run("cloth-fold", {"dt": 1e-4}, tag="square", routes=("genesis",),
            sim={"cloth_pbd": True}),
        # Deformable ball (MuJoCo flex routes only): compression control with a repeat
        # and a halved step, then the Panda grasp across the article's effort limits.
        Run("ball-compress", {"dt": 5e-5}, seeds=(0, 1), routes=FLEX),
        Run("ball-compress", {"dt": 2.5e-5}, routes=FLEX),
        # The article's 10 ms pad-ball contact against a 2 ms one (`-stiff`).
        *[Run("ball-grasp", {"dt": 5e-5, "effort_N": e, "contact_tc": 0.002},
              tag=f"e{e:g}N-stiff", routes=FLEX) for e in (1.0, 1.5, 2.0)],
        *[Run("ball-grasp", {"dt": 5e-5, "effort_N": e}, tag=f"e{e:g}N", routes=FLEX)
          for e in (2.0, 1.0, 1.05, 1.1, 1.15, 1.5)],
    ],
    # MuJoCo 4.0 elasto-plasticity (joint springyield, flex yield) and the 4.0 fix of the
    # 3D flex elastic constants. Every route gets the plastic models, so routes without
    # plasticity show whether they reject or silently ignore it.
    "plastic": [
        Run("hinge-plastic", {"dt": 0.001, "yield_deg": 0.0}, tag="elastic"),
        Run("hinge-plastic", {"dt": 0.001, "yield_deg": 5.0}, tag="y5deg"),
        *[Run("ball-compress", {"dt": 5e-5, "yield_strain": y}, tag=f"y{y:g}", routes=FLEX)
          for y in (0.01, 0.03, 0.1)],
        Run("ball-compress", {"dt": 5e-5, "legacy": True}, tag="legacy",
            routes=("mujoco4", "mjwarp4")),
        *[Run("ball-grasp", {"dt": 5e-5, "effort_N": 2.0, "yield_strain": y},
              tag=f"e2N-y{y:g}", routes=FLEX) for y in (0.03, 0.1)],
    ],
    # Other MuJoCo 4.0 changes: speculative contacts on a thin card, fourth-order RK4
    # orientations on a tumbling free body, and 2D shell bending plasticity.
    "mj4new": [
        *[r for dt, v, w in CARD_CASES for r in [
            Run("card-catch", {"dt": dt, "speed": v, "spin": w}, tag=_card_tag(v, w)),
            *_option_runs("card-catch", {"dt": dt, "speed": v, "spin": w}, _card_tag(v, w),
                          [("speculative", {"enable": ["SPECULATIVE"]}, MJ4)])]],
        # The article's 10 ms contact time constant instead of two timesteps.
        *[Run("card-catch", {"dt": 0.001, "speed": v, "contact_tc": 0.01},
              tag=f"v{v:g}-soft") for v in (0.3, 1.0)],
        *[Run("tumble", {"dt": dt, "integrator": "RK4"}, tag="rk4")
          for dt in (0.01, 0.005, 0.002, 0.001)],
        *[Run("tumble", {"dt": dt, "integrator": integ}, tag=integ.lower())
          for integ in ("Euler", "implicitfast") for dt in (0.002, 0.001)],
        Run("sheet", {"dt": 1e-4, "yield_bend": 0.0}, tag="elastic", routes=FLEX),
        *[Run("sheet", {"dt": 1e-4, "yield_bend": y}, tag=f"yb{y:g}", routes=FLEX)
          for y in (0.004, 0.003, 0.002)],
    ],
    # Flat-on-flat patch contact: pinch-and-twist slip torque against the uniform-pressure
    # limit, point contacts on every route against Newton's hydroelastic patches.
    "patch": [
        *[r for shape in ("disc", "square") for r in [
            Run("twist", {"dt": 0.001, "shape": shape}, tag=shape),
            Run("twist", {"dt": 0.001, "shape": shape, "multiccd": True},
                tag=f"{shape}-multiccd", routes=MJ4, base=f"dt1ms-{shape}"),
            Run("twist", {"dt": 0.001, "shape": shape, "condim": 6,
                          "torsion": 2 / 3 * 0.5 * 0.02},
                tag=f"{shape}-condim6", routes=("mujoco4", "mjwarp4"), base=f"dt1ms-{shape}"),
            *[Run("twist", {"dt": 0.001, "shape": shape}, tag=f"{shape}-{name}",
                  sim=sim, routes=("newton",), base=f"dt1ms-{shape}")
              for name, sim in NEWTON_CONTACTS]]],
    ],
    # Unitree G1 with its pretrained walking policy on contact-parameter terrains.
    "terrain": [
        Run("g1-walk", {"dt": 0.002, "terrain": terrain, "speed": speed},
            tag=f"{terrain}-v{speed:g}")
        for terrain in ("flat", "glass", "sand", "mud", "sponge") for speed in (0.3, 0.6, 0.9)
    ],
}

# The variant each scene's headline comparison uses (the article's main figure).
PRIMARY = {"slide": "dt2ms", "drop": "dt1ms", "hinge": "dt1ms", "collision-chain": "dt0.5ms",
           "panda-effort": "dt1ms", "panda-tracking": "dt1ms", "grasp": "dt1ms",
           "stack": "dt1ms-offset20mm", "push": "dt1ms-w90mm", "sweep": "dt1ms",
           "insertion": "dt1ms-c100-tilted", "soft-insertion": "dt1ms-fem-peg",
           "cloth-fold": "dt0.1ms-square",
           "ball-compress": "dt0.05ms",
           "hinge-plastic": "dt1ms-y5deg", "card-catch": "dt1ms-v1",
           "tumble": "dt2ms-rk4", "sheet": "dt0.1ms-yb0.002",
           "ball-grasp": "dt0.05ms-e2N-stiff", "twist": "dt1ms-disc",
           "g1-walk": "dt2ms-flat-v0.6"}
