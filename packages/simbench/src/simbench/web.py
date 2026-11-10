"""Export compact replay data and summaries for the static site in `web/`.

Run on the MuJoCo 4.0 route (`make web`): geometry comes from the compiled scene
MJCF, trajectories from `results/raw`, tables from `results/summary`.

Binary arrays are base64 little-endian int16 with a per-array scale, so a replay is
`value = int16 * scale`. Positions use 0.05 mm, quaternions 1/32767, forces 0.01 N.
Meshes are shared by all replays in `meshes.json`, keyed by content hash, with a
per-mesh vertex scale.
"""

import base64
import hashlib
import json
import shutil

import numpy as np

from simbench.paths import REPO, SUMMARY, UNITREE
from simbench.plans import PLANS, PRIMARY
from simbench.reference import BUNDLES, ENGINE_LABELS, article_traces
from simbench.report import ARTICLE_VARIANTS, HEADLINE, ROUTE_ORDER, _load
from simbench.scenes import make
from simbench.trace import Trace

WEB = REPO / "web"
DATA = WEB / "data"
MAX_FRAMES = 700
MAX_FACES = 40000
POS_SCALE, QUAT_SCALE, FORCE_SCALE = 5e-5, 1 / 32767, 0.01
# Per-contact forces need a finer scale than net body force: hydroelastic patches are mN each.
CONTACT_FORCE_SCALE = 1e-4
# Drop far zero-force proximity reports (Newton's mesh contact buffer can list thousands).
CONTACT_IDLE_DIST = 0.003

ROUTE_LABELS = {"mujoco311": "MuJoCo 3.11 CPU", "mujoco4": "MuJoCo 4.0 CPU",
                "mjwarp4": "MuJoCo-Warp 4.0", "newton": "Newton 1.7",
                "newton-hydro": "Newton 1.7 hydro", "genesis": "Genesis 1.4.5",
                **{f"article:{k}": v for k, v in ENGINE_LABELS.items()},
                **{f"article:{k}#2": v.replace("(article)", "(article, repeat 2)")
                   for k, v in ENGINE_LABELS.items()}}

# Replays shown on the site: primary variants plus the informative contrasts.
# Insertion hydroelastic Newton runs live in <variant>-hydro folders and are merged
# into each replay as route "newton-hydro" (not listed as separate scene entries).
# Continuum soft-insertion (FEM / flex peg) is its own deformable scene, not a panel
# on rigid peg-insertion.
EXTRA_VARIANTS = {"stack": ["dt1ms-offset24mm"], "push": ["dt1ms-w80mm"],
                  "grasp": ["dt1ms-offset8mm"],
                  "insertion": ["dt1ms-c100-tilted-guided", "dt1ms-c100-offset",
                                "dt1ms-c500-aligned"],
                  "soft-insertion": ["dt1ms-fem-bore", "dt0.05ms-flex-peg",
                                     "dt0.05ms-flex-peg-y0.05"],
                  "ball-compress": ["dt0.05ms-y0.03"],
                  "ball-grasp": ["dt0.05ms-e2N", "dt0.05ms-e1N-stiff", "dt0.05ms-e2N-y0.03",
                                 "dt0.05ms-e1N", "dt0.05ms-e1.1N"],
                  "hinge-plastic": ["dt1ms-elastic"],
                  "card-catch": ["dt1ms-v1-soft", "dt1ms-v3", "dt1ms-v3-speculative",
                                 "dt1ms-v1-speculative", "dt1ms-v1-w30",
                                 "dt1ms-v1-w30-speculative", "dt1ms-v10-speculative"],
                  "tumble": ["dt10ms-rk4", "dt2ms-euler"], "sheet": ["dt0.1ms-elastic"],
                  "twist": ["dt1ms-disc-iw", "dt1ms-disc-hydro-iw", "dt1ms-square"],
                  "g1-walk": [f"dt2ms-{t}-v0.6" for t in ("glass", "sand", "mud", "sponge")]}
# Scenes from contact-free to contact-rich; the site's replay list and scenario ladder.
LADDER = [("No contact", ["hinge", "tumble", "panda-effort", "panda-tracking"]),
          ("One body on a surface", ["slide", "drop", "card-catch"]),
          ("Multi-body impacts", ["collision-chain"]),
          ("Grasping and stacking", ["grasp", "stack"]),
          ("Multi-object manipulation", ["push", "insertion"]),
          ("Patch contact", ["twist"]),
          ("Crowded", ["sweep"]),
          ("Deformable", ["ball-compress", "ball-grasp", "soft-insertion", "cloth-fold"]),
          ("Elasto-plastic (new in 4.0)", ["hinge-plastic", "sheet"]),
          ("Legged locomotion", ["g1-walk"])]
TITLES = {"slide": "Sliding block", "drop": "Drop and settle", "hinge": "Hinged link",
          "hinge-plastic": "Plastic hinge spring",
          "collision-chain": "Collision chain", "panda-effort": "Panda open-loop effort",
          "panda-tracking": "Panda joint tracking", "grasp": "Grasp and lift",
          "stack": "Stacking", "push": "Channel push", "sweep": "Crowded sweep",
          "insertion": "Peg insertion", "soft-insertion": "Soft peg / bore insertion",
          "cloth-fold": "Dual-Franka cloth fold",
          "ball-compress": "Ball compression",
          "ball-grasp": "Soft-ball grasp", "card-catch": "Thin card catch",
          "tumble": "Tumbling T-handle", "sheet": "Plastic sheet bend",
          "twist": "Pinch and twist", "g1-walk": "G1 walking on terrain"}


def _b64(a: np.ndarray, scale: float) -> str:
    q = np.clip(np.round(np.asarray(a, dtype=float) / scale), -32767, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode()


def _b64f(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()


def _overrides(scene: str, variant: str) -> dict:
    for p in PLANS.values():
        for r in p:
            if r.scene == scene and r.variant == variant:
                return r.overrides
    raise KeyError(f"{scene}/{variant}")


def decimate(vert: np.ndarray, face: np.ndarray, max_faces: int = MAX_FACES):
    """Vertex clustering on a grid, coarsened until at most `max_faces` remain."""
    if len(face) <= max_faces:
        return vert, face
    src_vert, src_face = vert, face
    cells = 400
    while len(face) > max_faces and cells >= 8:
        vert, face = src_vert, src_face
        lo, hi = vert.min(axis=0), vert.max(axis=0)
        key = np.floor((vert - lo) / ((hi - lo).max() / cells + 1e-12)).astype(np.int64)
        _, cluster, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        cluster = cluster.ravel()
        merged = np.zeros((len(counts), 3))
        np.add.at(merged, cluster, vert)
        f = cluster[face]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]
        vert, face = merged / counts[:, None], f[np.sort(keep)]
        cells = int(cells * 0.85)
    return vert, face


def _mesh(m, mid: int, meshes: dict) -> str:
    """Content-addressed mesh entry in `meshes` (shared by all replays); returns its key."""
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    vert, face = decimate(m.mesh_vert[va:va + vn], m.mesh_face[fa:fa + fn])
    vert, face = vert.ravel(), face.ravel()
    face = face.astype("<u2" if len(vert) // 3 < 65536 else "<u4")
    key = hashlib.sha1(vert.tobytes() + face.tobytes()).hexdigest()[:12]
    if key not in meshes:
        scale = float(np.abs(vert).max()) / 32767 or 1.0
        meshes[key] = {"vert": _b64(vert, scale), "scale": scale,
                       "face": base64.b64encode(face.tobytes()).decode(),
                       "index_bytes": face.itemsize}
    return key


def geometry(scene_name: str, overrides: dict, meshes: dict) -> dict:
    """Bodies and their visual/colliding geoms from the compiled MuJoCo model."""
    import mujoco

    scene = make(scene_name, **overrides)
    m = mujoco.MjModel.from_xml_path(str(scene.mjcf_path()))
    tracked = {m.body(n).id: i for i, n in enumerate(scene.bodies)}
    types = {int(v): k.replace("mjGEOM_", "").lower() for k, v in
             vars(mujoco.mjtGeom).items() if k.startswith("mjGEOM_")}
    geoms = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        colliding = m.geom_contype[g] or m.geom_conaffinity[g]
        if not colliding and m.geom_group[g] == 3:  # Menagerie collision-only meshes
            continue
        if b != 0 and b not in tracked:
            continue
        rgba = m.geom_rgba[g]
        if m.geom_matid[g] >= 0:
            rgba = m.mat_rgba[m.geom_matid[g]]
        geom = {"body": tracked.get(b, -1), "type": types[int(m.geom_type[g])],
                "size": m.geom_size[g].round(6).tolist(), "pos": m.geom_pos[g].round(6).tolist(),
                "quat": m.geom_quat[g].round(6).tolist(), "rgba": rgba.round(3).tolist(),
                "name": m.geom(g).name}
        mid = int(m.geom_dataid[g])
        if geom["type"] == "mesh" and mid >= 0:
            geom["mesh"] = _mesh(m, mid, meshes)
        geoms.append(geom)
    out = {"bodies": scene.bodies, "joints": scene.joints, "geoms": geoms, "dt": scene.dt,
           "duration": scene.duration, "params": _jsonable(scene.params)}
    if m.nflex:  # deformable surface drawn from its vertex bodies' positions
        prefix = scene.params.get("flex_prefix", "ball_")
        out["flex"] = {"bodies": [i for i, b in enumerate(scene.bodies) if b.startswith(prefix)],
                       "faces": scene.params["faces"], "rgba": m.flex_rgba[0].round(3).tolist()}
    return out


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                 else str(o)))


def replay(tr: Trace, bodies: list[str], stride: int) -> dict:
    """Downsampled poses, COM, net contact forces, joints and per-contact arrows."""
    idx = [tr.body(b) if b in tr.bodies else None for b in bodies]
    k = np.arange(0, len(tr.t), stride)
    # Keep frames that carry contact force so brief yellow dots survive stride — but
    # continuum flex contacts every step, so subsample hot frames to a budget.
    if "c_frame" in tr.arrays and len(tr.c_frame):
        hot = np.unique(tr.c_frame[np.linalg.norm(tr.c_force, axis=1) > 1e-6])
        if len(hot):
            budget = max(MAX_FRAMES, len(k) * 2)
            if len(hot) > budget:
                hot = hot[::int(np.ceil(len(hot) / budget))]
            k = np.unique(np.concatenate([k, hot]))
    nb = len(bodies)

    def gather(arr, width):
        out = np.zeros((len(k), nb, width))
        for j, i in enumerate(idx):
            if i is not None:
                out[:, j] = arr[k, i]
        return out

    out = {"t": _b64f(tr.t[k]), "frames": len(k),
           "pos": _b64(gather(tr.pos, 3), POS_SCALE),
           "quat": _b64(gather(tr.quat, 4), QUAT_SCALE),
           "com": _b64(gather(tr.com, 3), POS_SCALE),
           "force": _b64(gather(tr.contact_net, 3), FORCE_SCALE)}
    if tr.arrays.get("q") is not None and tr.q.size:
        out["q"] = _b64f(tr.q[k].ravel())
        out["nq"] = int(tr.q.shape[1])
    if "c_frame" in tr.arrays and len(tr.c_frame):
        sel = np.isin(tr.c_frame, k)
        mag = np.linalg.norm(tr.c_force[sel], axis=1)
        dist = tr.c_dist[sel] if "c_dist" in tr.arrays else np.zeros(int(sel.sum()))
        keep = (mag > 1e-6) | (dist < CONTACT_IDLE_DIST)
        frame = np.searchsorted(k, tr.c_frame[sel][keep])
        out["contacts"] = {"count": np.bincount(frame, minlength=len(k)).tolist(),
                           "pos": _b64(tr.c_pos[sel][keep], POS_SCALE),
                           "force": _b64(tr.c_force[sel][keep], CONTACT_FORCE_SCALE)}
    # Per-route solver / cloth stack (replay page fair-comparison notes).
    if tr.meta.get("settings"):
        out["settings"] = _jsonable(tr.meta["settings"])
    if tr.meta.get("notes"):
        out["notes"] = list(tr.meta["notes"])
    return out


def export_scene(scene: str, variant: str, meshes: dict, article: bool = True) -> dict | None:
    overrides = _overrides(scene, variant)
    geo = geometry(scene, overrides, meshes)
    routes = {}
    for route in ROUTE_ORDER:
        tr = _load(scene, variant, route)
        if tr is not None and tr.meta["status"] == "ok":
            stride = max(1, int(np.ceil(len(tr.t) / MAX_FRAMES)))
            routes[route] = replay(tr, geo["bodies"], stride)
        # Companion Newton hydroelastic trace (same MJCF, SDF patches) as a sixth panel.
        if route == "newton":
            hydro = _load(scene, f"{variant}-hydro", "newton")
            if hydro is not None and hydro.meta["status"] == "ok":
                stride = max(1, int(np.ceil(len(hydro.t) / MAX_FRAMES)))
                routes["newton-hydro"] = replay(hydro, geo["bodies"], stride)
    bundle = (BUNDLES.get(scene) if PRIMARY.get(scene) == variant
              else ARTICLE_VARIANTS.get(f"{scene}/{variant}"))
    if bundle and article:
        for tr in article_traces(scene, bundle).values():
            stride = max(1, int(np.ceil(len(tr.t) / MAX_FRAMES)))
            routes[tr.meta["route"]] = replay(tr, geo["bodies"], stride)
    if not routes:
        return None
    # Soft-shell scenes: stamp per-route cloth model labels even on older traces.
    engines = (geo.get("params") or {}).get("method", {}).get("engines") or {}
    soft = scene == "cloth-fold"
    for name, blob in routes.items():
        settings = dict(blob.get("settings") or {})
        key = name.replace("-hydro", "")
        if soft and key in ("mujoco4", "mjwarp4"):
            settings.setdefault("cloth_model", "2D flex")
            settings.setdefault("cloth_solver", "MuJoCo flex elasticity" if key == "mujoco4"
                                else "MuJoCo-Warp flex elasticity")
            settings.setdefault("robot_drive", "kinematic-tape")
        elif soft and key == "newton":
            settings.setdefault("cloth_model", settings.pop("cloth", None) or "SolverVBD")
            settings.setdefault("cloth_solver", "newton.solvers.SolverVBD")
            settings.setdefault("robot_drive", settings.pop("robot", None) or "kinematic-tape")
        elif soft and key == "genesis":
            settings.setdefault("cloth_model", settings.pop("cloth", None) or "PBD.Cloth")
            settings.setdefault("cloth_solver", "genesis PBDSolver")
            settings.setdefault("robot_drive", "kinematic-tape")
        if engines.get(key):
            settings.setdefault("cloth_stack", engines[key])
        blob["settings"] = settings
    return {"scene": scene, "variant": variant, "title": TITLES[scene], **geo,
            "routes": routes, "scales": {
                "pos": POS_SCALE, "quat": QUAT_SCALE, "force": FORCE_SCALE,
                "contact_force": CONTACT_FORCE_SCALE}}


def ladder() -> list[dict]:
    """Contact load of each scene's primary variant on MuJoCo 4.0 (native contacts)."""
    out = []
    for group, scenes in LADDER:
        for scene in scenes:
            tr = _load(scene, PRIMARY[scene], "mujoco4")
            if tr is None or not tr.meta["frames"]:
                continue
            counts = (np.bincount(tr.c_frame, minlength=len(tr.t))
                      if "c_frame" in tr.arrays and len(tr.c_frame) else np.zeros(1))
            out.append({"group": group, "scene": scene, "title": TITLES[scene],
                        "variant": PRIMARY[scene], "bodies": len(tr.bodies),
                        "mean_contacts": float(counts.mean()), "peak_contacts": int(counts.max())})
    return out


def export(public: bool = False) -> None:
    """Write web/data. `public` leaves out the article's own traces, which are © Manda Robotics."""
    if (DATA / "replays").exists():
        shutil.rmtree(DATA / "replays")
    (DATA / "replays").mkdir(parents=True)
    index, meshes = [], {}
    for scene, primary in PRIMARY.items():
        for variant in [primary, *EXTRA_VARIANTS.get(scene, [])]:
            data = export_scene(scene, variant, meshes, article=not public)
            if data is None:
                continue
            name = f"{scene}--{variant}.json"
            (DATA / "replays" / name).write_text(json.dumps(data, separators=(",", ":")))
            index.append({"file": name, "scene": scene, "variant": variant,
                          "title": TITLES[scene], "routes": list(data["routes"]),
                          "primary": variant == primary})
            size = (DATA / "replays" / name).stat().st_size / 1e6
            print(f"replays/{name}: {len(data['routes'])} routes, {size:.2f} MB", flush=True)
    (DATA / "meshes.json").write_text(json.dumps(meshes, separators=(",", ":")))
    summaries = {}
    for path in sorted(SUMMARY.rglob("*.json")):
        key = path.relative_to(SUMMARY).with_suffix("").as_posix()
        if key.startswith("audit/") and not key.endswith("/compare"):
            continue
        summaries[key] = json.loads(path.read_text())
    (DATA / "summary.json").write_text(json.dumps(summaries, separators=(",", ":")))
    import time

    (DATA / "index.json").write_text(json.dumps(
        {"replays": index, "routes": ROUTE_LABELS,
         "route_order": [*ROUTE_ORDER[:ROUTE_ORDER.index("newton") + 1], "newton-hydro",
                         *ROUTE_ORDER[ROUTE_ORDER.index("newton") + 1:]],
         "headline": HEADLINE, "ladder_groups": LADDER, "ladder": ladder(),
         "public": public, "build": int(time.time())}, indent=1))
    for licence, name in ((REPO / "assets" / "menagerie" / "franka_emika_panda" / "LICENSE",
                           "PANDA-LICENSE"), (UNITREE / "LICENSE", "UNITREE-LICENSE")):
        if licence.exists():
            shutil.copy(licence, DATA / name)
    print(f"wrote {DATA.relative_to(REPO)} ({len(index)} replays, {len(summaries)} summaries)")
