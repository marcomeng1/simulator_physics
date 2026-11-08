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

from simbench.paths import REPO, SUMMARY
from simbench.plans import PLANS, PRIMARY
from simbench.reference import BUNDLES, ENGINE_LABELS, article_traces
from simbench.report import HEADLINE, ROUTE_ORDER, _load
from simbench.scenes import make
from simbench.trace import Trace

WEB = REPO / "web"
DATA = WEB / "data"
MAX_FRAMES = 700
POS_SCALE, QUAT_SCALE, FORCE_SCALE = 5e-5, 1 / 32767, 0.01

ROUTE_LABELS = {"mujoco311": "MuJoCo 3.11 CPU", "mujoco4": "MuJoCo 4.0 CPU",
                "mjwarp4": "MuJoCo-Warp 4.0", "newton": "Newton 1.7",
                "genesis": "Genesis 1.4.5", **{f"article:{k}": v for k, v in
                                               ENGINE_LABELS.items()}}

# Replays shown on the site: primary variants plus the informative contrasts.
EXTRA_VARIANTS = {"stack": ["dt1ms-offset24mm"], "push": ["dt1ms-w80mm"],
                  "grasp": ["dt1ms-offset8mm"]}
TITLES = {"slide": "Sliding block", "drop": "Drop and settle", "hinge": "Hinged link",
          "collision-chain": "Collision chain", "panda-effort": "Panda open-loop effort",
          "panda-tracking": "Panda joint tracking", "grasp": "Grasp and lift",
          "stack": "Stacking", "push": "Channel push"}


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


def _mesh(m, mid: int, meshes: dict) -> str:
    """Content-addressed mesh entry in `meshes` (shared by all replays); returns its key."""
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    vert = m.mesh_vert[va:va + vn].ravel()
    face = m.mesh_face[fa:fa + fn].ravel()
    face = face.astype("<u2" if vn < 65536 else "<u4")
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
    return {"bodies": scene.bodies, "joints": scene.joints, "geoms": geoms, "dt": scene.dt,
            "duration": scene.duration, "params": _jsonable(scene.params)}


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                 else str(o)))


def replay(tr: Trace, bodies: list[str], stride: int) -> dict:
    """Downsampled poses, COM, net contact forces, joints and per-contact arrows."""
    idx = [tr.body(b) if b in tr.bodies else None for b in bodies]
    k = np.arange(0, len(tr.t), stride)
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
        frame = np.searchsorted(k, tr.c_frame[sel])
        out["contacts"] = {"count": np.bincount(frame, minlength=len(k)).tolist(),
                           "pos": _b64(tr.c_pos[sel], POS_SCALE),
                           "force": _b64(tr.c_force[sel], FORCE_SCALE)}
    return out


def export_scene(scene: str, variant: str, meshes: dict) -> dict | None:
    overrides = _overrides(scene, variant)
    geo = geometry(scene, overrides, meshes)
    stride = max(1, int(np.ceil(geo["duration"] / geo["dt"] / MAX_FRAMES)))
    routes = {}
    for route in ROUTE_ORDER:
        tr = _load(scene, variant, route)
        if tr is not None and tr.meta["status"] == "ok":
            routes[route] = replay(tr, geo["bodies"], stride)
    if PRIMARY.get(scene) == variant and scene in BUNDLES:
        for tr in article_traces(scene).values():
            routes[tr.meta["route"]] = replay(tr, geo["bodies"], stride)
    if not routes:
        return None
    return {"scene": scene, "variant": variant, "title": TITLES[scene], **geo,
            "stride": stride, "routes": routes, "scales": {
                "pos": POS_SCALE, "quat": QUAT_SCALE, "force": FORCE_SCALE}}


def export() -> None:
    (DATA / "replays").mkdir(parents=True, exist_ok=True)
    index, meshes = [], {}
    for scene, primary in PRIMARY.items():
        for variant in [primary, *EXTRA_VARIANTS.get(scene, [])]:
            data = export_scene(scene, variant, meshes)
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
    (DATA / "index.json").write_text(json.dumps(
        {"replays": index, "routes": ROUTE_LABELS, "route_order": ROUTE_ORDER,
         "headline": HEADLINE}, indent=1))
    licence = REPO / "assets" / "menagerie" / "franka_emika_panda" / "LICENSE"
    if licence.exists():
        shutil.copy(licence, DATA / "PANDA-LICENSE")
    print(f"wrote {DATA.relative_to(REPO)} ({len(index)} replays, {len(summaries)} summaries)")
