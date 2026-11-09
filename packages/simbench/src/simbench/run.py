"""Route-independent simulation loop: record, compute effort, step, read contacts."""

import random
import time

import numpy as np

from simbench import adapters
from simbench.envinfo import environment
from simbench.paths import RAW
from simbench.scene import Scene
from simbench.trace import Trace, TraceBuilder


def trace_stem(scene: Scene, route: str, seed: int, tag: str = ""):
    variant = f"dt{scene.dt * 1e3:g}ms" + (f"-{tag}" if tag else "")
    return RAW / scene.name / variant / f"{route}-seed{seed}"


def simulate(route: str, scene: Scene, seed: int = 0, tag: str = "",
             sim_kwargs: dict | None = None) -> Trace:
    random.seed(seed)
    np.random.seed(seed)
    adapter = adapters.load(route)
    start = time.perf_counter()
    try:
        sim = adapter.Sim(scene, seed=seed, **(sim_kwargs or {}))
    except Exception as exc:  # unsupported option combinations are results too
        return Trace({"schema": "simbench.trace.v1", "route": route, "scene": scene.name,
                      "scene_sha256": scene.sha256, "params": scene.params, "tag": tag,
                      "sim_kwargs": sim_kwargs or {}, "dt": scene.dt, "seed": seed,
                      "bodies": scene.bodies, "joints": scene.joints, "contact_names": [],
                      "status": "error", "error": f"{type(exc).__name__} at build: {exc}"[:500],
                      "frames": 0, "environment": environment(route)}, {})
    build_s = time.perf_counter() - start

    stride = scene.stride
    rec = TraceBuilder(-(-scene.steps // stride), len(scene.bodies), len(scene.joints))
    status, error = "ok", None
    # Contact-aware controllers see each step's contacts and may stop the run.
    observe = getattr(scene.controller, "observe", None)
    start = time.perf_counter()
    for k in range(scene.steps):
        t = k * scene.dt
        state = sim.state()
        if not all(np.all(np.isfinite(state[key])) for key in ("pos", "quat", "linvel")):
            status, error = "nonfinite", f"non-finite state at t={t:.6f}s"
            break
        tau = scene.effort_at(t, state["q"], state["qd"], state)
        # Optional pre-step hook (e.g. cloth-fold kinematic joint tape).
        prepare = getattr(scene.controller, "prepare", None)
        if prepare is not None:
            prepare(t, sim)
        try:
            sim.step(tau)
        except Exception as exc:  # engine-level numerical failures are results, not crashes
            status, error = "error", f"{type(exc).__name__} at t={t:.6f}s: {exc}"
            break
        # Optional controller hook (e.g. cloth-fold closed-finger particle attach).
        post = getattr(scene.controller, "post_step", None)
        if post is not None:
            post(t, sim)
            state = sim.state()  # record/observe the attached state
        if observe is None and k % stride:
            continue
        contacts = sim.contacts()
        if k % stride == 0:
            rec.record(k // stride, t, state, tau, contacts)
        if observe is not None and (stop := observe(t, contacts, sim.contact_names)):
            status, error = stop
            break
    loop_s = time.perf_counter() - start

    meta = {
        "schema": "simbench.trace.v1",
        "route": route,
        "scene": scene.name,
        "scene_sha256": scene.sha256,
        "params": scene.params,
        "tag": tag,
        "sim_kwargs": sim_kwargs or {},
        "dt": scene.dt,
        "seed": seed,
        "bodies": scene.bodies,
        "joints": scene.joints,
        "contact_names": sim.contact_names,
        "status": status,
        "error": error,
        "frames": rec.frames,
        "settings": sim.settings(),
        "effective_model": sim.effective_model(),
        "notes": getattr(sim, "notes", []),
        "timing": {"build_s": build_s, "loop_s": loop_s, "note": "includes recording"},
        "environment": environment(route),
    }
    return rec.build(meta)
