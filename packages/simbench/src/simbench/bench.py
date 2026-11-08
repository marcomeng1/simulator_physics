"""Phase 5 runtime benchmark: native batched stepping without trace recording.

Extends the article's native batching probe (runtime/RUNTIME-PROTOCOL, part B):
one untimed warm-up trial (compilation, graph capture), then three timed trials,
each starting from the same reset state, synchronized at the trial boundaries.
Reported: aggregate world-steps/s and per-world real-time factor, setup and warm-up
time, and GPU memory held by the process.

Workloads (dt = 1 ms)
  slide, drop  the Phase 1 cube scenes (contact, no control)
  panda        free-space Panda with constant gravity-feedforward torque (no contact)
  push         three-block channel push with the PD tape controller (contact-heavy);
               on GPU routes the controller runs on the device inside the step

Routes
  mujoco311/mujoco4  one world with mj_step; plus `mujoco.rollout` on 1-32 threads
  mjwarp4            native MuJoCo-Warp, one CUDA graph per step
  newton             Newton SolverMuJoCo over replicated worlds, CUDA graph per 2 steps
  genesis            native n_envs batching (no graph capture API)
"""

import os
import subprocess
import time

import numpy as np

from simbench.scenes import make

WORKLOADS = {
    "slide": lambda: make("slide", dt=0.001, duration=1.0),
    "drop": lambda: make("drop", dt=0.001),
    "panda": lambda: make("panda-effort", dt=0.001, duration=1.0),
    "push": lambda: make("push", dt=0.001, width_mm=90.0),
}
# Per-world contact and constraint capacities for the GPU routes.
CAPACITY = {"slide": (8, 32), "drop": (8, 32), "panda": (1, 16), "push": (64, 320)}
WORLDS = [1, 128, 1024, 8192, 32768, 131072]
THREADS = [1, 8, 16, 32]
ROLLOUT_BATCH = 256
TRIALS = 3


def trial_steps(n: int) -> int:
    """1,000 steps (1 s) per trial, fewer for very large batches to bound wall time."""
    return int(max(100, min(1000, 4_000_000 // n)))


def gpu_memory_mib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.stdout.splitlines():
        pid, mem = (s.strip() for s in line.split(","))
        if int(pid) == os.getpid():
            return float(mem)
    return None


def _initial(scene):
    """Initial state prepared with the MuJoCo CPU adapter (same compiler and conventions)."""
    from simbench.adapters.mujoco_cpu import Sim

    return Sim(scene)


EQUIVALENCE_WORLDS = (1, 128)
EQUIVALENCE_STEPS = 1000


def equivalence(trajs: dict[int, list[np.ndarray]]) -> dict:
    """Batched worlds vs the one-world run, and one-world run vs its repeat.

    trajs[n] holds (steps, n, nq) generalized-position trajectories; max |dq| over
    every step and world (positions in m or rad; free-joint quaternions included).
    """
    out = {}
    if 1 in trajs and len(trajs[1]) == 2:
        out["repeat_n1_max_dq"] = float(np.abs(trajs[1][0] - trajs[1][1]).max())
    for n, (tr, *_) in trajs.items():
        if n != 1 and 1 in trajs:
            out[f"batch_n{n}_vs_n1_max_dq"] = float(np.abs(tr - trajs[1][0]).max())
    return out


def _measure(run_trial, reset, n: int, steps: int, dt: float, sync) -> dict:
    start = time.perf_counter()
    reset()
    run_trial(steps)
    sync()
    warmup = time.perf_counter() - start
    times = []
    for _ in range(TRIALS):
        reset()
        sync()
        start = time.perf_counter()
        run_trial(steps)
        sync()
        times.append(time.perf_counter() - start)
    med = float(np.median(times))
    return {"worlds": n, "steps": steps, "trials_s": times, "warmup_s": warmup,
            "world_steps_per_s": n * steps / med, "rtf_per_world": steps * dt / med,
            "gpu_memory_mib": gpu_memory_mib()}


# --- MuJoCo CPU ----------------------------------------------------------------------


def bench_mujoco_cpu(workload: str) -> list[dict]:
    import mujoco
    from mujoco import rollout

    scene = WORKLOADS[workload]()
    cpu = _initial(scene)
    m, d = cpu.m, cpu.d
    tau = np.zeros(m.nv)
    if workload == "panda":
        tau[cpu.dofadr] = scene.effort_at(0.0, None, None)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    s0 = np.zeros(mujoco.mj_stateSize(m, state_spec))
    mujoco.mj_getState(m, d, s0, state_spec)
    results = []

    def reset():
        mujoco.mj_setState(m, d, s0, state_spec)
        d.qfrc_applied[:] = tau
        if scene.controller is not None:
            scene.controller.u = None

    def run_trial(steps):
        for k in range(steps):
            if scene.controller is not None:
                q, qd = d.qpos[cpu.qposadr], d.qvel[cpu.dofadr]
                d.qfrc_applied[cpu.dofadr] = scene.controller(k * scene.dt, q, qd)
            mujoco.mj_step(m, d)

    t0 = time.perf_counter()
    row = _measure(run_trial, reset, 1, 1000, scene.dt, lambda: None)
    results.append({"mode": "mj_step", "threads": 1, "setup_s": time.perf_counter() - t0, **row})
    if scene.controller is not None:
        return results, {}  # rollout replays open-loop inputs only

    init = np.tile(s0, (ROLLOUT_BATCH, 1))
    control = np.tile(tau, (ROLLOUT_BATCH, 1000, 1))
    spec = int(mujoco.mjtState.mjSTATE_QFRC_APPLIED)
    for threads in THREADS:
        datas = [mujoco.MjData(m) for _ in range(threads)]

        def run_rollout(steps, datas=datas):
            rollout.rollout(m, datas, init, control[:, :steps], control_spec=spec, nstep=steps,
                            persistent_pool=True)

        row = _measure(run_rollout, lambda: None, ROLLOUT_BATCH, 1000, scene.dt, lambda: None)
        results.append({"mode": "rollout", "threads": threads, **row})
        rollout.shutdown_persistent_pool()
    return results, {}


# --- MuJoCo-Warp and Newton share the device-side PD controller ----------------------


def _warp_kernels():
    import warp as wp

    @wp.kernel
    def pd_2d(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
              step: wp.array(dtype=int), qref: wp.array2d(dtype=float),
              qdref: wp.array2d(dtype=float), ff: wp.array2d(dtype=float),
              gains: wp.array2d(dtype=float), qadr: wp.array(dtype=int),
              dadr: wp.array(dtype=int), qfrc: wp.array2d(dtype=float)):
        w, j = wp.tid()
        k = wp.min(step[0], qref.shape[0] - 1)
        u = (ff[k, j] + gains[0, j] * (qref[k, j] - qpos[w, qadr[j]])
             + gains[1, j] * (qdref[k, j] - qvel[w, dadr[j]]))
        qfrc[w, dadr[j]] = wp.clamp(u, -gains[2, j], gains[2, j])

    @wp.kernel
    def pd_flat(q: wp.array(dtype=float), qd: wp.array(dtype=float), nq: int, nd: int,
                step: wp.array(dtype=int), qref: wp.array2d(dtype=float),
                qdref: wp.array2d(dtype=float), ff: wp.array2d(dtype=float),
                gains: wp.array2d(dtype=float), qadr: wp.array(dtype=int),
                dadr: wp.array(dtype=int), f: wp.array(dtype=float)):
        w, j = wp.tid()
        k = wp.min(step[0], qref.shape[0] - 1)
        u = (ff[k, j] + gains[0, j] * (qref[k, j] - q[w * nq + qadr[j]])
             + gains[1, j] * (qdref[k, j] - qd[w * nd + dadr[j]]))
        f[w * nd + dadr[j]] = wp.clamp(u, -gains[2, j], gains[2, j])

    @wp.kernel
    def advance(step: wp.array(dtype=int)):
        step[0] = step[0] + 1

    return pd_2d, pd_flat, advance


def _tape_arrays(scene):
    import warp as wp

    ctl = scene.controller
    gains = np.stack([ctl.kp, ctl.kd, ctl.limit]).astype(np.float32)
    return {k: wp.array(np.asarray(v, dtype=np.float32), dtype=float)
            for k, v in (("qref", ctl.qref), ("qdref", ctl.qdref), ("ff", ctl.ff),
                         ("gains", gains))}


def bench_mujoco_warp(workload: str, worlds=WORLDS) -> list[dict]:
    import mujoco_warp as mjw
    import warp as wp

    scene = WORKLOADS[workload]()
    cpu = _initial(scene)
    if workload == "panda":
        cpu.d.qfrc_applied[cpu.dofadr] = scene.effort_at(0.0, None, None)
    nconmax, njmax = CAPACITY[workload]
    pd_2d, _, advance = _warp_kernels()
    ctl = _tape_arrays(scene) if scene.controller is not None else None
    qadr = wp.array(cpu.qposadr.astype(np.int32), dtype=int)
    dadr = wp.array(cpu.dofadr.astype(np.int32), dtype=int)
    results, trajs = [], {}
    for n in worlds:
        try:
            t0 = time.perf_counter()
            m = mjw.put_model(cpu.m)
            d = mjw.put_data(cpu.m, cpu.d, nworld=n, nconmax=nconmax, njmax=njmax)
            init = {k: wp.clone(getattr(d, k)) for k in
                    ("qpos", "qvel", "qacc_warmstart", "qfrc_applied", "time")}
            step = wp.zeros(1, dtype=int)

            def one_step(m=m, d=d, step=step, n=n):
                if ctl is not None:
                    wp.launch(pd_2d, dim=(n, len(cpu.dofadr)),
                              inputs=[d.qpos, d.qvel, step, ctl["qref"], ctl["qdref"],
                                      ctl["ff"], ctl["gains"], qadr, dadr, d.qfrc_applied])
                mjw.step(m, d)
                wp.launch(advance, dim=1, inputs=[step])

            one_step()  # compile kernels before capture
            with wp.ScopedCapture() as capture:
                one_step()
            graph = capture.graph
            setup = time.perf_counter() - t0

            def reset(d=d, init=init, step=step):
                for k, v in init.items():
                    wp.copy(getattr(d, k), v)
                step.zero_()

            def run_trial(steps, graph=graph):
                for _ in range(steps):
                    wp.capture_launch(graph)

            row = _measure(run_trial, reset, n, trial_steps(n), scene.dt, wp.synchronize)
            row["finite"] = bool(np.isfinite(d.qpos.numpy()).all())
            row["nacon_last"] = int(d.nacon.numpy()[0])
            results.append({"mode": "cuda_graph", "setup_s": setup, **row})
            if n in EQUIVALENCE_WORLDS:
                for _ in range(2 if n == 1 else 1):
                    reset()
                    traj = []
                    for _ in range(EQUIVALENCE_STEPS):
                        wp.capture_launch(graph)
                        traj.append(d.qpos.numpy().astype(float))
                    trajs.setdefault(n, []).append(np.array(traj))
        except Exception as exc:  # out-of-memory and capacity failures are results
            results.append({"worlds": n, "error": f"{type(exc).__name__}: {exc}"[:300]})
            break
        finally:
            d = m = init = graph = None
            wp.synchronize()
    return results, equivalence(trajs)


def bench_newton(workload: str, worlds=WORLDS) -> list[dict]:
    import newton
    import warp as wp

    from simbench.adapters.newton_mjwarp import Sim as NewtonSim

    scene = WORKLOADS[workload]()
    single = NewtonSim(scene)
    q0, qd0 = single.s0.joint_q.numpy(), single.s0.joint_qd.numpy()
    nq, nd = len(q0), len(qd0)
    f0 = np.zeros(nd, dtype=np.float32)
    if workload == "panda":
        f0[single.qd_idx] = scene.effort_at(0.0, None, None)
    nconmax, njmax = CAPACITY[workload]
    _, pd_flat, advance = _warp_kernels()
    ctl = _tape_arrays(scene) if scene.controller is not None else None
    qadr = wp.array(single.q_idx.astype(np.int32), dtype=int)
    dadr = wp.array(single.qd_idx.astype(np.int32), dtype=int)
    results, trajs = [], {}
    for n in worlds:
        try:
            t0 = time.perf_counter()
            sub = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(sub)
            sub.add_mjcf(str(scene.mjcf_path()))
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            builder.replicate(sub, n)
            model = builder.finalize()
            solver = newton.solvers.SolverMuJoCo(
                model, solver="newton", integrator="euler", cone="elliptic", iterations=100,
                tolerance=1e-10, nconmax=nconmax, njmax=njmax)
            s0, s1, control = model.state(), model.state(), model.control()
            init_q = wp.array(np.tile(q0, n), dtype=float)
            init_qd = wp.array(np.tile(qd0, n), dtype=float)
            init_f = wp.array(np.tile(f0, n), dtype=float)
            step = wp.zeros(1, dtype=int)

            def half(a, b, solver=solver, control=control, step=step, n=n):
                if ctl is not None:
                    wp.launch(pd_flat, dim=(n, len(single.qd_idx)),
                              inputs=[a.joint_q, a.joint_qd, nq, nd, step, ctl["qref"],
                                      ctl["qdref"], ctl["ff"], ctl["gains"], qadr, dadr,
                                      control.joint_f])
                solver.step(a, b, control, None, scene.dt)
                wp.launch(advance, dim=1, inputs=[step])

            def reset(s0=s0, model=model, control=control, step=step, solver=solver,
                      init_q=init_q, init_qd=init_qd, init_f=init_f):
                wp.copy(s0.joint_q, init_q)
                wp.copy(s0.joint_qd, init_qd)
                wp.copy(control.joint_f, init_f)
                newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
                solver.mjw_data.qacc_warmstart.zero_()
                step.zero_()

            reset()
            half(s0, s1)
            half(s1, s0)  # compile kernels before capture
            with wp.ScopedCapture() as capture:
                half(s0, s1)
                half(s1, s0)
            graph = capture.graph
            setup = time.perf_counter() - t0

            def run_trial(steps, graph=graph):
                for _ in range(steps // 2):
                    wp.capture_launch(graph)

            row = _measure(run_trial, reset, n, trial_steps(n), scene.dt, wp.synchronize)
            row["finite"] = bool(np.isfinite(s0.joint_q.numpy()).all())
            results.append({"mode": "cuda_graph", "setup_s": setup, **row})
            if n in EQUIVALENCE_WORLDS:
                for _ in range(2 if n == 1 else 1):
                    reset()
                    traj = []
                    for _ in range(EQUIVALENCE_STEPS // 2):
                        wp.capture_launch(graph)
                        traj.append(s0.joint_q.numpy().astype(float).reshape(n, nq))
                    trajs.setdefault(n, []).append(np.array(traj))
        except Exception as exc:
            results.append({"worlds": n, "error": f"{type(exc).__name__}: {exc}"[:300]})
            break
        finally:
            model = solver = s0 = s1 = control = graph = None
            wp.synchronize()
    return results, equivalence(trajs)


# --- Genesis -------------------------------------------------------------------------


def bench_genesis(workload: str, worlds=WORLDS) -> list[dict]:
    import genesis as gs
    import torch

    from simbench.adapters import genesis_rigid as g

    scene = WORKLOADS[workload]()
    single = g.Sim(scene)
    q0 = single.entity.get_qpos().detach().clone()
    v0 = single.entity.get_dofs_velocity().detach().clone()
    path = scene.mjcf_path()
    opts = g.rigid_options(path, scene.dt)
    gravity, dt = opts.pop("gravity"), opts.pop("dt")
    tau0 = torch.zeros(single.entity.n_dofs, dtype=gs.tc_float, device=gs.device)
    if workload == "panda":
        tau0[single.dofs] = torch.as_tensor(scene.effort_at(0.0, None, None), dtype=gs.tc_float,
                                            device=gs.device)
    ctl, dofs = scene.controller, single.dofs
    tape = None
    if ctl is not None:
        tape = {k: torch.as_tensor(np.asarray(v), dtype=gs.tc_float, device=gs.device)
                for k, v in (("qref", ctl.qref), ("qdref", ctl.qdref), ("ff", ctl.ff),
                             ("kp", ctl.kp), ("kd", ctl.kd), ("lim", ctl.limit))}
    results, trajs = [], {}
    for n in worlds:
        try:
            t0 = time.perf_counter()
            gscene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
                              rigid_options=gs.options.RigidOptions(**opts), show_viewer=False)
            entity = gscene.add_entity(gs.morphs.MJCF(file=str(path)))
            gscene.build(n_envs=n)
            g.apply_mjcf_dof_properties(path, entity)
            setup = time.perf_counter() - t0

            def reset(entity=entity, n=n):
                entity.set_qpos(q0.repeat(n, 1))
                entity.set_dofs_velocity(v0.repeat(n, 1))
                entity.control_dofs_force(tau0.repeat(n, 1))

            def run_trial(steps, gscene=gscene, entity=entity, record=None):
                for k in range(steps):
                    if tape is not None:
                        kk = min(k, len(tape["qref"]) - 1)
                        q = entity.get_dofs_position(dofs)
                        qd = entity.get_dofs_velocity(dofs)
                        u = (tape["ff"][kk] + tape["kp"] * (tape["qref"][kk] - q)
                             + tape["kd"] * (tape["qdref"][kk] - qd))
                        entity.control_dofs_force(torch.clamp(u, -tape["lim"], tape["lim"]),
                                                  dofs)
                    gscene.step()
                    if record is not None:
                        record.append(entity.get_qpos().detach().cpu().numpy().astype(float))

            row = _measure(run_trial, reset, n, trial_steps(n), scene.dt, torch.cuda.synchronize)
            row["finite"] = bool(torch.isfinite(entity.get_qpos()).all())
            results.append({"mode": "n_envs", "setup_s": setup, **row})
            if n in EQUIVALENCE_WORLDS:
                for _ in range(2 if n == 1 else 1):
                    reset()
                    traj = []
                    run_trial(EQUIVALENCE_STEPS, record=traj)
                    trajs.setdefault(n, []).append(np.array(traj).reshape(len(traj), n, -1))
            if hasattr(gscene, "destroy"):
                gscene.destroy()
        except Exception as exc:
            results.append({"worlds": n, "error": f"{type(exc).__name__}: {exc}"[:300]})
            break
        finally:
            gscene = entity = None
            torch.cuda.empty_cache()
    return results, equivalence(trajs)


BENCH = {"mujoco311": bench_mujoco_cpu, "mujoco4": bench_mujoco_cpu,
         "mjwarp4": bench_mujoco_warp, "newton": bench_newton, "genesis": bench_genesis}
