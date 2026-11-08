# Reproducing "Comparing physics engines for robotics simulation" with MuJoCo 4.0

Source article: <https://mandarobotics.com/blog/comparing-physics-engines/index.html>
(Manda Robotics, October 5, 2026). The original used MuJoCo 3.11.0, Newton 1.5.2 /
MuJoCo-Warp 3.11.0, Genesis 1.4.1 and Isaac Sim 6.1 (PhysX).

This reproduction replaces MuJoCo 3.11 with MuJoCo 4.0 (released November 6, 2026),
keeps 3.11 as the baseline, and measures what 4.0 changes in contact physics and
runtime.

## Scope

| Route | Package pins | Device | Precision |
| --- | --- | --- | --- |
| MuJoCo 3.11 CPU (article baseline) | `mujoco==3.11.0` | CPU | float64 |
| MuJoCo 4.0 CPU | `mujoco==4.0.0` | CPU | float64 |
| MuJoCo-Warp 4.0 (standalone) | `mujoco-warp==4.0.0`, `warp-lang` | GPU | float32 |
| Newton (MuJoCo-Warp backend) | `newton==1.7.0` (MuJoCo-Warp as pinned by Newton) | GPU | float32 |
| Genesis | `genesis-world==1.4.5` | GPU | float64 |

PhysX / Isaac Sim is out of scope, so every route here uses soft, MuJoCo-style
contact. The article's hard-contact versus soft-contact findings are not reproduced.

Hardware: RTX 5090 (32 GB), NVIDIA driver 580.178.04, 32 logical CPU cores. The
article used an RTX PRO 5000 Blackwell (48 GB), so absolute throughput is not
directly comparable.

Robot asset: `franka_emika_panda` from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), pinned to a
fixed commit. This is the asset the article used. Its visual meshes are also used by
the web viewer.

## Repository layout

```text
simulator_physics/
  PLAN.md
  pyproject.toml            # dev environment (fetch, analysis), ruff / pytest config
  Makefile                  # one-line commands for every step
  packages/simbench/        # engine-agnostic core, depends only on numpy
    scenes/                 # scene specs (geometry, mass, friction, dt, initial state)
    controller.py           # shared 1 kHz PD + inverse-dynamics feedforward
    trace.py                # common trace format: poses, contact forces, efforts
    metrics.py              # travel, peak force, overlap, settle time, momentum check
    adapters/               # one module per engine: build scene, step, read back
  envs/<route>/             # one locked uv project per route (pyproject + uv.lock)
  assets/                   # pinned Menagerie Panda + generated scene MJCF
  reference/                # article audit / metrics files, for parameter matching
  results/                  # raw traces (gitignored); summaries committed
  web/                      # interactive site source
```

Each engine pins different versions of mujoco, warp and torch, so each route gets
its own locked environment. Every run goes through
`uv run --locked --project envs/<route> simbench ...` and writes a trace plus a
metadata file (package versions, git commit, GPU, driver, seed).

## Status

| Phase | State |
| --- | --- |
| 0. Setup and import audit | Done (2026-11-08) |
| 1–6 | Not started |

### Phase 0 results
- Five locked routes install and step both Panda models on the RTX 5090
  (`results/summary/env/`). Newton 1.7.0 resolves to MuJoCo / MuJoCo-Warp 3.15.0
  through its `sim` extra, so the Newton route is not a MuJoCo 4.0 route.
- Menagerie pinned to `822c2d8f877d`, the revision in the article's Panda audit.
  The current head (`98a70c5`) only moves the light from `panda.xml` to `scene.xml`.
- All 44 article reference files fetched and hashed in `reference/manifest.json`.
- Import audit against MuJoCo 4.0 on the raw Menagerie Panda and the article's
  simplified Panda (`panda_article`): every route passes the article's tolerances.
  Worst float32 errors are mass 8.4e-8 kg, COM 2.6e-9 m, inertia 2.6e-8 kg·m², FK
  position 9.6e-7 m (`results/summary/audit/<model>/compare.json`).
- Reproduced the article's import pitfall: Genesis 1.4.5 still substitutes 0.1 kg·m²
  armature where the MJCF specifies zero. The Genesis adapter now sets armature,
  damping and frictionloss from the compiled MJCF and records the import default.
- Genesis silently filters six self-collision pairs on the raw Panda at qpos0;
  revisit when contact scenes use the Panda collision meshes.
- MuJoCo 3.11 vs 4.0: identical except link1 inertia (4.6e-7 kg·m²); 4.0 reproduces
  the authored `fullinertia` exactly, 3.11's principal-axes decomposition does not.
- Newton's generated MjModel names elements by full path (`/` -> `_`); the adapter
  maps them back to MJCF names and audits both the Newton model and that MjModel.

## Phases

### Phase 0: setup and import audit
- Scaffold the repo, create and lock one environment per route.
- Pin the Menagerie Panda by commit.
- Fetch the article's linked `audit.md` / `metrics.json` files into `reference/`.
- Environment report: every route imports, reports versions and steps a model.
- Import audit: after loading in each engine, read back masses, COMs, inertias,
  joint axes and limits, armature and forward kinematics against MuJoCo 4.0.

### Phase 1: single-object controls
Sliding block, drop and settle, single hinged link, at 2 ms and 1 ms, with the
per-step momentum-balance check. Target: MuJoCo 3.11 reproduces the article's
numbers (for example, 126.47 mm travel and a 10.46 N peak for the sliding block).

### Phase 2: multiple objects
Sphere–cube–prism collision chain; three seeds per route; 1 ms and 0.5 ms.

### Phase 3: Panda robot
- Free space: identical open-loop torques, then 2.5 s tracking with the shared
  controller.
- Contact: grasp (centred and 8 mm offset), stacking (0 / 16 / 20 / 22 mm), and
  three-block pushing (90 / 80 mm channel, ±1 mm layout and half-timestep variants).

### Phase 4: what MuJoCo 4.0 changes
- Identical MJCF and options on 3.11 and 4.0; per-scene difference report.
- Speculative contacts on/off for drop, collision chain and fast pushes.
- The article's unresolved sliding-block transient (1.1 mm lift, tenfold force
  peak): 3.11 vs 4.0, elliptic vs pyramidal cones, `impratio`, nativeccd.
- Sparse Cholesky regression in 4.0.0 (mujoco issue #3767): run each scene with the
  Jacobian forced dense and forced sparse under the Newton solver; flag any
  difference as the bug, and rerun once the fixed release ships.
- Repeatability of MuJoCo-Warp 4.0 across seeds in contact scenes.

### Phase 5: runtime benchmark
- Contact-free Panda at 1 ms, no rendering, median of 3 after warm-up.
- MuJoCo CPU 3.11 vs 4.0: one world, and all 32 cores via `mujoco.rollout`.
- MuJoCo-Warp 4.0, Newton and Genesis at 1, 128, 8,192, 32,768 and 131,072 worlds,
  with GPU memory use.
- A contact-heavy variant (pushing scene).
- Check that a batched run reproduces the single-world trajectory.

### Phase 6 (later): remaining scenes
Peg insertion, 15-object crowded sweep (most exposed to the sparse Cholesky bug),
deformable ball (4.0 flex contact and elasticity fixes), controller tuning.

## Interactive website
Static site (Vite + Three.js + uPlot), built to `web/dist`:
- Replay viewer: scene picker, engine toggles, side-by-side or overlay, timeline
  scrubber, force arrows at 1 N = 1 cm.
- Plots linked to the same time cursor as the 3D view.
- MuJoCo 3.11 vs 4.0 toggle per scene with the difference highlighted.
- Benchmark dashboard: log-scale throughput vs batch size, CPU core scaling.
- Tables generated from committed summary JSON; per-scene audit panel with versions,
  hashes and readback checks.

## Risks
- Newton 1.7 may pin an older MuJoCo-Warp than 4.0; record it and rely on the
  standalone MuJoCo-Warp route for 4.0.
- Genesis on Blackwell needs a torch build with CUDA 12.8 or newer.
- The article's scene code is not published; parameters are matched from its audit
  files, and any mismatch is reported as a reproduction difference.
- MuJoCo 4.0.0 sparse Cholesky regression (see Phase 4).
- Node.js is not installed on this machine; the web build either installs it
  (e.g. via `nodeenv` inside the dev environment) or ships plain ES modules.
