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
    scenes/                 # scene specs; panda.py holds the shared 1 kHz PD controller
    tapes.py                # IK, reference and feedforward tapes (generated on 3.11)
    trace.py                # common trace format: poses, contact forces, efforts
    metrics.py              # per-scene metrics and the momentum check
    plans.py, report.py     # run plans per phase, summaries and version/option reports
    bench.py                # batched runtime benchmark per route
    web.py                  # export of replays and summaries for the site
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
| 1. Single-object controls | Done (2026-11-08) |
| 2. Collision chain | Done (2026-11-08) |
| 3. Panda (free space, grasp, stack, push) | Done (2026-11-08) |
| 4. What MuJoCo 4.0 changes | Done (2026-11-08) |
| 5. Runtime benchmark | Done (2026-11-08) |
| Website | Done (2026-11-08), `make web serve` |
| 6. Insertion, sweep, deformables | Not started |

Numbers below come from `results/summary/`; "path" means the largest COM distance
between two traces over all bodies and frames.

### Phase 1 results (`phase1.json`)
- MuJoCo 3.11 reproduces the article's MuJoCo traces exactly: sliding block 126.471 mm
  travel and 10.458 N peak at 2 ms, drop 52.039 N peak at 1 ms, hinge 0.0839° error.
- MuJoCo 4.0 is identical to 3.11 on all three (path ≤ 5e-12 m).
- MuJoCo-Warp 4.0 and Newton are within 2 µm of MuJoCo 4.0, with momentum residuals
  of about 1e-6 N·s from float32.
- Genesis 1.4.5 with matched options reproduces the article's Genesis slide (1.52 N
  vs 1.50 N peak) but not its drop (52.039 N vs the article's 53.44 N).

### Phase 2 results (`phase2.json`)
- MuJoCo 3.11 matches the article's MuJoCo chain exactly. MuJoCo 4.0 ends 33 mm away
  at 0.5 ms (18.7 mm at 1 ms) because it generates more prism contacts (prism-ground
  1.74 → 2.62 and cube-prism 1.05 → 2.02 contacts per step).
- With that change, MuJoCo 4.0 lands closer to the GPU routes than 3.11 does:
  MuJoCo-Warp 4.0 5.3 mm, Genesis 8.6 mm, Newton 16.6 mm.
- GPU seed spread: MuJoCo-Warp 0.85 mm, Newton 2.55 mm, Genesis 0.

### Phase 3 results (`phase3.json`)
- Panda models: the article's free-space Panda (fingers welded, with the article's
  +0.04 m offset on both fingers reproduced) and a grasping Panda with box finger
  pads, where only pads collide. Controller tapes (IK, quintic references, unloaded
  inverse-dynamics feedforward) are generated once on MuJoCo 3.11 and replayed
  unchanged by every route.
- Open-loop effort: the article's MuJoCo trace is matched to 1e-10 m. All routes
  agree within 0.003 mm (the article's PhysX: 0.04 mm).
- Tracking: 0.00716° max error on every route, as in the article.
- Grasp: 7.54 cm lift and 3.19 N per finger on all routes. With an 8 mm offset,
  Genesis ends 10.9 mm sideways vs about 4 mm on the others.
- Stack: every route survives up to the 22 mm nominal offset and topples at 24 and
  26 mm. The loaded hand lags its reference by about 4 mm in −X, so the effective
  offsets are about 4 mm smaller than nominal.
- Push: every route passes the 90 and 100 mm channels and fails at 80 mm. Genesis's
  peak pad force is about 3 N vs about 10 N on the others.

### Phase 4 results (`mujoco4-vs-311.json`, `phase4.json`)
- 3.11 vs 4.0 on identical MJCF: 18 of 27 variants agree within 1 µm (primitives,
  free-space Panda, centred grasp, all stacks within 1 µm). Differences come
  from box-face contacts. The collision chain diverges after 0.001 s. Push diverges at
  0.877 s (box-face contacts change from 3 or 5 to 4) and ends 1.8–41 mm apart.
  The offset grasp differs by 0.69 mm.
- mujoco#3767: forcing a sparse vs dense Jacobian under the Newton solver differs
  by ≤ 3e-12 m on both versions in the chain, grasp, stack and push. The sparse
  Cholesky regression is not triggered by these scenes.
- New 4.0 options: speculative contacts change nothing at default margins. The
  discrete integrator lowers the slide spike from 10.46 N to 7.41 N and the drop
  peak from 52.0 N to 45.3 N. IPC (which requires the discrete integrator) gives
  results identical to discrete alone here.
- Pyramidal cone: 0.05 mm on slide, 0.005 mm on drop, 12–35 mm on the chain.
  Genesis's own defaults use a pyramidal cone and give 10.46 N on the slide, the
  same as MuJoCo. The 1.5 N Genesis slide comes from the elliptic cone.
- Determinism: MuJoCo-Warp `DeterminismType.ALL` reduces seed spread to zero
  (chain 0.85 mm → 0, push 3.66 mm → 0). Newton 1.7's
  `DeterministicMode.RUN_TO_RUN` fails to compile its own contact-conversion kernel.

### Phase 5 results (`bench/<route>.json`)
- MuJoCo 4.0 vs 3.11 CPU, contact-free Panda: 405k vs 275k steps/s on one thread
  (1.47×), 7.02M vs 4.62M with `mujoco.rollout` on 32 threads (1.52×). Push
  (Python controller, `mj_step`): 51.4k vs 45.8k. Slide and drop are unchanged.
- Peak GPU world-steps/s, contact-free Panda at 131,072 worlds: MuJoCo-Warp 4.0 51M,
  Newton 38.6M, Genesis 14.8M.
- Push (contact-heavy, controller on device): MuJoCo-Warp 5.8M and Newton 3.6M at
  8,192 worlds, Genesis 1.0M at 32,768. 131,072 worlds runs out of GPU memory on
  all three.
- Batch equivalence (the same world run with n = 1 and as one of n = 128):
  - Genesis is exact.
  - MuJoCo-Warp and Newton match to 2e-8 on drop and exactly on the Panda. On
    slide they differ by up to 8e-4 and on push by up to 4e-3.
  - Newton is also not run-to-run repeatable on push (3e-3 between identical n = 1
    runs).

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
Static site in plain ES modules (Three.js and uPlot from a CDN, no build step).
`make web` exports replays and summaries to `web/data/` (gitignored), and
`make serve` serves it:
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
