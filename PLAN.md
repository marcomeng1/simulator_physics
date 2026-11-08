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
  scripts/probes/           # one-off scene-building and diagnostic scripts (kept for reference)
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
| 6. Sweep, insertion, deformables, tuning | Done (2026-11-08) |

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

### Phase 6 results (`phase6.json`, `tuning/<route>.json`)
The sweep, insertion and deformable scenes are rebuilt from the article's replay
bundles (object layout, peg and bore meshes, ball mesh, timings). The article's own
traces are scored with the same metric code.

- Crowded sweep (15 objects, 12.5 s tape, 3 runs):
  - MuJoCo 4.0 collects 10, the same as the article's MuJoCo in both published
    repeats. 3.11 collects 11. Forcing a dense Jacobian gives 10 on both versions.
  - GPU, per run: MuJoCo-Warp 4.0 10 / 10 / 10, Genesis 11 / 11 / 11, Newton 1.7
    6 / 5 / 7. The article's Newton 1.5.2 collected 11 and 8. Newton's peak paddle force
    is about 105 N vs 20 N on MuJoCo CPU.
  - MuJoCo-Warp `DeterminismType.ALL` does not make this scene repeatable. Its two
    runs first differ at 0.111 s (4 nm on a cylinder) and end metres apart because
    objects that leave the table fall freely. Both still collect 10.
- Peg insertion (3 clearances × aligned / 0.25 mm offset / 1° tilt × 3 repeats):
  - The reconstructed impedance controller matches the article's MuJoCo hand path to
    a few µm until the tilted peg jams at about 2.0 s.
  - MuJoCo 3.11 and 4.0 pass 24 of 27, as in the article. Only the tilted peg at
    0.1 mm fails. MuJoCo-Warp 4.0, Newton and Genesis also pass 24 of 27, failing
    the same case. Their tilted pegs stall between 13.74 and 13.77 mm.
  - The tilted 0.1 mm peg stalls at 13.7 mm here vs 15.3 mm in the article. The jam
    itself matches; the difference after it comes from controller details the
    article does not publish.
  - Contact-guided rerun (the article's force relief and tilt admittance): every
    route reaches 19.84–19.89 mm (article 19.95 mm). Peak peg force is 20–22 N on
    MuJoCo, MuJoCo-Warp and Newton and 4.8 N on Genesis (article: 14–16 N, Genesis
    3.6 N).
- Deformable ball (125 vertices, 384 tetrahedra, E = 20 kPa), MuJoCo routes only:
  - Compression hold load: 3.11 2.355 N, 4.0 2.132 N, MuJoCo-Warp 4.0 2.020 N.
    The article's pair was 2.151 N (CPU) vs 1.631 N (MuJoCo-Warp, with the ball 4 mm
    less compressed), so the CPU-GPU gap shrinks from 24% to 5% in 4.0.
  - The article's MuJoCo CPU compression (2.151 N) is closer to our 4.0 than to our
    3.11, so its flex settings differ from ours somewhere we cannot see.
  - 2 N grasp: 3.11 holds 67.5 mm of lift (article CPU 65.6 mm). On 4.0 one
    tetrahedron inverts for 6 frames at pad contact (0.59 s), so the run is invalid
    by the article's rule. Ignoring that, 4.0 holds 66.2 mm.
  - MuJoCo-Warp 4.0 holds the 2 N grasp with no invalid frame (63.8 mm lift). The
    article's MuJoCo-Warp went invalid at 2.525 s. At 1.5 N it also holds (62.8 mm).
  - 1 N grasp drops the ball on every MuJoCo route. 4.0 goes invalid at 1.204 s
    (article CPU 1.345 s), MuJoCo-Warp 4.0 at 1.946 s (article 1.335 s), and 3.11 at
    2.83 s.
  - Effort sweep: 3.11 holds from 1.1 N up and MuJoCo-Warp 4.0 from 1.5 N, while 4.0
    CPU inverts a tetrahedron at every effort from 1 to 2 N (at 1.5 N only after the
    hold).
- Controller tuning (27 candidates on our Phase 3 push layout, not the article's):
  - MuJoCo 4.0 picks p0.5-d1.4-t4 and 3.11 picks p0.5-d1-t4, each by under 0.4% effort
    over the runner-up. Newton picks p0.5-d1.4-t4 (0.12%) and Genesis p0.5-d1-t4
    (0.02%). The article picked p0.5-d0.7-t4 everywhere, also by under 0.6%.
    MuJoCo-Warp 4.0 tuning was skipped for time
    (`simbench tune --route mjwarp4`, about 1.5 h).
  - Held-out: 4 of 6 layouts for both the baseline and the pick on every route, as in
    the article.
    Both fail the two 80 mm channels.
  - On every tuned route, every candidate and every speed-control push (down to
    0.75 s) passes all three selection layouts, so the ranking comes down to effort
    alone. The top three on every route are the ×0.5-stiffness, 4 s candidates, which
    differ only in damping.
- The article's push layout (cube, blocks, table and walls) is in its
  controller-transfer bundle. Phase 3 and the tuning still use our own layout.

### Elasto-plasticity results (`plastic.json`, new in MuJoCo 4.0)
- Plastic hinge: a 2 N·m/rad joint spring with `springyield` 5° (degrees, the
  compiler's default angle unit), bent to about 9.5° by a PD under displacement
  control, then released. Ideal plasticity has no hardening, so a torque load would
  flow without bound.
  - MuJoCo 4.0 and MuJoCo-Warp 4.0 rest at 4.533° against an ideal 4.530°.
  - MuJoCo 3.11 rejects the attribute. Newton 1.7 and Genesis 1.4.5 load it silently
    and behave elastically (both rest at 0.006°).
  - Elastic control: every route peaks at 9.128°. An earlier Newton run peaked at
    8.56° because `add_mjcf` ignores `<option gravity>`; the adapter now passes it.
- Plastic ball (flex `<elasticity yield>`, von Mises radial return), 12 mm squeeze:
  - MuJoCo 4.0: yield strains 0.01 / 0.03 / 0.1 leave 10.46 / 7.85 / 2.70 mm of
    permanent set, with hold loads 0.287 / 0.885 / 1.973 N (elastic 2.132 N, set
    0.01 mm).
  - MuJoCo-Warp 4.0 goes non-finite once the flex yields (0.25 / 0.37 / 0.34 s), also
    at half the timestep. A yield that is never reached runs normally.
  - 2 N grasp on 4.0: at 0.1 the ball is lifted 64.4 mm and keeps a 6.78 mm dent
    (elastic about 0.8–1.0 mm). At 0.03 the yield stress (600 Pa) is about the ball's
    weight pressure, so it slumps on the table and goes invalid at 0.082 s.
    MuJoCo-Warp goes non-finite at 0.044–0.046 s.
- Legacy elastic constants: before 4.0 the 3D SVK flex used 2λ. Running 4.0 with
  ν′ = 2ν/(1+2ν) and E′ = E(1+ν′)/(1+ν) gives a 2.327 N hold load against 2.355 N on
  3.11 and 2.132 N as authored, about 88% of the version gap. MuJoCo-Warp legacy gives
  2.188 N.

### Other MuJoCo 4.0 features (`mj4new.json`)
- Thin card catch (speculative contacts): an 88 × 63 × 0.3 mm, 1.8 g card released
  5 mm above a 1 mm static plate.
  - Without speculative contacts it falls through on every route, at 0.3, 1, 3 and
    10 m/s at 1 ms, and at 1 m/s at 2 ms. Box–box contact is only found at the step
    poses. Once penetration passes half the combined thickness (0.65 mm), the normal
    flips and the card is pushed out of the far side.
  - MuJoCo 4.0 with `enable="SPECULATIVE"` catches it at every speed (rest gap
    −0.0025 mm, peak penetration 0.05–0.40 mm). A card spinning at 30 rad/s still
    goes through at 10 ms.
  - MuJoCo-Warp 4.0 does not implement the flag. Newton and Genesis have no equivalent.
- Tumbling T-handle (fourth-order RK4 for free bodies): a T-handle with no gravity and
  inertia (1, 4, 4.8)e-4 kg·m², spun at 6 rad/s about its intermediate axis for 4 s.
  It is measured against an independent RK4 solve at 0.1 ms.
  - RK4 max orientation error at 10 / 5 / 2 / 1 ms: MuJoCo 3.11 gives 0.027 / 0.0067
    / 0.0011 / 0.00027° (second order). MuJoCo 4.0 stays below 2.5e-5° (fourth order,
    at the reference's resolution from 5 ms).
  - Euler / implicitfast on MuJoCo: 7.6 / 7.5° at 2 ms and 3.8° at 1 ms, identical on
    3.11 and 4.0.
  - MuJoCo-Warp 4.0 RK4: 0.06 / 0.22 / 0.21 / 0.62° at 10 / 5 / 2 / 1 ms. It does not
    converge, consistent with float32 round-off growing with step count. Its Euler
    matches the CPU.
  - Newton 1.7 (MuJoCo-Warp 3.15): 1.4 / 1.2 / 6.7 / 13° for RK4, and 9–12° for Euler
    and implicitfast. The cause is not isolated (the effective inertia matches).
  - Genesis has no RK4 (the adapter raises). Its Euler matches MuJoCo, but its
    implicitfast is accurate: 0.0026° at 2 ms against 7.5° on MuJoCo.
  - The Newton adapter now passes the scene's integrator instead of forcing Euler.
- Plastic sheet (2D flex shell, `yield="0 εb"`): a PET-like 120 × 20 × 1 mm
  cantilever (E 2 GPa, ν 0.38), two clamped vertex columns, pressed 15 mm by a sphere
  at 70 mm and released. It needs `integrator="discrete"`.
  - MuJoCo 4.0: εb 0.004 / 0.003 / 0.002 leave 0 / 0.30 / 8.55 mm tip set, with the
    tip held at 22.5–23.9 mm. The press puts about 0.0046 outer-fibre strain at the
    clamp.
  - MuJoCo-Warp 4.0 runs it (220 s per run) but holds the tip at 16.1–16.2 mm. It keeps
    0.67 mm at εb 0.002 and none at 0.003.
  - MuJoCo 3.11 has neither the discrete integrator nor yield.

### Patch contact results (`patch.json`)
- Pinch and twist: a 0.1 kg puck (20 mm-radius disc or 30 mm square, 20 mm thick)
  squeezed by two pads with 10 N each (μ 0.5), then twisted by a 0.2 N·m/s torque ramp.
  Uniform pressure gives a slip torque of 0.133 N·m (disc) and 0.115 N·m (square);
  slip is when the puck has turned 1°.
- Point contacts over-resist twist: slip at 1.52× (disc) and 1.87× (square) on MuJoCo
  3.11, 4.0, MuJoCo-Warp 4.0 and Genesis. Box–cylinder and box–box give four contacts
  per face at the rim or corners, the largest lever arm. Torsional friction (condim 6,
  torsion matched to the patch) adds on top: 1.83× / 2.21×. multiccd changes nothing.
- Newton 1.7 as shipped slips at 0.05× / 0.14×, with 4–10 mm pad penetration. Its
  SolverMuJoCo writes `body_invweight0` [22222, 22222] for the hinged puck where MuJoCo
  compiles [0, 22222]. That inflates the constraint regulariser and softens the
  contacts about 1,000×. It is not the 0.1 geom gap, which was ruled out on MuJoCo 3.15
  CPU and standalone MuJoCo-Warp 3.15. The adapter's `mujoco_invweight=True` copies
  MuJoCo's values. With it, MuJoCo-Warp contacts give 0.89× / 1.87× and Newton's own
  point pipeline 1.54× / 1.89×.
- Newton hydroelastic contact (SDF, kh 1e11, reduced patch of about 180 contacts):
  0.64× / 0.57× as shipped, 1.36× / 1.22× with the invweight fix. The full patch (kh
  1e10, about 8,500 contacts per step, 26 s per run) gives 1.41× / 1.16× and hits the
  solver's iteration limit; it needs naconmax 12,000 and njmax 40,000.

### Legged locomotion results (`terrain.json`)
- Unitree G1 12-DOF from `unitree_rl_gym` (`276801e46c5d`, BSD-3) with its pretrained
  LSTM walking policy (`motion.pt`), read with numpy (matches torch to 7e-6). It runs
  as in Unitree's MuJoCo deploy script: 50 Hz policy, 500 Hz PD, torques clipped to the
  actuator ranges, a 0.8 s gait clock and a constant command for 10 s. Menagerie's G1 is
  the 29-DOF robot, which this policy was not trained on.
- Terrains are contact-parameter models applied to every colliding geom and the ground.
  flat: MuJoCo defaults. glass: μ 0.1. sand: μ 0.6, solref 0.05 1.2, solimp 0.3 0.95
  0.02, with 1.5 cm ripples. mud: μ 1.2, solref 0.05 2, solimp 0.6 0.95 0.03. sponge:
  μ 0.9, solref 0.05 0.15, solimp 0.3 0.95 0.02, with bumps up to 2.5 cm.
  Commands are 0.3, 0.6 and 0.9 m/s.
- Flat: all five engines walk 2.8 / 5.5 / 8.0 m (0.93–0.97× the command). MuJoCo 3.11,
  4.0 and MuJoCo-Warp agree to 0.02 mm, Genesis to 3 mm, Newton to 15 cm.
- Glass topples the policy at 0.6 and 0.9 m/s on every engine (1.1–1.8 s), except
  Newton at 0.6 m/s, which slides on until 8.9 s. At 0.3 m/s the MuJoCo engines keep
  walking, slipping 0.4 m/s at the feet, while Newton and Genesis fall.
- Sand, mud and sponge: the MuJoCo engines and Newton walk every case (sink 14 / 33–45
  / 7.5 mm on MuJoCo). Genesis falls in all 9 runs within 1–4.5 s.
- Cause, from `scripts/probes/soft_contact_probe.py`: a 35 kg block on eight 5 mm
  spheres with the same parameters settles 4.8 / 8.1 / 6.4 mm into sand on MuJoCo 4.0
  / Newton / Genesis, and 12.0 / 9.2 mm into mud on MuJoCo 4.0 / Newton. Genesis lets
  it fall through the floor on mud. Genesis reads the same friction, solref and solimp
  per geom (checked after build), so it maps soft-contact impedance differently.

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

### Phase 6: contact-rich tasks
- Crowded sweep: 15 objects on the article's layout and 12.5 s tape, 3 runs per
  route, dense vs sparse Jacobian (most exposed to the sparse Cholesky bug), and
  MuJoCo-Warp determinism.
- Peg insertion: 64-facet peg and chamfered bore, Cartesian impedance controller
  (20 N / 2 N·m caps, stop at 40 N / 4 N·m), 0.1 / 0.25 / 0.5 mm clearance ×
  aligned / offset / tilted × 3 repeats, plus the contact-guided rerun.
- Deformable ball (4.0 flex contact and elasticity fixes): plate compression and the
  Panda grasp at 1–2 N finger effort, on MuJoCo 3.11, 4.0 and MuJoCo-Warp 4.0.
- Controller tuning: 27-candidate grid on the push, three selection and six
  held-out layouts, and a speed-only control.

## Interactive website
Static site in plain ES modules (Three.js and uPlot from a CDN, no build step).
`make web` exports replays and summaries to `web/data/` (gitignored), and
`make serve` serves it. Tabs: Overview (foldable conclusions with mini charts, test
and replay links, training advice per task, findings beyond the article), Background
(simulator primer summary, references, licences), Scenarios (contact-free to
contact-rich outcome ladder, then manipulation, deformable and plastic, and legged
locomotion), MuJoCo 4.0 (with feature support), Replay, Runtime and Audit. Every
section ends with a one-line "In short" summary.
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
