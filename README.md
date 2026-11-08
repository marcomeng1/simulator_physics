# Simulator physics

Reproduction of Manda Robotics' [Comparing physics engines for robotics simulation](https://mandarobotics.com/blog/comparing-physics-engines/index.html),
with MuJoCo 4.0 added next to the article's MuJoCo 3.11 baseline. See [PLAN.md](PLAN.md)
for scope, phases and status.

## Routes

Each route is an isolated, locked uv project under `envs/`:

| Route | Engine | Device / precision |
| --- | --- | --- |
| `mujoco311` | MuJoCo 3.11.0 | CPU / float64 |
| `mujoco4` | MuJoCo 4.0.0 | CPU / float64 |
| `mjwarp4` | MuJoCo-Warp 4.0.0, Warp 1.19.0 | GPU / float32 |
| `newton` | Newton 1.7.0 (pins MuJoCo-Warp 3.15.0) | GPU / float32 |
| `genesis` | Genesis 1.4.5, torch 2.14.1+cu130 | GPU / float64 |

The shared, engine-agnostic code lives in `packages/simbench` and depends only on numpy.

## Usage

```bash
make sync         # create every environment from its lock file
make fetch        # pinned Menagerie Panda, Unitree G1 + policy, article reference files (hash-checked)
make env-report   # versions, hardware and a smoke step per route -> results/summary/env/
make audit        # post-import model readback per route + comparison -> results/summary/audit/
make phase1 phase2 phase3   # scenes on every route -> results/raw/, results/summary/phaseN.json
make versions     # MuJoCo 3.11 vs 4.0 on every variant -> results/summary/mujoco4-vs-311.json
make phase4       # option, Jacobian and determinism variants -> results/summary/phase4.json
make phase6       # crowded sweep, peg insertion, flex ball -> results/summary/phase6.json
make tuning       # push-controller grid per route -> results/summary/tuning/<route>.json
make plastic      # 4.0 elasto-plasticity: hinge springyield, flex yield -> results/summary/plastic.json
make mj4new       # 4.0 features: thin card (speculative), RK4 tumble, plastic sheet -> results/summary/mj4new.json
make patch        # pinch-and-twist slip torque, point vs Newton hydroelastic contact -> results/summary/patch.json
make terrain      # Unitree G1 walking policy on flat/glass/sand/mud/sponge -> results/summary/terrain.json
make bench        # batched runtime per route -> results/summary/bench/
make web serve    # export web/data/ and open the site at http://localhost:8000
make lint test
```

`make phase3`, `phase6` and `tuning` first generate the Panda controller tapes on
MuJoCo 3.11 (`make tapes`). `make phase6` needs the article's bundles (`make fetch`),
since the sweep layout, peg, bore and ball meshes are rebuilt from them.
Raw traces and `web/data/` are gitignored; summaries under `results/summary/` are
committed.

## Website

`web/` is a static site with no build step:
- **Overview:** nine conclusions in large type that fold open to their evidence, a small
  chart, a link to the test and a replay; training advice per task family; further
  findings beyond the article. No engine is treated as the reference.
- **Background:** a short primer on choosing a simulator (summarising Manda Robotics'
  primer), the five setups tested, references and licences.
- **Scenarios:** every scene ordered from contact-free to contact-rich (outcome per
  engine), then grouped by task family: contact-rich manipulation (sweep,
  insertion, tuning, pinch-and-twist with hydroelastic contact), deformable and
  elasto-plastic bodies, and legged locomotion (G1 on five terrains).
- **MuJoCo 4.0:** 3.11 vs 4.0 on identical MJCF, the thin-card catch with speculative
  contacts, the RK4 tumble convergence, feature support per engine, and the option /
  determinism variants.
- **Replay:** 3D replay of every scene with engines overlaid or side by side, contact
  force arrows and points, an opacity slider and X-ray toggle, COM trails, a
  time-linked plot, and a "Studio look" (image-based lighting, filmic tone mapping,
  gradient sky, checker floor) built from Three.js add-ons only; `?studio=0` turns it off.

Every section ends with a one-line "In short" summary.
- **Runtime:** throughput against batch size, CPU speedups and GPU memory.
- **Audit:** import readback and environment versions.

It needs the results above, then `make web serve`.

`scripts/probes/` holds the one-off scripts used to build and check the Phase 6,
plasticity, MuJoCo 4.0 feature, patch-contact and terrain scenes (sweep layout, peg
pose, flex ball, MuJoCo-Warp yield diagnostics, card tunnelling, shell yield, RK4 order,
twist contact models, soft-contact sinkage). They are kept for reference and excluded from ruff.

Any command can be run in a single route, e.g.
`uv run --locked --project envs/mujoco4 simbench audit --route mujoco4 --model panda_article`.

### Publishing to GitHub Pages

The data comes from the GPU runs, so CI only deploys it and never regenerates it:
1. `make publish-data` writes a public export, packs it as `dist/site-data.tar.gz` and
   uploads it to the `site-data` release. The export leaves out the article's own
   replay traces, which are © Manda Robotics. It then triggers the Pages workflow.
2. `.github/workflows/pages.yml` runs on pushes to `main` that touch `web/`, and on
   demand. It downloads the release asset into `web/data/` and deploys `web/` with
   `actions/deploy-pages`.

One-time setup: in Settings → Pages, set Source to "GitHub Actions". `make web`
restores the full local data, including the article traces.
