# simulator_physics

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
make fetch        # pinned Menagerie Panda + article reference files (hash-checked)
make env-report   # versions, hardware and a smoke step per route -> results/summary/env/
make audit        # post-import model readback per route + comparison -> results/summary/audit/
make phase1 phase2 phase3   # scenes on every route -> results/raw/, results/summary/phaseN.json
make versions     # MuJoCo 3.11 vs 4.0 on every variant -> results/summary/mujoco4-vs-311.json
make phase4       # option, Jacobian and determinism variants -> results/summary/phase4.json
make bench        # batched runtime per route -> results/summary/bench/
make web serve    # export web/data/ and open the site at http://localhost:8000
make lint test
```

`make phase3` first generates the Panda controller tapes on MuJoCo 3.11 (`make tapes`).
Raw traces and `web/data/` are gitignored; summaries under `results/summary/` are
committed.

## Website

`web/` is a static site with no build step:
- **Overview:** headline findings.
- **Replay:** 3D replay of every scene with engines overlaid or side by side, contact
  force arrows, COM trails and a time-linked plot.
- **MuJoCo 4.0 vs 3.11:** the version comparison.
- **Sensitivity:** option and determinism variants.
- **Runtime:** throughput against batch size, CPU speedups and GPU memory.
- **Audit:** import readback and environment versions.

It needs the results above, then `make web serve`.

Any command can be run in a single route, e.g.
`uv run --locked --project envs/mujoco4 simbench audit --route mujoco4 --model panda_article`.
