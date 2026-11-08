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
make lint test
```

Any command can be run in a single route, e.g.
`uv run --locked --project envs/mujoco4 simbench audit --route mujoco4 --model panda_article`.
