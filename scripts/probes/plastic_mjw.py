"""Probe: why the plastic flex ball goes non-finite on MuJoCo-Warp 4.0.

Runs the first 0.5 s of the plastic compression (the squeeze starts at 0.1 s) for each
`yield:dt` case and prints the status. A yield of 1.0 allocates the plastic state but
never yields, separating "plastic state present" from "material actually yielding".
Usage:
    uv run --locked --project envs/mjwarp4 python scripts/probes/plastic_mjw.py \
        0.1:5e-5 0.1:2.5e-5 1.0:5e-5
"""

import sys

from simbench.run import simulate
from simbench.scenes.deform import compress

for case in sys.argv[1:] or ["0.1:5e-5", "0.1:2.5e-5", "1.0:5e-5"]:
    y, dt = (float(x) for x in case.split(":"))
    scene = compress(dt=dt, yield_strain=y)
    scene.duration = 0.5
    tr = simulate("mjwarp4", scene, tag="probe")
    print(f"yield {y:g} dt {dt:g}: {tr.meta['status']} {tr.meta['error'] or ''}", flush=True)
