"""Static sinkage of a G1-weight block on eight 5 mm spheres, per G1 terrain contact model.

Isolates how each engine maps the same friction/solref/solimp to steady penetration.
Usage: uv run --project envs/<route> python scripts/probes/soft_contact_probe.py <route>
"""

import sys

import numpy as np
from simbench.run import simulate
from simbench.scene import Scene
from simbench.scenes.g1 import TERRAINS, robot_mass


def block(terrain: str) -> Scene:
    spec = TERRAINS[terrain]
    attrs = (f'friction="{spec["friction"]} 0.005 0.0001" solref="{spec["solref"]}" '
             f'solimp="{spec["solimp"]}"')
    feet = "".join(f'<geom type="sphere" size="0.005" pos="{x} {y} -0.105" {attrs}/>'
                   for x in (-0.05, 0.12) for y in (-0.15, -0.1, 0.1, 0.15))
    mjcf = f"""<mujoco model="soft_{terrain}">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" {attrs}/>
    <body name="block" pos="0 0 0.11">
      <freejoint/>
      <geom type="box" size="0.1 0.18 0.1" mass="{robot_mass()}" contype="0" conaffinity="0"/>
      {feet}
    </body>
  </worldbody>
</mujoco>"""
    return Scene(name=f"soft-{terrain}", mjcf=mjcf, dt=0.002, duration=2.0, bodies=["block"],
                 record_dt=0.02, params={"terrain": terrain})


route = sys.argv[1]
for terrain in TERRAINS:
    tr = simulate(route, block(terrain))
    z = tr.pos[:, 0, 2]
    sink = (0.11 - 0.105 - 0.005) - (z - 0.11)
    print(f"{route:9s} {terrain:6s} status={tr.meta['status']} final sink {sink[-1] * 1e3:6.2f} mm"
          f"  peak {sink.max() * 1e3:6.2f} mm  rebound {(sink.max() - sink[-1]) * 1e3:5.2f} mm",
          flush=True)
