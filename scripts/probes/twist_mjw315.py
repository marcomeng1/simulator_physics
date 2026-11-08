"""Pinched square puck on plain MuJoCo-Warp (no Newton), with and without geom gap.

uv run --locked --project envs/newton python scripts/probes/twist_mjw315.py
"""

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from simbench.scenes import patch

scene = patch.twist(shape="square")
for gap in (0.0, 0.1):
    m = mujoco.MjModel.from_xml_path(str(scene.mjcf_path()))
    m.geom_gap[:] = gap
    d = mujoco.MjData(m)
    mm, dd = mjw.put_model(m), mjw.put_data(m, d, nconmax=64, njmax=256)
    f = np.zeros(m.nv, np.float32)
    f[m.jnt_dofadr[m.joint("pad_l_slide").id]] = 10
    f[m.jnt_dofadr[m.joint("pad_r_slide").id]] = -10
    dd.qfrc_applied.assign(f[None])
    for _ in range(150):
        mjw.step(mm, dd)
    wp.synchronize()
    print(mjw.__version__ if hasattr(mjw, "__version__") else "", "gap", gap,
          "pads", dd.qpos.numpy()[0][1:], "nacon", dd.nacon.numpy())
