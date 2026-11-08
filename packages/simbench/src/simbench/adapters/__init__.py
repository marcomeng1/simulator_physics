"""Engine adapters. Each module imports its engine lazily so routes stay independent."""

import importlib

ROUTES = {
    "mujoco311": "simbench.adapters.mujoco_cpu",
    "mujoco4": "simbench.adapters.mujoco_cpu",
    "mjwarp4": "simbench.adapters.mujoco_warp",
    "newton": "simbench.adapters.newton_mjwarp",
    "genesis": "simbench.adapters.genesis_rigid",
}


def load(route: str):
    return importlib.import_module(ROUTES[route])
