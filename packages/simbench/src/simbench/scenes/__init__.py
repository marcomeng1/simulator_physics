"""Scene registry: name -> factory(**overrides) -> Scene."""

from simbench.scenes import panda, primitives

SCENES = {
    "slide": primitives.slide,
    "drop": primitives.drop,
    "hinge": primitives.hinge,
    "collision-chain": primitives.collision_chain,
    "panda-effort": panda.effort,
    "panda-tracking": panda.tracking,
    "grasp": panda.grasp,
    "stack": panda.stack,
    "push": panda.push,
}


def make(name: str, **overrides):
    return SCENES[name](**overrides)
