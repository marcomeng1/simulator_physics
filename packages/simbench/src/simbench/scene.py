"""Engine-neutral scene description: one MJCF document plus initial state and inputs.

Every route loads the same MJCF text (MuJoCo, MuJoCo-Warp and Genesis through the
MuJoCo compiler, Newton through its MJCF importer). Initial velocities and joint
efforts are applied through each engine's native state and effort APIs.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simbench.paths import ASSETS

SCENE_DIR = ASSETS / "generated" / "scenes"

# Contact settings shared by every primitive scene, as in the article's model_xml.
SOLREF = "0.01 1"
SOLIMP = "0.99 0.999 0.001 0.5 2"
OPTION = ('integrator="Euler" solver="Newton" iterations="100" tolerance="1e-10" '
          'cone="elliptic"')


@dataclass
class FreeInit:
    """Initial COM velocity of a free body, world frame."""
    linear: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Scene:
    name: str
    mjcf: str
    dt: float
    duration: float
    bodies: list[str]
    joints: list[str] = field(default_factory=list)
    free_init: dict[str, FreeInit] = field(default_factory=dict)
    joint_init: dict[str, float] = field(default_factory=dict)
    # Open-loop generalized forces on `joints`, as a function of time.
    effort: Callable[[float], np.ndarray] | None = None
    # Closed-loop controller: (t, q, qd) -> generalized forces on `joints`; controllers
    # with `uses_state = True` also receive the route's full state dict.
    controller: Callable[..., np.ndarray] | None = None
    params: dict = field(default_factory=dict)
    # Record every `record_dt` (a multiple of dt) instead of every step.
    record_dt: float | None = None

    @property
    def steps(self) -> int:
        return int(round(self.duration / self.dt))

    @property
    def stride(self) -> int:
        return max(1, int(round((self.record_dt or self.dt) / self.dt)))

    @property
    def sha256(self) -> str:
        payload = json.dumps({"mjcf": self.mjcf, "params": self.params}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def mjcf_path(self) -> Path:
        """Write the MJCF to a content-addressed file (Genesis and Newton load from disk)."""
        SCENE_DIR.mkdir(parents=True, exist_ok=True)
        path = SCENE_DIR / f"{self.name}-{self.sha256[:12]}.xml"
        if not path.exists():
            path.write_text(self.mjcf)
        return path

    def effort_at(self, t: float, q: np.ndarray, qd: np.ndarray,
                  state: dict | None = None) -> np.ndarray:
        tau = np.zeros(len(self.joints))
        if self.effort is not None:
            tau += self.effort(t)
        if self.controller is not None:
            if getattr(self.controller, "uses_state", False):
                tau += self.controller(t, q, qd, state)
            else:
                tau += self.controller(t, q, qd)
        return tau


def fmt(values) -> str:
    return " ".join(repr(float(v)) if not float(v).is_integer() else str(float(v))
                    for v in np.atleast_1d(values))
