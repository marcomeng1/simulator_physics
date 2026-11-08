"""Common trace format written by every route.

Frame k holds the state at t_k = k * dt *before* step k, the generalized effort
applied during step k, and the native contact forces the solver computed for
step k (acting over [t_k, t_k + dt)), as in the article's recordings.

Arrays (npz):
  t (N,)                       time [s]
  pos, com (N, B, 3)           body-frame origin and COM, world [m]
  quat (N, B, 4)               body orientation, world, (w, x, y, z)
  linvel, angvel (N, B, 3)     COM linear and angular velocity, world [m/s, rad/s]
  q, qd, tau (N, J)            tracked joints [rad or m, /s, N m or N]
  contact_net (N, B, 3)        net native contact force on each tracked body, world [N]
  c_frame, c_a, c_b (K,)       per-contact frame index and body indices (into contact_names)
  c_pos, c_force (K, 3)        contact point and force acting on body b (on a: -force)
  c_dist (K,)                  signed contact distance [m] (negative = penetration)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARRAYS = ["t", "pos", "com", "quat", "linvel", "angvel", "q", "qd", "tau", "contact_net",
          "c_frame", "c_a", "c_b", "c_pos", "c_force", "c_dist"]


@dataclass
class Trace:
    meta: dict
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self.__dict__["arrays"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def bodies(self) -> list[str]:
        return self.meta["bodies"]

    def body(self, name: str) -> int:
        return self.meta["bodies"].index(name)

    def save(self, stem: Path) -> None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stem.with_suffix(".npz"), **self.arrays)
        stem.with_suffix(".json").write_text(json.dumps(self.meta, indent=2) + "\n")

    @classmethod
    def load(cls, stem: Path) -> "Trace":
        stem = Path(stem)
        with np.load(stem.with_suffix(".npz")) as data:
            arrays = {k: data[k] for k in data.files}
        return cls(json.loads(stem.with_suffix(".json").read_text()), arrays)


class TraceBuilder:
    """Preallocated per-frame recording; contacts are appended as they occur."""

    def __init__(self, n: int, n_bodies: int, n_joints: int):
        self.n = n
        z = np.zeros
        self.a = {
            "t": z(n), "pos": z((n, n_bodies, 3)), "com": z((n, n_bodies, 3)),
            "quat": z((n, n_bodies, 4)), "linvel": z((n, n_bodies, 3)),
            "angvel": z((n, n_bodies, 3)), "q": z((n, n_joints)), "qd": z((n, n_joints)),
            "tau": z((n, n_joints)), "contact_net": z((n, n_bodies, 3)),
        }
        self.contacts: list[tuple] = []
        self.frames = 0

    def record(self, k: int, t: float, state: dict, tau: np.ndarray, contacts: dict) -> None:
        a = self.a
        a["t"][k] = t
        for key in ("pos", "com", "quat", "linvel", "angvel", "q", "qd"):
            a[key][k] = state[key]
        a["tau"][k] = tau
        a["contact_net"][k] = contacts["net"]
        if len(contacts["a"]):
            self.contacts.append((np.full(len(contacts["a"]), k), contacts["a"], contacts["b"],
                                  contacts["pos"], contacts["force"], contacts["dist"]))
        self.frames = k + 1

    def build(self, meta: dict) -> Trace:
        arrays = {k: v[: self.frames] for k, v in self.a.items()}
        if self.contacts:
            cols = list(zip(*self.contacts, strict=True))
            arrays["c_frame"] = np.concatenate(cols[0]).astype(np.int32)
            arrays["c_a"] = np.concatenate(cols[1]).astype(np.int16)
            arrays["c_b"] = np.concatenate(cols[2]).astype(np.int16)
            arrays["c_pos"] = np.concatenate(cols[3])
            arrays["c_force"] = np.concatenate(cols[4])
            arrays["c_dist"] = np.concatenate(cols[5])
        else:
            arrays.update(c_frame=np.zeros(0, np.int32), c_a=np.zeros(0, np.int16),
                          c_b=np.zeros(0, np.int16), c_pos=np.zeros((0, 3)),
                          c_force=np.zeros((0, 3)), c_dist=np.zeros(0))
        return Trace(meta, arrays)
