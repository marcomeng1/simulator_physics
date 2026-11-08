import numpy as np
import pytest
from simbench.scenes.panda import CTRL_DT, TapeController
from simbench.tapes import piecewise, quintic, spec_sha


def test_quintic_is_rest_to_rest():
    s, ds, dds = quintic(np.array([0.0, 0.5, 1.0]))
    assert s == pytest.approx([0.0, 0.5, 1.0])
    assert ds[[0, 2]] == pytest.approx([0.0, 0.0])
    assert dds[[0, 2]] == pytest.approx([0.0, 0.0])
    assert ds[1] == pytest.approx(1.875)


def test_piecewise_holds_between_moves():
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    v = piecewise(t, [0.04, 0.04], [[0.5, 1.5, [0.012, 0.012]], [3.0, 4.0, [0.04, 0.04]]])
    assert v[:2] == pytest.approx(np.full((2, 2), 0.04))
    assert v[3:6] == pytest.approx(np.full((3, 2), 0.012))
    assert v[-1] == pytest.approx([0.04, 0.04])


def test_spec_sha_ignores_key_order():
    assert spec_sha({"a": 1, "b": [1, 2]}) == spec_sha({"b": [1, 2], "a": 1})


def test_controller_holds_command_between_samples():
    n = 5
    tape = {"qref": np.ones((n, 9)), "qdref": np.zeros((n, 9)), "ff": np.zeros((n, 9))}
    ctl = TapeController(tape, 9)
    q, qd = np.zeros(9), np.zeros(9)
    u0 = ctl(0.0, q, qd)
    assert u0 == pytest.approx(np.minimum([80, 80, 60, 60, 20, 15, 8, 400, 400],
                                          [87, 87, 87, 87, 12, 12, 12, 4, 4]))
    held = ctl(CTRL_DT / 2, q + 1, qd)  # 0.5 ms step: command is held, not recomputed
    assert held == pytest.approx(u0)
    assert ctl(CTRL_DT, q + 1, qd) == pytest.approx(np.zeros(9))
