import numpy as np
import pytest
from simbench.paths import REFERENCE
from simbench.scenes import deform, insertion
from simbench.tuning import score

ARTICLE = REFERENCE / "article" / "assets"


def test_bore_wedges_match_clearance():
    for clearance in (0.0001, 0.00025, 0.0005):
        for piece in insertion.bore_pieces(clearance):
            r = piece[:, :2] @ (piece[:, :2].mean(axis=0) / np.linalg.norm(
                piece[:, :2].mean(axis=0)))
            assert r.min() == pytest.approx(insertion.PEG_R + clearance)
            assert piece[:, 2].min() == pytest.approx(-insertion.BORE_DEPTH)
    assert len(insertion.bore_pieces(0.0001)) == insertion.FACETS


@pytest.mark.skipif(not (ARTICLE / "insertion").exists(), reason="run `make fetch` first")
def test_bore_and_peg_match_article_geometry():
    from simbench.reference import BUNDLES, load_bundle

    trace = load_bundle(BUNDLES["insertion"])["traces"][1]
    shapes = {b["id"]: b for b in trace["scene"]["bodies"]}
    for k in (0, 17, 63):
        ours = insertion.bore_pieces(0.0001)[k]
        theirs = np.array(shapes[f"bore_{k:03d}"]["vertices"])
        assert np.sort(ours, axis=0) == pytest.approx(np.sort(theirs, axis=0), abs=1e-7)
    peg = np.array(shapes["paddle"]["vertices"])
    assert np.sort(insertion.peg_vertices(), axis=0) == pytest.approx(np.sort(peg, axis=0),
                                                                      abs=1e-9)


def test_tilted_tip_starts_on_bore_axis():
    tip = insertion.peg_tip_hand("tilted")
    hand_rot = np.diag([1.0, -1.0, -1.0])
    world_offset = hand_rot @ tip
    assert np.hypot(*world_offset[:2]) == pytest.approx(0.03 * np.sin(np.deg2rad(1.0)))


def test_plate_squeeze_profile():
    assert deform.squeeze(0.0) == (0.0, 0.0)
    assert deform.squeeze(0.7)[0] == pytest.approx(deform.SQUEEZE)
    assert deform.squeeze(0.35)[0] == pytest.approx(deform.SQUEEZE / 2)
    assert deform.squeeze(1.5)[0] == pytest.approx(0.0)


def test_tuning_rule_prefers_success_then_effort():
    ok = {"status": "ok", "all_passed": True, "passed_count": 3, "min_progress_capped": 1.0}
    cheap = [ok | {"mean_norm_sq_effort": 0.030}] * 3
    costly = [ok | {"mean_norm_sq_effort": 0.031}] * 3
    failing = [ok | {"mean_norm_sq_effort": 0.01, "all_passed": False, "passed_count": 2}] * 3
    assert score(cheap) > score(costly) > score(failing)
    assert score([*cheap[:2], {"status": "error"}]) < score(failing)
