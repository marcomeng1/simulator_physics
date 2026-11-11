"""Fetch pinned external inputs: the Menagerie Panda, Cartwheel G1, and article files.

Both Menagerie and Cartwheel are recorded in committed lock files with SHA-256 hashes,
so a fresh checkout reproduces byte-identical inputs and detects upstream changes.
"""

import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from simbench.paths import ASSETS, CARTWHEEL, CARTWHEEL_G1, MENAGERIE, REFERENCE

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie"
# The revision named in the article's Panda audit. Later revisions up to 98a70c5
# (2026-11-02) only move the light from panda.xml to scene.xml.
MENAGERIE_COMMIT = "822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7"
MENAGERIE_LOCK = ASSETS / "menagerie.lock.json"
MENAGERIE_MODELS = ["franka_emika_panda"]

CARTWHEEL_REPO = "andrew-cartwheel/cartwheel-control"
CARTWHEEL_REVISION = "24fc279af8f70097f503767b81db5384d779aae5"
CARTWHEEL_LOCK = ASSETS / "cartwheel.lock.json"
CARTWHEEL_PATTERNS = [
    "LICENSE", "NOTICE",
    "g1/config.json", "g1/policy.safetensors",
    "g1/motions/basketball_shot.npz",
    "g1/model/**",
]

ARTICLE_BASE = "https://mandarobotics.com/blog/comparing-physics-engines/"
REFERENCE_LOCK = REFERENCE / "manifest.json"
REFERENCE_FILES = [
    "assets/sliding-block/validated-controls.json",
    "assets/drop-settle/audit.md",
    "assets/drop-settle/metrics.json",
    "assets/hinge/audit.md",
    "assets/hinge/metrics.json",
    "assets/collision-chain/audit.md",
    "assets/collision-chain/metrics.json",
    "assets/collision-chain/seed-metrics.json",
    "assets/panda/audit.md",
    "assets/panda/metrics.json",
    "assets/panda-tracking/audit.md",
    "assets/panda-tracking/metrics.json",
    "assets/grasp/audit.md",
    "assets/grasp/metrics.json",
    "assets/stack/audit.md",
    "assets/stack/metrics.json",
    "assets/push/audit.md",
    "assets/push/metrics.json",
    "assets/sweep-comparison/PROTOCOL.md",
    "assets/sweep-comparison/plan.json",
    "assets/sweep-comparison/summary.json",
    "assets/sweep-comparison/verification.json",
    "assets/insertion/summary.json",
    "assets/insertion/guided-summary.json",
    "assets/insertion/guided-final-summary.json",
    "assets/deformable/REPRODUCING.md",
    "assets/deformable/RESOLUTION.md",
    "assets/deformable/FEEDBACK.md",
    "assets/deformable/REVISION-2026-09-30.md",
    "assets/deformable/summary.json",
    "assets/deformable/threshold-summary.json",
    "assets/deformable/resolution-summary.json",
    "assets/controller-transfer/plan.json",
    "assets/controller-transfer/TRANSFER.md",
    "assets/controller-transfer/TRACE-ANALYSIS.md",
    "assets/controller-speed/REPORT.md",
    "assets/runtime/runtime-summary.md",
    "assets/runtime/RUNTIME-PROTOCOL-2026-09-25.md",
    "assets/runtime/RUNTIME-EQUIVALENCE-2026-09-25.md",
    "replay/sliding-block-20260921.json.gz",
    "replay/drop-20260921.json.gz",
    "replay/hinge-20260921.json.gz",
    "replay/collision-chain-20260921.json.gz",
    "replay/panda-20260921.json.gz",
    # Phase 3 replays: the article's grasp, stack and push layouts.
    "replay/panda-tracking-20260921.json.gz",
    "replay/grasp-center-20260921.json.gz",
    "replay/grasp-offset-20260921.json.gz",
    "replay/stack-center-20260921.json.gz",
    "replay/push-medium-20260921.json.gz",
    # Phase 6 replays: sweep geometry and outcomes, insertion, deformable ball.
    "assets/sweep-comparison/repeat-1.json.gz",
    "assets/sweep-comparison/repeat-2.json.gz",
    "assets/insertion/replay-c100-tilted-r0.json.gz",
    "assets/insertion/replay-guided-tilted-contact_guided.json.gz",
    "assets/deformable/compression.json.gz",
    "assets/deformable/grasp-1N.json.gz",
    "assets/deformable/grasp-2N.json.gz",
    "assets/controller-transfer/replay-selected-test-0.json.gz",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch_menagerie(update: bool = False) -> None:
    """Sparse-checkout the pinned Menagerie commit into assets/menagerie/.

    With update=True, rewrites the lock (e.g. after changing MENAGERIE_COMMIT).
    """
    lock = json.loads(MENAGERIE_LOCK.read_text()) if MENAGERIE_LOCK.exists() else None
    if lock is not None and lock["commit"] != MENAGERIE_COMMIT:
        update = True
    commit = MENAGERIE_COMMIT

    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "menagerie"
        _git("clone", "--filter=blob:none", "--no-checkout", "--quiet", MENAGERIE_REPO, str(clone))
        _git("sparse-checkout", "set", *MENAGERIE_MODELS, cwd=clone)
        _git("checkout", "--quiet", commit, cwd=clone)
        commit_date = _git("show", "-s", "--format=%cI", commit, cwd=clone)
        if MENAGERIE.exists():
            shutil.rmtree(MENAGERIE)
        MENAGERIE.mkdir(parents=True)
        for name in MENAGERIE_MODELS:
            shutil.copytree(clone / name, MENAGERIE / name)

    files = {
        str(p.relative_to(MENAGERIE)): _sha256(p)
        for p in sorted(MENAGERIE.rglob("*"))
        if p.is_file()
    }
    if lock is not None and not update and files != lock["files"]:
        raise RuntimeError("Menagerie checkout does not match assets/menagerie.lock.json")
    MENAGERIE_LOCK.write_text(
        json.dumps(
            {"repo": MENAGERIE_REPO, "commit": commit, "commit_date": commit_date,
             "models": MENAGERIE_MODELS, "files": files},
            indent=2,
        )
        + "\n"
    )
    print(f"menagerie {commit[:12]} ({commit_date}): {len(files)} files")


def fetch_cartwheel(update: bool = False) -> None:
    """Download the pinned Cartwheel G1 policy, MJCF and basketball_shot motion."""
    from huggingface_hub import snapshot_download

    lock = json.loads(CARTWHEEL_LOCK.read_text()) if CARTWHEEL_LOCK.exists() else None
    if lock is not None and lock["revision"] != CARTWHEEL_REVISION:
        update = True

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = snapshot_download(
            CARTWHEEL_REPO, revision=CARTWHEEL_REVISION,
            allow_patterns=CARTWHEEL_PATTERNS, local_dir=tmp,
        )
        src = Path(downloaded)
        if CARTWHEEL.exists():
            shutil.rmtree(CARTWHEEL)
        CARTWHEEL.mkdir(parents=True)
        for name in ("LICENSE", "NOTICE"):
            shutil.copy2(src / name, CARTWHEEL / name)
        shutil.copytree(src / "g1", CARTWHEEL_G1)

    required = [CARTWHEEL_G1 / "config.json", CARTWHEEL_G1 / "policy.safetensors",
                CARTWHEEL_G1 / "motions" / "basketball_shot.npz",
                CARTWHEEL_G1 / "model" / "g1.xml", CARTWHEEL / "LICENSE", CARTWHEEL / "NOTICE"]
    missing = [str(p.relative_to(CARTWHEEL)) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"Cartwheel fetch incomplete: missing {missing}")

    files = {str(p.relative_to(CARTWHEEL)): _sha256(p)
             for p in sorted(CARTWHEEL.rglob("*")) if p.is_file()}
    if lock is not None and not update and files != lock["files"]:
        raise RuntimeError("Cartwheel checkout does not match assets/cartwheel.lock.json")
    CARTWHEEL_LOCK.write_text(json.dumps({
        "repo": CARTWHEEL_REPO, "revision": CARTWHEEL_REVISION,
        "license": "Cartwheel Open Model License (weights/motions); Apache-2.0 (code); "
                   "BSD-3-Clause (g1/model)",
        "files": files,
    }, indent=2) + "\n")
    print(f"cartwheel {CARTWHEEL_REVISION[:12]}: {len(files)} files")


def fetch_reference(update: bool = False) -> None:
    """Download the article's audit notes, metrics and replay traces into reference/."""
    lock = json.loads(REFERENCE_LOCK.read_text()) if REFERENCE_LOCK.exists() else None
    out_root = REFERENCE / "article"
    hashes: dict[str, str] = {}
    for rel in REFERENCE_FILES:
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(ARTICLE_BASE + rel, headers={"User-Agent": "simbench"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        hashes[rel] = _sha256(dest)
        if lock is not None and not update and lock["files"].get(rel) != hashes[rel]:
            print(f"warning: {rel} changed upstream since it was locked")
    REFERENCE_LOCK.write_text(
        json.dumps({"base": ARTICLE_BASE, "files": hashes}, indent=2) + "\n"
    )
    print(f"reference: {len(hashes)} files")
