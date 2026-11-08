"""Provenance recorded alongside every run: packages, interpreter, git state, hardware."""

import contextlib
import importlib.metadata
import os
import platform
import subprocess
import sys

from simbench.paths import REPO

TRACKED_PACKAGES = [
    "simbench", "numpy", "mujoco", "mujoco-core", "mujoco-warp", "warp-lang",
    "newton", "genesis-world", "quadrants", "torch",
]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return versions


def gpu_info() -> list[dict[str, str]]:
    out = _run([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ])
    if not out:
        return []
    keys = ("name", "driver", "memory")
    return [dict(zip(keys, (v.strip() for v in line.split(",")), strict=True))
            for line in out.splitlines()]


def environment(route: str) -> dict:
    return {
        "route": route,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpus": gpu_info(),
        "git_commit": _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "-C", str(REPO), "status", "--porcelain"])),
        "packages": package_versions(),
    }
