from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ASSETS = REPO / "assets"
MENAGERIE = ASSETS / "menagerie"
PANDA_DIR = MENAGERIE / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"
UNITREE = ASSETS / "unitree"
REFERENCE = REPO / "reference"
RESULTS = REPO / "results"
SUMMARY = RESULTS / "summary"
RAW = RESULTS / "raw"
