from __future__ import annotations

from pathlib import Path

# Project root = repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARY_DIR = RESULTS_DIR / "summary"

CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

TMP_DIR = DATA_DIR / "_downloads"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        EXPERIMENTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        SUMMARY_DIR,
        CACHE_DIR,
        CHECKPOINTS_DIR,
        NOTEBOOKS_DIR,
        SCRIPTS_DIR,
        SRC_DIR,
        TMP_DIR,
    ]:
        ensure_dir(path)