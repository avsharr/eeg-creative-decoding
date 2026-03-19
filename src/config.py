from __future__ import annotations

from src.paths import RAW_DATA_DIR, CACHE_DIR

# -----------------------------
# Reproducibility
# -----------------------------
RANDOM_SEED = 42

# -----------------------------
# Signal / window configuration
# -----------------------------
FS = 250
WINDOW_SEC = 4
WINDOW_SAMPLES = FS * WINDOW_SEC
OVERLAP = 0.5
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))

N_CHANNELS = 63
N_BANDS = 4
BANDPOWER_FEATURE_DIM = N_CHANNELS * N_BANDS

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

# -----------------------------
# Canonical dataset locations
# -----------------------------
DESIGN_DATASET_DIR = RAW_DATA_DIR / "Design_EEG_Dataset"
CREATIVITY_DATASET_DIR = RAW_DATA_DIR / "Creativity_EEG_Dataset"

DATASET_DIRS = {
    "design": DESIGN_DATASET_DIR,
    "creativity": CREATIVITY_DATASET_DIR,
}

# -----------------------------
# Label mappings
# -----------------------------
FINAL_LABEL_TO_INT = {
    "REST": 0,
    "IG": 1,
    "IE": 2,
}

INT_TO_FINAL_LABEL = {v: k for k, v in FINAL_LABEL_TO_INT.items()}

DESIGN_TASK_TO_FINAL = {
    "RST1": "REST",
    "RST2": "REST",
    "IG": "IG",
    "IE": "IE",
    "PU": None,
    "RIG": None,
    "RIE": None,
}

CREATIVITY_TASK_TO_FINAL = {
    "RST1": "REST",
    "RST2": "REST",
    "IDG": "IG",
    "IDR": "IE",
    "IDE": None,
}

EXPECTED_SUBJECTS = {
    "design": 27,
    "creativity": 28,
}

# -----------------------------
# Cache files
# -----------------------------
CACHE_FILES = {
    "design_windows": CACHE_DIR / "design_windows.npz",
    "design_bandpower": CACHE_DIR / "design_bandpower.npz",
    "creativity_windows": CACHE_DIR / "creativity_windows.npz",
    "creativity_bandpower": CACHE_DIR / "creativity_bandpower.npz",
    "cache_manifest": CACHE_DIR / "cache_manifest.json",
}