from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch

from src.config import (
    BANDS,
    CACHE_FILES,
    FINAL_LABEL_TO_INT,
    FS,
    N_CHANNELS,
    OVERLAP,
    RANDOM_SEED,
    STEP_SAMPLES,
    WINDOW_SAMPLES,
    WINDOW_SEC,
)
from src.io_utils import atomic_save_npz, atomic_write_json, read_json, utc_now_str
from src.labels import INT_TO_FINAL_LABEL


# -----------------------------
# File discovery helpers
# -----------------------------
def list_files_recursive(root: Path, extensions=None) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")

    files = []
    for p in root.rglob("*"):
        if p.is_file():
            if extensions is None:
                files.append(p)
            else:
                ext_list = [e.lower() for e in extensions]
                if p.suffix.lower() in ext_list:
                    files.append(p)
    return sorted(files)


def summarize_file_list(files: List[Path], max_show: int = 20) -> None:
    print(f"Found {len(files)} files")
    for p in files[:max_show]:
        print(" -", p)
    if len(files) > max_show:
        print(f"... and {len(files) - max_show} more")


def assert_dataset_dir_ready(dataset_name: str, dataset_dir: Path, min_mat_files: int = 1) -> None:
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"[{dataset_name}] Dataset folder does not exist: {dataset_dir}\n"
            f"Run: python -m scripts.download_data --show-instructions"
        )

    mat_files = sorted(dataset_dir.rglob("*.mat"))
    if len(mat_files) < min_mat_files:
        raise FileNotFoundError(
            f"[{dataset_name}] No .mat files were found under: {dataset_dir}\n"
            f"Put the raw dataset files there, then run:\n"
            f"  python -m scripts.download_data --check-only\n"
            f"  python -m scripts.build_caches --dataset {dataset_name}"
        )


def discover_dataset_files(dataset_name: str, dataset_dir: Path) -> List[Path]:
    assert_dataset_dir_ready(dataset_name=dataset_name, dataset_dir=dataset_dir, min_mat_files=1)
    return list_files_recursive(dataset_dir, extensions=[".mat"])


# -----------------------------
# Windowing helpers
# -----------------------------
def zscore_per_window_per_channel(X_window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = X_window.mean(axis=1, keepdims=True)
    std = X_window.std(axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X_window - mean) / std


def make_overlapping_windows(
    X_continuous: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    step_samples: int = STEP_SAMPLES,
    normalize_per_window: bool = True,
) -> np.ndarray:
    n_channels, n_total_samples = X_continuous.shape

    if n_total_samples < window_samples:
        return np.empty((0, n_channels, window_samples), dtype=np.float32)

    windows = []
    for start in range(0, n_total_samples - window_samples + 1, step_samples):
        end = start + window_samples
        w = X_continuous[:, start:end].astype(np.float32, copy=True)
        if normalize_per_window:
            w = zscore_per_window_per_channel(w)
        windows.append(w)

    if len(windows) == 0:
        return np.empty((0, n_channels, window_samples), dtype=np.float32)

    return np.stack(windows, axis=0).astype(np.float32)


# -----------------------------
# Bandpower features
# -----------------------------
def compute_bandpower_features_from_windows(
    X_windows: np.ndarray,
    fs: int = FS,
    bands: dict = None,
    welch_nperseg: int = 256,
) -> np.ndarray:
    if bands is None:
        bands = BANDS

    n_windows, n_channels, n_samples = X_windows.shape
    feature_list = []

    for i in range(n_windows):
        window = X_windows[i]
        channel_features = []

        for ch in range(n_channels):
            freqs, psd = welch(window[ch], fs=fs, nperseg=min(welch_nperseg, n_samples))

            for _, (fmin, fmax) in bands.items():
                band_mask = (freqs >= fmin) & (freqs < fmax)
                if not np.any(band_mask):
                    bp = 0.0
                else:
                    bp = np.trapezoid(psd[band_mask], freqs[band_mask])
                channel_features.append(bp)

        feature_list.append(channel_features)

    return np.asarray(feature_list, dtype=np.float32)


# -----------------------------
# MAT adapters
# -----------------------------
DESIGN_TASK_VAR_REGEX = re.compile(
    r"^Design_(?P<pn>\d+)_(?P<dp>[A-Za-z0-9]+)_(?P<task>PU|IG|RIG|IE|RIE)$",
    re.IGNORECASE,
)

DESIGN_REST_VAR_REGEX = re.compile(
    r"^Design_(?P<pn>\d+)_(?P<task>RST1|RST2)$",
    re.IGNORECASE,
)

CREATIVITY_TASK_VAR_REGEX = re.compile(
    r"^Creativity_(?P<pn>\d+)_(?P<tn>[A-Za-z0-9]+)_(?P<task>IDG|IDE|IDR)$",
    re.IGNORECASE,
)

CREATIVITY_REST_VAR_REGEX = re.compile(
    r"^Creativity_(?P<pn>\d+)_(?P<task>RST1|RST2)$",
    re.IGNORECASE,
)


def detect_mat_format(mat_path: Path) -> str:
    mat_path = Path(mat_path)
    try:
        with open(mat_path, "rb") as f:
            header = f.read(128)
        if b"MATLAB 7.3 MAT-file" in header:
            return "v7.3"
        return "old"
    except Exception:
        return "unknown"


def extract_subject_id_from_filename(file_path: Path, dataset_name: str) -> str:
    m = re.search(r"Sub_(\d+)", file_path.stem, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Could not extract subject number from filename: {file_path.name}")

    pn = int(m.group(1))
    prefix = "design" if dataset_name.lower() == "design" else "creativity"
    return f"{prefix}_{pn:02d}"


def coerce_eeg_array_to_63xn(x, var_name: str, file_path: Path) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{var_name}: not a numpy array in {file_path.name}")

    if x.ndim != 2:
        raise ValueError(f"{var_name}: expected 2D array, got shape={x.shape} in {file_path.name}")

    if not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"{var_name}: non-numeric dtype={x.dtype} in {file_path.name}")

    x = np.asarray(x, dtype=np.float32)

    if x.shape[0] == 63:
        return x
    if x.shape[1] == 63:
        return x.T

    raise ValueError(f"{var_name}: neither dimension is 63, got shape={x.shape} in {file_path.name}")


def load_mat_variables_old(file_path: Path) -> dict:
    mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    out = {}
    for k, v in mat.items():
        if not k.startswith("__"):
            out[k] = v
    return out


def load_mat_variables_v73(file_path: Path) -> dict:
    out = {}
    with h5py.File(file_path, "r") as f:
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                out[k] = np.array(obj)
    return out


def load_mat_variables_any(file_path: Path) -> Tuple[dict, str]:
    mat_format = detect_mat_format(file_path)
    if mat_format == "v7.3":
        variables = load_mat_variables_v73(file_path)
    else:
        variables = load_mat_variables_old(file_path)
    return variables, mat_format


def parse_variable_name(var_name: str, dataset_name: str):
    dataset_name = dataset_name.lower()

    if dataset_name == "design":
        m = DESIGN_TASK_VAR_REGEX.match(var_name)
        if m is not None:
            return m.group("task").upper(), str(m.group("dp"))

        m = DESIGN_REST_VAR_REGEX.match(var_name)
        if m is not None:
            return m.group("task").upper(), "REST"

        return None, None

    if dataset_name == "creativity":
        m = CREATIVITY_TASK_VAR_REGEX.match(var_name)
        if m is not None:
            return m.group("task").upper(), str(m.group("tn"))

        m = CREATIVITY_REST_VAR_REGEX.match(var_name)
        if m is not None:
            return m.group("task").upper(), "REST"

        return None, None

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def load_subject_file_records(file_path: Path, dataset_name: str) -> list:
    file_path = Path(file_path)
    dataset_name = dataset_name.lower()

    if file_path.suffix.lower() != ".mat":
        raise ValueError(f"Expected .mat file, got: {file_path}")

    subject_id = extract_subject_id_from_filename(file_path, dataset_name)
    variables, mat_format = load_mat_variables_any(file_path)

    records = []
    skipped = []

    for var_name, var_value in variables.items():
        task_name, run_id = parse_variable_name(var_name, dataset_name)

        if task_name is None:
            skipped.append((var_name, "name_pattern_mismatch"))
            continue

        try:
            eeg = coerce_eeg_array_to_63xn(var_value, var_name=var_name, file_path=file_path)
        except Exception as e:
            skipped.append((var_name, f"bad_array: {e}"))
            continue

        records.append(
            {
                "subject_id": subject_id,
                "task_name": task_name,
                "run_id": str(run_id),
                "eeg": eeg,
                "source_file": str(file_path),
                "source_variable": var_name,
                "mat_format": mat_format,
            }
        )

    if len(records) == 0:
        raise RuntimeError(
            f"No usable EEG variables found in {file_path.name}. First skipped examples: {skipped[:10]}"
        )

    return records


# -----------------------------
# Cache metadata helpers
# -----------------------------
def get_cache_meta_path(cache_key: str) -> Path:
    if cache_key not in CACHE_FILES:
        raise KeyError(f"Unknown cache key: {cache_key}")
    return CACHE_FILES[cache_key].with_suffix(".meta.json")


def build_standard_cache_metadata(cache_key: str, dataset_name: str, extra: dict | None = None) -> dict:
    if extra is None:
        extra = {}

    meta = {
        "cache_key": cache_key,
        "dataset_name": dataset_name,
        "saved_at_utc": utc_now_str(),
        "random_seed": RANDOM_SEED,
        "fs": FS,
        "window_sec": WINDOW_SEC,
        "window_samples": WINDOW_SAMPLES,
        "overlap": OVERLAP,
        "step_samples": STEP_SAMPLES,
        "n_channels": N_CHANNELS,
        "bands": BANDS,
        "final_label_to_int": FINAL_LABEL_TO_INT,
    }
    meta.update(extra)
    return meta


def save_cache_with_metadata(cache_key: str, arrays: dict, metadata: dict) -> None:
    cache_path = CACHE_FILES[cache_key]
    meta_path = get_cache_meta_path(cache_key)

    atomic_save_npz(cache_path, **arrays)
    atomic_write_json(meta_path, metadata)

    print(f"[CACHE SAVED] {cache_key}")
    print("  npz :", cache_path)
    print("  meta:", meta_path)


def load_cache_with_metadata(cache_key: str):
    cache_path = CACHE_FILES[cache_key]
    meta_path = get_cache_meta_path(cache_key)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Cache metadata file not found: {meta_path}")

    data = np.load(cache_path, allow_pickle=True)
    metadata = read_json(meta_path, default={})
    return data, metadata


def cache_ready(cache_key: str) -> bool:
    return CACHE_FILES[cache_key].exists() and get_cache_meta_path(cache_key).exists()


# -----------------------------
# Cache builders
# -----------------------------
def build_windows_cache_for_dataset(
    dataset_name: str,
    dataset_dir: Path,
    task_to_final_map: dict,
    cache_key: str,
) -> None:
    files = discover_dataset_files(dataset_name=dataset_name, dataset_dir=dataset_dir)
    print(f"[{dataset_name}] discovered files: {len(files)}")
    summarize_file_list(files, max_show=20)

    X_list = []
    y_list = []
    subject_list = []
    segment_list = []
    task_list = []
    run_id_list = []
    source_file_list = []
    source_variable_list = []
    mat_format_list = []

    record_rows = []
    skipped_files = []
    skipped_variables = []

    for file_path in files:
        try:
            records = load_subject_file_records(file_path, dataset_name=dataset_name)
        except Exception as e:
            skipped_files.append({"file": str(file_path), "reason": f"exception_while_loading_file: {e}"})
            continue

        for rec in records:
            subject_id = rec["subject_id"]
            task_name = rec["task_name"]
            run_id = rec["run_id"]
            eeg = rec["eeg"]
            source_file = rec["source_file"]
            source_variable = rec["source_variable"]
            mat_format = rec["mat_format"]

            final_label_name = task_to_final_map.get(task_name, None)
            if final_label_name is None:
                skipped_variables.append(
                    {
                        "file": source_file,
                        "variable": source_variable,
                        "reason": f"task excluded or unmapped: {task_name}",
                    }
                )
                continue

            y_int = FINAL_LABEL_TO_INT[final_label_name]

            windows = make_overlapping_windows(
                X_continuous=eeg,
                window_samples=WINDOW_SAMPLES,
                step_samples=STEP_SAMPLES,
                normalize_per_window=True,
            )

            n_windows = windows.shape[0]
            if n_windows == 0:
                skipped_variables.append(
                    {
                        "file": source_file,
                        "variable": source_variable,
                        "reason": "not enough samples for one full window",
                    }
                )
                continue

            segment_id = f"{subject_id}__{run_id}__{task_name}__{source_variable}"
            segment_ids = np.array([segment_id] * n_windows, dtype=object)

            X_list.append(windows)
            y_list.append(np.full(n_windows, y_int, dtype=np.int64))
            subject_list.append(np.array([subject_id] * n_windows, dtype=object))
            segment_list.append(segment_ids)
            task_list.append(np.array([task_name] * n_windows, dtype=object))
            run_id_list.append(np.array([run_id] * n_windows, dtype=object))
            source_file_list.append(np.array([source_file] * n_windows, dtype=object))
            source_variable_list.append(np.array([source_variable] * n_windows, dtype=object))
            mat_format_list.append(np.array([mat_format] * n_windows, dtype=object))

            record_rows.append(
                {
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "task_name": task_name,
                    "final_label": final_label_name,
                    "n_windows": int(n_windows),
                    "segment_id": segment_id,
                    "source_file": source_file,
                    "source_variable": source_variable,
                    "mat_format": mat_format,
                    "n_channels": int(windows.shape[1]),
                    "window_samples": int(windows.shape[2]),
                }
            )

    if len(X_list) == 0:
        raise RuntimeError(f"No usable windows were created for dataset={dataset_name}")

    X_windows = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    subject_ids = np.concatenate(subject_list, axis=0)
    segment_ids = np.concatenate(segment_list, axis=0)
    task_names = np.concatenate(task_list, axis=0)
    run_ids = np.concatenate(run_id_list, axis=0)
    source_files = np.concatenate(source_file_list, axis=0)
    source_variables = np.concatenate(source_variable_list, axis=0)
    mat_formats = np.concatenate(mat_format_list, axis=0)

    arrays = {
        "X_windows": X_windows,
        "y": y,
        "subject_ids": subject_ids,
        "segment_ids": segment_ids,
        "task_names": task_names,
        "run_ids": run_ids,
        "source_files": source_files,
        "source_variables": source_variables,
        "mat_formats": mat_formats,
    }

    class_count_map = {}
    uniq_y, uniq_counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(uniq_y, uniq_counts):
        class_count_map[INT_TO_FINAL_LABEL[int(cls)]] = int(cnt)

    metadata = build_standard_cache_metadata(
        cache_key=cache_key,
        dataset_name=dataset_name,
        extra={
            "dataset_dir": str(dataset_dir),
            "n_total_windows": int(X_windows.shape[0]),
            "n_unique_segments": int(len(np.unique(segment_ids))),
            "n_channels": int(X_windows.shape[1]),
            "window_samples": int(X_windows.shape[2]),
            "n_subjects_found": int(len(np.unique(subject_ids))),
            "subjects_found": sorted([str(x) for x in np.unique(subject_ids)]),
            "class_counts": class_count_map,
            "record_rows": record_rows,
            "skipped_files": skipped_files,
            "skipped_variables": skipped_variables,
        },
    )

    save_cache_with_metadata(cache_key, arrays, metadata)

    print(f"[{dataset_name}] X_windows shape:", X_windows.shape)
    print(f"[{dataset_name}] y shape:", y.shape)
    print(f"[{dataset_name}] unique subjects:", len(np.unique(subject_ids)))
    print(f"[{dataset_name}] unique segments:", len(np.unique(segment_ids)))
    print(f"[{dataset_name}] unique tasks:", sorted(np.unique(task_names).tolist()))
    print(f"[{dataset_name}] class counts:", class_count_map)


def build_bandpower_cache_from_windows_cache(
    source_cache_key: str,
    target_cache_key: str,
    dataset_name: str,
) -> None:
    data, source_meta = load_cache_with_metadata(source_cache_key)

    X_windows = data["X_windows"]
    y = data["y"]
    subject_ids = data["subject_ids"]
    segment_ids = data["segment_ids"]
    task_names = data["task_names"]
    run_ids = data["run_ids"]
    source_files = data["source_files"]
    source_variables = data["source_variables"]
    mat_formats = data["mat_formats"]

    X_bandpower = compute_bandpower_features_from_windows(
        X_windows=X_windows,
        fs=FS,
        bands=BANDS,
        welch_nperseg=256,
    )

    arrays = {
        "X_bandpower": X_bandpower.astype(np.float32),
        "y": y.astype(np.int64),
        "subject_ids": subject_ids,
        "segment_ids": segment_ids,
        "task_names": task_names,
        "run_ids": run_ids,
        "source_files": source_files,
        "source_variables": source_variables,
        "mat_formats": mat_formats,
    }

    metadata = build_standard_cache_metadata(
        cache_key=target_cache_key,
        dataset_name=dataset_name,
        extra={
            "derived_from": source_cache_key,
            "source_cache_path": str(CACHE_FILES[source_cache_key]),
            "n_samples": int(X_bandpower.shape[0]),
            "feature_dim": int(X_bandpower.shape[1]),
            "class_counts": source_meta.get("class_counts", {}),
            "n_subjects_found": source_meta.get("n_subjects_found", None),
            "welch_nperseg": 256,
        },
    )

    save_cache_with_metadata(target_cache_key, arrays, metadata)

    print(f"[{dataset_name}] X_bandpower shape:", X_bandpower.shape)


# -----------------------------
# Validation helpers
# -----------------------------
def validate_windows_cache(
    cache_key: str,
    expected_n_channels: int = N_CHANNELS,
    expected_window_samples: int = WINDOW_SAMPLES,
) -> None:
    data, _ = load_cache_with_metadata(cache_key)

    X_windows = data["X_windows"]
    y = data["y"]
    subject_ids = data["subject_ids"]
    segment_ids = data["segment_ids"]
    task_names = data["task_names"]
    run_ids = data["run_ids"]
    source_files = data["source_files"]
    source_variables = data["source_variables"]
    mat_formats = data["mat_formats"]

    assert X_windows.ndim == 3, f"{cache_key}: X_windows must be 3D"
    assert X_windows.shape[1] == expected_n_channels, f"{cache_key}: wrong n_channels"
    assert X_windows.shape[2] == expected_window_samples, f"{cache_key}: wrong window_samples"
    assert len(y) == len(X_windows), f"{cache_key}: y length mismatch"
    assert len(subject_ids) == len(X_windows), f"{cache_key}: subject_ids length mismatch"
    assert len(segment_ids) == len(X_windows), f"{cache_key}: segment_ids length mismatch"
    assert len(task_names) == len(X_windows), f"{cache_key}: task_names length mismatch"
    assert len(run_ids) == len(X_windows), f"{cache_key}: run_ids length mismatch"
    assert len(source_files) == len(X_windows), f"{cache_key}: source_files length mismatch"
    assert len(source_variables) == len(X_windows), f"{cache_key}: source_variables length mismatch"
    assert len(mat_formats) == len(X_windows), f"{cache_key}: mat_formats length mismatch"

    print(f"[VALID] {cache_key}")
    print("  X_windows:", X_windows.shape)
    print(
        "  class counts:",
        {INT_TO_FINAL_LABEL[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    )
    print("  n_subjects:", len(np.unique(subject_ids)))
    print("  tasks:", sorted(np.unique(task_names).tolist())[:20])


def validate_bandpower_cache(cache_key: str, expected_feature_dim: int | None = None) -> None:
    data, _ = load_cache_with_metadata(cache_key)

    X_bandpower = data["X_bandpower"]
    y = data["y"]
    subject_ids = data["subject_ids"]
    segment_ids = data["segment_ids"]

    assert X_bandpower.ndim == 2, f"{cache_key}: X_bandpower must be 2D"
    assert len(y) == len(X_bandpower), f"{cache_key}: y length mismatch"
    assert len(subject_ids) == len(X_bandpower), f"{cache_key}: subject_ids length mismatch"
    assert len(segment_ids) == len(X_bandpower), f"{cache_key}: segment_ids length mismatch"

    if expected_feature_dim is not None:
        assert X_bandpower.shape[1] == expected_feature_dim, f"{cache_key}: wrong feature_dim"

    print(f"[VALID] {cache_key}")
    print("  X_bandpower:", X_bandpower.shape)
    print(
        "  class counts:",
        {INT_TO_FINAL_LABEL[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    )
    print("  n_subjects:", len(np.unique(subject_ids)))


def report_grouping_sanity(cache_key: str) -> None:
    data, _ = load_cache_with_metadata(cache_key)

    subject_ids = data["subject_ids"]
    segment_ids = data["segment_ids"]
    task_names = data["task_names"]
    source_variables = data["source_variables"]

    print("=" * 70)
    print(f"Grouping sanity report: {cache_key}")
    print("=" * 70)
    print("n_samples:", len(subject_ids))
    print("n_subjects:", len(np.unique(subject_ids)))
    print("n_segments:", len(np.unique(segment_ids)))
    print("n_source_variables:", len(np.unique(source_variables)))

    df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "segment_id": segment_ids,
            "task_name": task_names,
            "source_variable": source_variables,
        }
    )

    seg_per_subject = df.groupby("subject_id")["segment_id"].nunique()
    print("\nUnique segments per subject:")
    print(seg_per_subject.describe())

    windows_per_segment = df.groupby("segment_id").size()
    print("\nWindows per segment:")
    print(windows_per_segment.describe())

    mismatch = df.groupby("segment_id")["source_variable"].nunique().max()
    print("\nMax number of source_variable values inside one segment_id:", mismatch)

    if mismatch != 1:
        print("[WARN] A segment_id maps to multiple source variables.")
    else:
        print("[OK] Each segment_id maps to exactly one source variable.")