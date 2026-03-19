from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from src.config import (
    CREATIVITY_DATASET_DIR,
    DESIGN_DATASET_DIR,
    DESIGN_TASK_TO_FINAL,
    CREATIVITY_TASK_TO_FINAL,
    FINAL_LABEL_TO_INT,
    WINDOW_SAMPLES,
    STEP_SAMPLES,
)
from src.eeg_utils import (
    discover_dataset_files,
    load_subject_file_records,
    make_overlapping_windows,
)
from src.paths import CACHE_DIR, ensure_project_dirs


def estimate_total_windows(dataset_name: str, dataset_dir: Path, task_to_final_map: dict) -> int:
    total = 0
    files = discover_dataset_files(dataset_name=dataset_name, dataset_dir=dataset_dir)

    for file_path in files:
        records = load_subject_file_records(file_path, dataset_name=dataset_name)
        for rec in records:
            task_name = rec["task_name"]
            final_label = task_to_final_map.get(task_name, None)
            if final_label is None:
                continue

            eeg = rec["eeg"]
            n_samples = eeg.shape[1]
            if n_samples < WINDOW_SAMPLES:
                continue

            n_w = 1 + (n_samples - WINDOW_SAMPLES) // STEP_SAMPLES
            total += n_w

    return total


def build_raw_windows_h5_for_dataset(
    dataset_name: str,
    dataset_dir: Path,
    task_to_final_map: dict,
    output_path: Path,
):
    files = discover_dataset_files(dataset_name=dataset_name, dataset_dir=dataset_dir)
    total_windows = estimate_total_windows(dataset_name, dataset_dir, task_to_final_map)

    if total_windows == 0:
        raise RuntimeError(f"No windows found for dataset={dataset_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        X_ds = h5.create_dataset(
            "X_windows",
            shape=(total_windows, 63, WINDOW_SAMPLES),
            dtype="float32",
            chunks=(64, 63, WINDOW_SAMPLES),
            compression="gzip",
        )
        y_ds = h5.create_dataset("y", shape=(total_windows,), dtype="int64", compression="gzip")
        subj_ds = h5.create_dataset(
            "subject_ids",
            shape=(total_windows,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
        )
        seg_ds = h5.create_dataset(
            "segment_ids",
            shape=(total_windows,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
        )
        task_ds = h5.create_dataset(
            "task_names",
            shape=(total_windows,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
        )

        write_pos = 0

        for file_path in files:
            records = load_subject_file_records(file_path, dataset_name=dataset_name)

            for rec in records:
                subject_id = rec["subject_id"]
                task_name = rec["task_name"]
                run_id = rec["run_id"]
                eeg = rec["eeg"]
                source_variable = rec["source_variable"]

                final_label_name = task_to_final_map.get(task_name, None)
                if final_label_name is None:
                    continue

                windows = make_overlapping_windows(
                    X_continuous=eeg,
                    window_samples=WINDOW_SAMPLES,
                    step_samples=STEP_SAMPLES,
                    normalize_per_window=True,
                )

                n_w = windows.shape[0]
                if n_w == 0:
                    continue

                y_int = FINAL_LABEL_TO_INT[final_label_name]
                segment_id = f"{subject_id}__{run_id}__{task_name}__{source_variable}"

                end = write_pos + n_w
                X_ds[write_pos:end] = windows
                y_ds[write_pos:end] = np.full(n_w, y_int, dtype=np.int64)
                subj_ds[write_pos:end] = np.array([subject_id] * n_w, dtype=object)
                seg_ds[write_pos:end] = np.array([segment_id] * n_w, dtype=object)
                task_ds[write_pos:end] = np.array([task_name] * n_w, dtype=object)

                write_pos = end

        h5.attrs["dataset_name"] = dataset_name
        h5.attrs["n_samples"] = write_pos

    print(f"[DONE] {dataset_name} raw windows -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build memory-safe HDF5 raw-window caches for EEGNet")
    parser.add_argument("--dataset", choices=["design", "creativity", "all"], default="all")
    args = parser.parse_args()

    ensure_project_dirs()

    if args.dataset in ("design", "all"):
        build_raw_windows_h5_for_dataset(
            dataset_name="design",
            dataset_dir=DESIGN_DATASET_DIR,
            task_to_final_map=DESIGN_TASK_TO_FINAL,
            output_path=CACHE_DIR / "design_windows.h5",
        )

    if args.dataset in ("creativity", "all"):
        build_raw_windows_h5_for_dataset(
            dataset_name="creativity",
            dataset_dir=CREATIVITY_DATASET_DIR,
            task_to_final_map=CREATIVITY_TASK_TO_FINAL,
            output_path=CACHE_DIR / "creativity_windows.h5",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)