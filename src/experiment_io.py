from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.io_utils import atomic_save_npz, atomic_write_json
from src.paths import EXPERIMENTS_DIR, ensure_project_dirs


DONE_MARKER_FILE = "_DONE"
FAILED_MARKER_FILE = "_FAILED"
RUNNING_MARKER_FILE = "_RUNNING"


def get_experiment_dir(experiment_name: str) -> Path:
    ensure_project_dirs()
    exp_dir = EXPERIMENTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_experiment_paths(experiment_name: str) -> Dict[str, Path]:
    exp_dir = get_experiment_dir(experiment_name)
    return {
        "dir": exp_dir,
        "metrics": exp_dir / "metrics.json",
        "config": exp_dir / "config.json",
        "notes": exp_dir / "notes.json",
        "history": exp_dir / "history.json",
        "confusion_matrix": exp_dir / "confusion_matrix.npy",
        "predictions": exp_dir / "predictions.npz",
        "done": exp_dir / DONE_MARKER_FILE,
        "failed": exp_dir / FAILED_MARKER_FILE,
        "running": exp_dir / RUNNING_MARKER_FILE,
    }


def experiment_is_complete(experiment_name: str) -> bool:
    paths = get_experiment_paths(experiment_name)
    required = [
        paths["metrics"],
        paths["config"],
        paths["notes"],
        paths["history"],
        paths["confusion_matrix"],
        paths["predictions"],
        paths["done"],
    ]
    return all(p.exists() for p in required)


def mark_running(experiment_name: str) -> None:
    paths = get_experiment_paths(experiment_name)
    paths["running"].write_text("running\n", encoding="utf-8")
    if paths["failed"].exists():
        paths["failed"].unlink()


def mark_done(experiment_name: str) -> None:
    paths = get_experiment_paths(experiment_name)
    if paths["running"].exists():
        paths["running"].unlink()
    if paths["failed"].exists():
        paths["failed"].unlink()
    paths["done"].write_text("done\n", encoding="utf-8")


def mark_failed(experiment_name: str, message: str = "") -> None:
    paths = get_experiment_paths(experiment_name)
    if paths["running"].exists():
        paths["running"].unlink()
    paths["failed"].write_text(message + "\n", encoding="utf-8")


def reset_experiment_folder(experiment_name: str) -> None:
    exp_dir = get_experiment_dir(experiment_name)
    for p in exp_dir.glob("*"):
        if p.is_file():
            p.unlink()


def save_final_experiment_artifacts(experiment_name: str, result: dict) -> None:
    paths = get_experiment_paths(experiment_name)

    required_keys = [
        "config",
        "metrics",
        "notes",
        "history",
        "confusion_matrix_array",
        "y_true",
        "y_pred",
    ]
    missing = [k for k in required_keys if k not in result]
    if missing:
        raise KeyError(
            f"Result for '{experiment_name}' is missing required keys: {missing}"
        )

    atomic_write_json(paths["config"], result["config"])
    atomic_write_json(paths["metrics"], result["metrics"])
    atomic_write_json(paths["notes"], result["notes"])
    atomic_write_json(paths["history"], result["history"])

    np.save(paths["confusion_matrix"], np.asarray(result["confusion_matrix_array"]))

    pred_payload = {
        "y_true": np.asarray(result["y_true"]),
        "y_pred": np.asarray(result["y_pred"]),
    }

    optional_prediction_keys = [
        "y_prob",
        "segment_y_true",
        "segment_y_pred",
        "segment_prob",
        "segment_ids",
        "subject_ids",
    ]
    for key in optional_prediction_keys:
        if key in result and result[key] is not None:
            pred_payload[key] = np.asarray(result[key])

    atomic_save_npz(paths["predictions"], **pred_payload)
    if "sklearn_model" in result and result["sklearn_model"] is not None:
        from src.io_utils import atomic_joblib_dump
        atomic_joblib_dump(paths["dir"] / "model.joblib", result["sklearn_model"])
    mark_done(experiment_name)


def run_experiment_if_needed(
    experiment_name: str,
    run_fn,
    force_rerun: bool = False,
):
    if experiment_is_complete(experiment_name) and not force_rerun:
        print(f"[SKIP] {experiment_name} already completed.")
        return None

    if force_rerun:
        print(f"[RESET] Removing previous artifacts for {experiment_name}")
        reset_experiment_folder(experiment_name)

    print(f"[START] {experiment_name}")
    mark_running(experiment_name)

    try:
        result = run_fn(experiment_name)
        save_final_experiment_artifacts(experiment_name, result)
        print(f"[DONE] {experiment_name}")
        return result
    except Exception as e:
        mark_failed(experiment_name, message=str(e))
        print(f"[FAILED] {experiment_name}: {e}")
        raise