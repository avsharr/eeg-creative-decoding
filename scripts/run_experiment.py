from __future__ import annotations

import argparse
import sys

from src.experiment_io import run_experiment_if_needed
from src.experiment_registry import get_experiment, list_runnable_experiments
from src.runners import (
    run_cross_subject_classical_experiment,
    run_deep_experiment_by_name,
    run_within_subject_classical_experiment,
)


def run_classical_dispatch(experiment_name: str, cfg: dict) -> dict:
    dataset_name = cfg["dataset"]
    evaluation = cfg["evaluation"]
    model_type = cfg["model"]

    if evaluation == "within":
        return run_within_subject_classical_experiment(
            dataset_name=dataset_name,
            model_type=model_type,
        )

    if evaluation == "cross_subject":
        return run_cross_subject_classical_experiment(
            dataset_name=dataset_name,
            model_type=model_type,
        )

    raise ValueError(
        f"Unsupported classical experiment '{experiment_name}' with evaluation='{evaluation}'"
    )


def run_deep_dispatch(experiment_name: str, cfg: dict) -> dict:
    return run_deep_experiment_by_name(experiment_name)


def run_one_experiment(experiment_name: str, force_rerun: bool = False):
    cfg = get_experiment(experiment_name)

    if cfg["model"] == "multiple":
        raise ValueError(
            f"'{experiment_name}' is an umbrella entry, not a directly runnable experiment."
        )

    if cfg["family"] == "classical":
        run_fn = lambda name: run_classical_dispatch(name, cfg)
    elif cfg["family"] == "deep":
        run_fn = lambda name: run_deep_dispatch(name, cfg)
    else:
        raise ValueError(
            f"Unsupported family='{cfg['family']}' for experiment '{experiment_name}'"
        )

    return run_experiment_if_needed(
        experiment_name=experiment_name,
        run_fn=run_fn,
        force_rerun=force_rerun,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EEG ML experiment")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=list_runnable_experiments(),
        help="Experiment name from src.experiment_registry.EXPERIMENTS",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if the experiment already has completed artifacts",
    )
    args = parser.parse_args()

    run_one_experiment(
        experiment_name=args.experiment,
        force_rerun=args.force,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)