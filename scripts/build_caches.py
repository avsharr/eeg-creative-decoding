from __future__ import annotations

import argparse
import sys

from src.config import (
    BANDPOWER_FEATURE_DIM,
    CACHE_FILES,
    CREATIVITY_DATASET_DIR,
    DESIGN_DATASET_DIR,
)
from src.labels import CREATIVITY_TASK_TO_FINAL, DESIGN_TASK_TO_FINAL
from src.paths import ensure_project_dirs
from src.eeg_utils import (
    build_bandpower_cache_from_windows_cache,
    build_windows_cache_for_dataset,
    cache_ready,
    report_grouping_sanity,
    validate_bandpower_cache,
    validate_windows_cache,
)
from src.io_utils import atomic_write_json, utc_now_str


def write_cache_manifest() -> None:
    manifest = {
        "saved_at_utc": utc_now_str(),
        "cache_files": {k: str(v) for k, v in CACHE_FILES.items()},
    }
    atomic_write_json(CACHE_FILES["cache_manifest"], manifest)


def build_design(force_windows: bool = False, force_bandpower: bool = False) -> None:
    if force_windows or not cache_ready("design_windows"):
        print("\n[BUILD] design_windows")
        build_windows_cache_for_dataset(
            dataset_name="design",
            dataset_dir=DESIGN_DATASET_DIR,
            task_to_final_map=DESIGN_TASK_TO_FINAL,
            cache_key="design_windows",
        )
    else:
        print("\n[SKIP] design_windows already exists")

    if force_bandpower or not cache_ready("design_bandpower"):
        print("\n[BUILD] design_bandpower")
        build_bandpower_cache_from_windows_cache(
            source_cache_key="design_windows",
            target_cache_key="design_bandpower",
            dataset_name="design",
        )
    else:
        print("\n[SKIP] design_bandpower already exists")


def build_creativity(force_windows: bool = False, force_bandpower: bool = False) -> None:
    if force_windows or not cache_ready("creativity_windows"):
        print("\n[BUILD] creativity_windows")
        build_windows_cache_for_dataset(
            dataset_name="creativity",
            dataset_dir=CREATIVITY_DATASET_DIR,
            task_to_final_map=CREATIVITY_TASK_TO_FINAL,
            cache_key="creativity_windows",
        )
    else:
        print("\n[SKIP] creativity_windows already exists")

    if force_bandpower or not cache_ready("creativity_bandpower"):
        print("\n[BUILD] creativity_bandpower")
        build_bandpower_cache_from_windows_cache(
            source_cache_key="creativity_windows",
            target_cache_key="creativity_bandpower",
            dataset_name="creativity",
        )
    else:
        print("\n[SKIP] creativity_bandpower already exists")


def validate_all(run_grouping_sanity: bool = False) -> None:
    print("\n[VALIDATE] windows caches")
    validate_windows_cache("design_windows")
    validate_windows_cache("creativity_windows")

    print("\n[VALIDATE] bandpower caches")
    validate_bandpower_cache("design_bandpower", expected_feature_dim=BANDPOWER_FEATURE_DIM)
    validate_bandpower_cache("creativity_bandpower", expected_feature_dim=BANDPOWER_FEATURE_DIM)

    if run_grouping_sanity:
        print("\n[GROUPING SANITY]")
        report_grouping_sanity("design_windows")
        report_grouping_sanity("creativity_windows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EEG window and bandpower caches")
    parser.add_argument(
        "--dataset",
        choices=["design", "creativity", "all"],
        default="all",
        help="Which dataset to build",
    )
    parser.add_argument(
        "--force-windows",
        action="store_true",
        help="Rebuild windows cache even if it exists",
    )
    parser.add_argument(
        "--force-bandpower",
        action="store_true",
        help="Rebuild bandpower cache even if it exists",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip building and only validate caches",
    )
    parser.add_argument(
        "--grouping-sanity",
        action="store_true",
        help="Run grouping leakage sanity checks after validation",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    write_cache_manifest()

    if args.validate_only:
        validate_all(run_grouping_sanity=args.grouping_sanity)
        return

    if args.dataset in ("design", "all"):
        build_design(force_windows=args.force_windows, force_bandpower=args.force_bandpower)

    if args.dataset in ("creativity", "all"):
        build_creativity(force_windows=args.force_windows, force_bandpower=args.force_bandpower)

    validate_all(run_grouping_sanity=args.grouping_sanity)
    print("\nAll requested caches are ready.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[DATA ERROR]\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)