from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from src.paths import RAW_DATA_DIR, TMP_DIR, ensure_project_dirs


DATASETS = {
    "design": {
        "target_dir": RAW_DATA_DIR / "Design_EEG_Dataset",
        "source_page": "https://data.mendeley.com/datasets/h4rf6wzjcr/1",
        "expected_subject_count": 27,
        "allowed_names": ["Design_EEG_Dataset"],
    },
    "creativity": {
        "target_dir": RAW_DATA_DIR / "Creativity_EEG_Dataset",
        "source_page": "https://data.mendeley.com/datasets/24yp3xp58b/1",
        "expected_subject_count": 28,
        "allowed_names": ["Creativity_EEG_Dataset"],
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_mat_files(root: Path) -> int:
    return len(list(root.rglob("*.mat"))) if root.exists() else 0


def dataset_ready(name: str) -> bool:
    info = DATASETS[name]
    target_dir = info["target_dir"]
    n_mat = count_mat_files(target_dir)
    return n_mat >= info["expected_subject_count"]


def validate_dataset(name: str) -> bool:
    info = DATASETS[name]
    target_dir = info["target_dir"]

    if not target_dir.exists():
        print(f"[MISSING] {name}: folder does not exist -> {target_dir}")
        print(f"Public source: {info['source_page']}")
        return False

    mat_files = sorted(target_dir.rglob("*.mat"))
    if not mat_files:
        print(f"[INVALID] {name}: no .mat files found under {target_dir}")
        print(f"Public source: {info['source_page']}")
        return False

    print(f"[OK] {name}: found {len(mat_files)} .mat files under {target_dir}")

    expected = info["expected_subject_count"]
    if len(mat_files) < expected:
        print(f"[WARN] {name}: expected about {expected} subject files, found {len(mat_files)}")
        return False

    return True


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def extract_all_archives_from_downloads() -> None:
    ensure_dir(TMP_DIR)

    zip_files = sorted(TMP_DIR.rglob("*.zip"))
    if not zip_files:
        print(f"[INFO] No zip files found under {TMP_DIR}")
        return

    for zip_path in zip_files:
        lower = zip_path.name.lower()

        if "design" in lower:
            target = DATASETS["design"]["target_dir"]
            print(f"[EXTRACT] {zip_path.name} -> {target}")
            extract_zip(zip_path, target)

        elif "creativity" in lower:
            target = DATASETS["creativity"]["target_dir"]
            print(f"[EXTRACT] {zip_path.name} -> {target}")
            extract_zip(zip_path, target)

        else:
            print(f"[SKIP] Could not infer dataset from zip name: {zip_path.name}")


def print_manual_instructions() -> None:
    print("\nManual dataset setup")
    print("=" * 60)
    print("1) Open these public dataset pages in your browser:")
    print(f"   - Design:     {DATASETS['design']['source_page']}")
    print(f"   - Creativity: {DATASETS['creativity']['source_page']}")
    print("\n2) Download the dataset files or the 'Download All' archive(s).")
    print(f"3) Either:")
    print(f"   A. Extract them directly into:")
    print(f"      - {DATASETS['design']['target_dir']}")
    print(f"      - {DATASETS['creativity']['target_dir']}")
    print("   or")
    print(f"   B. Put the downloaded zip files into:")
    print(f"      - {TMP_DIR}")
    print("      and run this script again with --extract-only")
    print("\n4) Then validate with:")
    print("   python -m scripts.download_data --check-only")
    print("\n5) Then continue with:")
    print("   python -m scripts.build_caches --dataset all")


def main() -> None:
    ensure_project_dirs()

    parser = argparse.ArgumentParser(description="Prepare and validate public EEG datasets")
    parser.add_argument(
        "--dataset",
        choices=["design", "creativity", "all"],
        default="all",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate dataset folders",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract zip files found in data/_downloads",
    )
    parser.add_argument(
        "--show-instructions",
        action="store_true",
        help="Print manual download instructions",
    )
    args = parser.parse_args()

    names = ["design", "creativity"] if args.dataset == "all" else [args.dataset]

    if args.show_instructions:
        print_manual_instructions()
        return

    if args.extract_only:
        extract_all_archives_from_downloads()

    if args.check_only or args.extract_only:
        all_ok = True
        for name in names:
            ok = validate_dataset(name)
            all_ok = all_ok and ok
        if not all_ok:
            sys.exit(1)
        return

    print_manual_instructions()


if __name__ == "__main__":
    main()