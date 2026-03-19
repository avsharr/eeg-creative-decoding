from __future__ import annotations

import os
import zipfile
from pathlib import Path

import gdown


FOLDER_ID = "1jyYfaerhUnetGNLq6yOFqz9CuztoK0Eu"


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def count_mat_files(folder: Path) -> int:
    return len(list(folder.rglob("*.mat"))) if folder.exists() else 0


def download_data() -> None:
    raw_path = Path("data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)

    design_dir = raw_path / "Design_EEG_Dataset"
    creativity_dir = raw_path / "Creativity_EEG_Dataset"

    print(f"[DOWNLOAD] Google Drive folder ID: {FOLDER_ID}")
    gdown.download_folder(id=FOLDER_ID, output=str(raw_path), quiet=False)

    for item in raw_path.iterdir():
        if item.suffix.lower() == ".zip":
            lower_name = item.name.lower()

            if "design" in lower_name:
                extract_dir = design_dir
            elif "creativity" in lower_name:
                extract_dir = creativity_dir
            else:
                print(f"[SKIP] Could not infer dataset type from zip file name: {item.name}")
                continue

            print(f"[EXTRACT] {item.name} -> {extract_dir}")
            extract_zip(item, extract_dir)
            os.remove(item)

    print("\n[CHECK] Final dataset contents:")
    print(f" - {design_dir}: {count_mat_files(design_dir)} .mat files")
    print(f" - {creativity_dir}: {count_mat_files(creativity_dir)} .mat files")


if __name__ == "__main__":
    download_data()