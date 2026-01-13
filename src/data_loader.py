import gdown
import zipfile
import os
from pathlib import Path

# data from colab
FOLDER_ID = "1jyYfaerhUnetGNLq6yOFqz9CuztoK0Eu"

def download_data():
    # data will be saved in data/raw
    raw_path = Path("data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(id=FOLDER_ID, output=str(raw_path), quiet=False)

    for item in raw_path.iterdir():
        if item.suffix == ".zip":
            extract_dir = raw_path / item.stem
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(item, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            os.remove(item)

if __name__ == "__main__":
    download_data()