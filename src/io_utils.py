from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_parent(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    text = json.dumps(obj, indent=indent, ensure_ascii=False)
    atomic_write_text(path, text)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    ensure_parent(path)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".npy") as tmp:
        tmp_path = Path(tmp.name)
    np.save(tmp_path, array)
    tmp_path.replace(path)


def atomic_save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    ensure_parent(path)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".npz") as tmp:
        tmp_path = Path(tmp.name)
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def atomic_joblib_dump(path: Path, obj: Any) -> None:
    path = Path(path)
    ensure_parent(path)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".joblib") as tmp:
        tmp_path = Path(tmp.name)
    joblib.dump(obj, tmp_path)
    tmp_path.replace(path)