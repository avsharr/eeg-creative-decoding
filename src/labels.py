from __future__ import annotations

from src.config import (
    FINAL_LABEL_TO_INT,
    INT_TO_FINAL_LABEL,
    DESIGN_TASK_TO_FINAL,
    CREATIVITY_TASK_TO_FINAL,
    EXPECTED_SUBJECTS,
)

__all__ = [
    "FINAL_LABEL_TO_INT",
    "INT_TO_FINAL_LABEL",
    "DESIGN_TASK_TO_FINAL",
    "CREATIVITY_TASK_TO_FINAL",
    "EXPECTED_SUBJECTS",
    "get_task_mapping",
]


def get_task_mapping(dataset_name: str) -> dict:
    dataset_name = dataset_name.lower()
    if dataset_name == "design":
        return DESIGN_TASK_TO_FINAL
    if dataset_name == "creativity":
        return CREATIVITY_TASK_TO_FINAL
    raise ValueError(f"Unknown dataset_name: {dataset_name}")