# Notebooks

This folder contains two closely related versions of the same project notebook.

## 1. Drive-based notebook

**File:** `Final_ML_Project_archived_drive.ipynb`

This is the original notebook version used during the project work.

Characteristics:
- reflects the original project workflow
- was used during development with a Google Drive / Colab-based setup
- preserves the original environment assumptions from the development phase

This notebook is kept mainly as an archived reference for the original work.

## 2. Non-Drive notebook

**File:** `Final_ML_Project.ipynb`

This notebook is based directly on the original Drive-based notebook, but adapted so it can run without depending on Google Drive.

Characteristics:
- follows the same project logic and overall notebook structure as the original version
- adjusted to work with the repository structure and local paths
- intended for the checker or reviewer, so the notebook can be run without needing the original Drive-based workflow

## Why both notebooks are included

Both notebooks are included intentionally:

- the **Drive-based notebook** preserves the original working version used during development
- the **non-Drive notebook** gives the checker a version of the same notebook that can be run without Google Drive dependency

## Note about results

The repository includes the original saved outputs under `results/`.

Because the non-Drive notebook runs in the refactored repository environment, minor numerical differences may appear compared with the original Drive-based workflow. This can happen because of differences such as:

- package versions
- runtime environment
- solver behavior
- hardware/runtime differences for deep-learning experiments

The saved outputs in the repository should be treated as the main final project outputs.
