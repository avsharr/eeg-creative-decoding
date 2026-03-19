# EEG Creative Decoding

> Machine-learning final project for EEG-based classification of cognitive states during **design** and **creativity** tasks using both **classical machine learning** and **EEGNet**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Bar-Ilan University](https://img.shields.io/badge/Bar--Ilan%20University-ML%20for%20Neuroscience-green)](https://www.biu.ac.il)

---

## Quick Start

1. Create and activate a virtual environment
2. Install dependencies
3. Check dataset placement
4. Run an experiment or inspect the saved results
5. Regenerate figures/tables if needed

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.download_data --check-only
python -m scripts.run_experiment --experiment design_within_logreg --force
python -m scripts.generate_report_artifacts

## Overview

This repository contains our final project for **Machine Learning for Neuroscience** at **Bar-Ilan University**. The goal is to classify cognitive states from EEG recordings collected during open-ended **design** and **creativity** tasks.

We compare **classical machine-learning models** and a deep-learning model (**EEGNet**) across two public EEG datasets, and include the saved experiment outputs, final figures, and final tables used in the final report.

### Final target classes

- **REST**
- **IG** — idea generation
- **IE** — evaluation-related state used in the final label mapping

### Main research questions

- Can EEG reliably distinguish **idea generation**, **evaluation-related processing**, and **rest**?
- How well do models generalize across participants (**cross-subject / LOSO**)?
- Can a model trained on one dataset transfer to another (**cross-dataset transfer**)?
- How do **classical feature-based models** compare with **EEGNet** on the same problem?

---

## Datasets

This project uses two public EEG datasets from **Mendeley Data**.

| Dataset | Subjects | Classes used in this project | Sampling Rate | Source |
|---|---:|---|---:|---|
| **Design EEG Dataset** | 27 | IG / IE / REST | 250 Hz | [Mendeley](https://data.mendeley.com/datasets/h4rf6wzjcr/1) |
| **Creativity EEG Dataset** | 28 | IG / IE / REST | 250 Hz in project workflow | [Mendeley](https://data.mendeley.com/datasets/24yp3xp58b/1) |

### Notes

- Raw data is **not included** in this repository.
- The creativity dataset originally comes from a higher sampling rate source and is handled in the project preprocessing workflow.
- Local raw data is expected under:

```text
data/raw/
├── Design_EEG_Dataset/
└── Creativity_EEG_Dataset/
Project Scope

The repository includes:

within-subject experiments

cross-subject experiments

cross-dataset transfer experiments

comparison between classical machine-learning models and EEGNet

saved outputs used for the final report

Models
Classical machine-learning models

Logistic Regression

Linear SVM

RBF SVM

Deep-learning model

EEGNet

Feature representation for classical models

Classical models use EEG band-power features extracted from 4 canonical frequency bands:

Band	Frequency Range
Delta	1–4 Hz
Theta	4–8 Hz
Alpha	8–13 Hz
Beta	13–30 Hz

With 63 EEG channels and 4 bands, this gives:

252 features per window

Supported Experiments
Within-subject

design_within_logreg

design_within_linear_svm

design_within_rbf_svm

design_within_eegnet

creativity_within_logreg

creativity_within_linear_svm

creativity_within_rbf_svm

creativity_within_eegnet

Cross-subject

design_cross_logreg

design_cross_eegnet

creativity_cross_logreg

creativity_cross_eegnet

Cross-dataset transfer

design_to_creativity_eegnet

creativity_to_design_eegnet

Binary cross-dataset transfer

design_to_creativity_rest_ig_eegnet

creativity_to_design_rest_ig_eegnet

Repository Structure
eeg-creative-decoding/
├── src/                        # Reusable project code
├── scripts/                    # Command-line entry points
├── notebooks/                  # Final notebooks and notebook notes
├── data/                       # Data setup instructions
├── results/                    # Saved experiment outputs, figures, and tables
├── cache/                      # Local caches (not tracked by git)
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
Main folders

src/ — modular project code

scripts/ — reproducible repo workflow

notebooks/ — archived notebook workflow

results/experiments/ — saved experiment outputs

results/figures/ — final figures

results/tables/ — final tables

Included Outputs

This repository already includes saved outputs generated during the project workflow:

results/experiments/

results/figures/

results/tables/

These are included so the checker can inspect the final outputs directly without having to rerun all experiments from scratch.

Main final outputs

Main report-ready outputs are available in:

results/tables/

results/figures/

These include:

segment-level result summaries

comparison plots

best-model confusion matrices

Environment Setup
Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Data Setup

See data/README.md for detailed dataset setup instructions.

Official public-source workflow
python -m scripts.download_data --show-instructions
python -m scripts.download_data --check-only
Optional Google Drive helper

If the shared Google Drive dataset mirror is available, the datasets can also be downloaded automatically with:

python -m scripts.download_data_gdrive

This helper is provided for convenience. The official public dataset sources remain the Mendeley dataset pages listed above.

Main Scripts
Validate dataset placement
python -m scripts.download_data --check-only
Build classical bandpower caches
python -m scripts.build_caches --dataset all
Build memory-safe raw-window HDF5 caches for EEGNet
python -m scripts.build_raw_windows_h5 --dataset all
Run one experiment
python -m scripts.run_experiment --experiment design_within_logreg --force
Regenerate report artifacts
python -m scripts.generate_report_artifacts
Notebooks

See notebooks/README.md for notebook-specific details.

Two archived notebook versions are included:

Final_ML_Project_archived_drive.ipynb — the original Drive-based notebook used during development

Final_ML_Project_archived.ipynb — the adapted version that can be used without the original Google Drive dependency

Why both notebooks are included

The Drive-based notebook is kept to preserve the original development workflow.
The non-Drive archived notebook is included so the checker can inspect and run the workflow without depending on the original Google Drive setup.

Reproducibility Notes

Raw data is not tracked in git

Generated local caches are not intended to be tracked

Saved final experiment outputs are included in results/experiments/

Classical experiments can be rerun through the repo scripts

Report figures and tables can be regenerated from saved experiment outputs

EEGNet reruns are supported through memory-safe raw-window caching, but deep-learning experiments remain more resource-intensive and may require long runtimes and stronger hardware

The original notebook workflow is preserved for reference, but the intended repository workflow is through:

the modular code in src/

the scripts in scripts/

the archived non-Drive notebook when notebook inspection is needed

Notes About Reproducibility vs. Original Results

The repository includes the original saved project outputs under results/.

The archived non-Drive notebook and the refactored repo code were adapted from the original project workflow so the checker can run the project without relying on Google Drive. Because of differences such as:

package versions

solver behavior

runtime environment

hardware/runtime differences for deep-learning experiments

minor numerical differences may appear between rerun results and the originally saved outputs.

Therefore

the saved outputs in results/ should be treated as the main final project outputs

the rerun code and archived non-Drive notebook should be treated as the checker-facing reproducible execution path

Evaluation Notes

Cross-subject experiments are based on Leave-One-Subject-Out (LOSO) evaluation

Performance is primarily reported using balanced accuracy

Segment-level summaries and confusion matrices are included in the final outputs

Team
Name	Role
Dilan Efe	ML pipeline, experiment design, classical models
Anastasiia Sharova	preprocessing, feature engineering
Orwa Yassin	EEGNet / deep learning, repository setup

Bar-Ilan University — Machine Learning for Neuroscience (2025/26)

References

Lawhern et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces.

Barachant et al. (2012). Multiclass Brain–Computer Interface Classification by Riemannian Geometry.

Design EEG Dataset — Mendeley Data

Creativity EEG Dataset — Mendeley Data
