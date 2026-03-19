# EEG Creative Decoding

Machine-learning final project for EEG-based classification of cognitive states during **design** and **creativity** tasks.

This repository compares **classical machine-learning models** and **EEGNet** on two public EEG datasets, and includes saved experiment outputs, final figures, and final tables used for reporting.

## Project overview

The goal of this project is to classify cognitive states from EEG recordings collected during open-ended design and creativity experiments.

Final target classes used in this project:

- **REST**
- **IG** — idea generation
- **IE** — evaluation-related state used in the final label mapping

The project includes:

- within-subject experiments
- cross-subject experiments
- cross-dataset transfer experiments
- comparison between classical ML models and EEGNet

## Public datasets

This project uses two public EEG datasets from Mendeley Data:

- **Design EEG Dataset**  
  `https://data.mendeley.com/datasets/h4rf6wzjcr/1`

- **Creativity EEG Dataset**  
  `https://data.mendeley.com/datasets/24yp3xp58b/1`

Raw data is **not included** in this repository.

Expected local structure:

```text
data/raw/
├── Design_EEG_Dataset/
└── Creativity_EEG_Dataset/
Repository structure
src/        reusable project code
scripts/    command-line entry points
data/       raw/processed data folders and setup instructions
results/    saved experiment outputs, figures, and tables
notebooks/  archived notebook reference
Included outputs

This repository already includes saved outputs generated during the project workflow:

results/experiments/

results/figures/

results/tables/

These are included so the checker can inspect final outputs directly without rerunning training.

Supported experiments
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

Saved outputs are included for:

design_to_creativity_eegnet

creativity_to_design_eegnet

Binary cross-dataset transfer

Saved outputs are included for:

design_to_creativity_rest_ig_eegnet

creativity_to_design_rest_ig_eegnet

Environment setup
Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Data setup

See data/README.md for the detailed dataset setup instructions.

Basic workflow:

python -m scripts.download_data --show-instructions
python -m scripts.download_data --check-only

After placing the raw .mat files in the expected dataset folders, continue with the cache and experiment workflow.

Basic usage
Validate dataset placement
python -m scripts.download_data --check-only
Build caches
python -m scripts.build_caches --dataset all
Saved outputs

Saved experiment outputs are already available in results/experiments/.

Reproducibility notes

Raw data is not tracked in git.

Generated local caches are not intended to be tracked in git.

Saved final experiment outputs are included in results/experiments/.

Classical pipelines are easier to rerun on standard hardware.

EEGNet and raw-window pipelines are more resource-intensive and may require more RAM and/or GPU support.

The archived notebook is kept for reference, but the intended repo workflow is through modular code and scripts.

Archived notebook

The original working notebook is kept as an archive/reference:

notebooks/Final_ML_Project_archived.ipynb

Main final outputs

Main final outputs are available in:

results/tables/

results/figures/

These include:

segment-level result summaries

comparison plots

best-model confusion matrices


Then make a small `results/README.md` too, since yours was empty:

```md
# Results

This folder contains saved project outputs.

## Structure

- `experiments/` — saved outputs for individual experiments
- `figures/` — final figures used in the report
- `tables/` — final tables used in the report
- `summary/` — optional summary artifacts

## Notes

Experiment folders may contain:
- `config.json`
- `metrics.json`
- `history.json`
- `notes.json`
- `predictions.npz`
- `confusion_matrix.npy`
- `_DONE`

These outputs are included so results can be inspected directly without requiring a full rerun.