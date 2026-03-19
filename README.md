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
- comparison between classical machine-learning models and EEGNet

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
notebooks/  original and non-Drive notebook versions
cache/      local cache outputs (not tracked)
Included outputs

This repository already includes saved outputs generated during the project workflow:

results/experiments/

results/figures/

results/tables/

These are included so the checker can inspect the final outputs directly without having to rerun all experiments.

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

design_to_creativity_eegnet

creativity_to_design_eegnet

Binary cross-dataset transfer

design_to_creativity_rest_ig_eegnet

creativity_to_design_rest_ig_eegnet

Environment setup
Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Data setup

See data/README.md for detailed dataset setup instructions.

Official public-source workflow
python -m scripts.download_data --show-instructions
python -m scripts.download_data --check-only
Optional Google Drive helper

If the shared Google Drive dataset mirror is available, the datasets can also be downloaded automatically with:

python -m scripts.download_data_gdrive

This helper is provided for convenience. The official public dataset sources remain the Mendeley dataset pages listed above.

Main scripts
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

See notebooks/README.md.

Two versions of the notebook are included:

Final_ML_Project_drive.ipynb — the original Drive-based notebook used during development

Final_ML_Project.ipynb — the same notebook adapted to run without Google Drive dependency

The Drive-based notebook is kept for preserving the original development workflow.
The non-Drive notebook is included so the checker can run the notebook without needing the original Google Drive setup.

Reproducibility notes

Raw data is not tracked in git.

Generated local caches are not intended to be tracked in git.

Saved final experiment outputs are included in results/experiments/.

Classical experiments can be rerun through the repo scripts.

Report tables and figures can be regenerated from saved experiment outputs.

EEGNet rerun infrastructure is included through memory-safe HDF5 raw-window caches, but deep-learning experiments remain substantially more resource-intensive and may require long runtimes and stronger hardware.

The original notebook is kept for reference, but the intended repository workflow is through the modular code in src/, the scripts in scripts/, and the non-Drive notebook when needed.

Notes about reproducibility vs original results

The repository includes the original saved project outputs under results/.

The non-Drive notebook and the refactored repo code were adapted from the original project workflow so the checker can run the project without needing Google Drive. Because of differences such as:

package versions

solver behavior

runtime environment

hardware/runtime differences for deep-learning experiments

minor numerical differences may appear between rerun results and the originally saved outputs.

For this reason:

the saved outputs in results/ should be treated as the main final project outputs

the rerun code and non-Drive notebook should be treated as the reproducible checker-facing execution path

Main final outputs

Main final outputs are available in:

results/tables/

results/figures/

These include:

segment-level result summaries

comparison plots

best-model confusion matrices