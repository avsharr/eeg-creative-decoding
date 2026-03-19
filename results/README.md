# Results

This folder contains saved project outputs generated during the EEG classification workflow.

## Structure

- `experiments/` — saved outputs for individual experiments
- `figures/` — final figures used in the report
- `tables/` — final tables used in the report
- `summary/` — optional summary artifacts

## Experiment folders

Each experiment folder may contain some or all of the following:

- `config.json` — experiment configuration
- `metrics.json` — final evaluation metrics
- `history.json` — training or evaluation history
- `notes.json` — additional experiment notes
- `predictions.npz` — saved predictions / probabilities
- `confusion_matrix.npy` — saved confusion matrix
- `model.joblib` — saved classical model when applicable
- `_DONE` — marker indicating the run completed successfully

## Notes

These saved outputs are included so the project results can be inspected directly without requiring a full rerun.

Raw data, local caches, and temporary artifacts are not tracked in git.