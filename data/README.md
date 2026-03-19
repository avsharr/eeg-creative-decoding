# Data setup

This repository does not include raw EEG data.

Public dataset sources:
- Design dataset: https://data.mendeley.com/datasets/h4rf6wzjcr/1
- Creativity dataset: https://data.mendeley.com/datasets/24yp3xp58b/1

Expected local layout:

data/raw/
├── Design_EEG_Dataset/
└── Creativity_EEG_Dataset/

Quick setup:
1. Download the public datasets from the Mendeley pages above.
2. Either:
   - extract them directly into `data/raw/...`, or
   - place the downloaded zip files into `data/_downloads/`
3. Run:
   - `python -m scripts.download_data --extract-only`
   - `python -m scripts.download_data --check-only`
   - `python -m scripts.build_caches --dataset all`