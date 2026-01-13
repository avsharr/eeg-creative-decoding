# Decoding Creative Cognition & Workload from EEG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository implements a supervised learning framework for decoding higher-order cognitive states (Idea Generation, Idea Evaluation) and estimating mental workload from EEG recordings during open-ended design tasks.

## Scientific Objectives
1.  **Cognitive State Classification**: 3-class identification (IG, IE, RST).
2.  **Workload Regression**: Continuous estimation of NASA-TLX scores.
3.  **Cross-Task Generalization**: Evaluating model robustness by training on Design datasets and testing on Creativity datasets (Zero-Shot).

## Data Source
The project utilizes data from the following Mendeley repositories:
- [Conceptual Design Exploration](https://data.mendeley.com/datasets/h4rf6wzjcr/1)
- [Design Creativity Experiments](https://data.mendeley.com/datasets/24yp3xp58b/1)

## Setup: