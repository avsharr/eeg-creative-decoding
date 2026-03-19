from __future__ import annotations

EXPERIMENTS = {
    # -------------------------
    # Within-subject: design
    # -------------------------
    "design_within_logreg": {
        "dataset": "design",
        "evaluation": "within",
        "model": "logreg",
        "family": "classical",
    },
    "design_within_linear_svm": {
        "dataset": "design",
        "evaluation": "within",
        "model": "linear_svm",
        "family": "classical",
    },
    "design_within_rbf_svm": {
        "dataset": "design",
        "evaluation": "within",
        "model": "rbf_svm",
        "family": "classical",
    },
    "design_within_eegnet": {
        "dataset": "design",
        "evaluation": "within",
        "model": "eegnet",
        "family": "deep",
    },

    # -------------------------
    # Within-subject: creativity
    # -------------------------
    "creativity_within_logreg": {
        "dataset": "creativity",
        "evaluation": "within",
        "model": "logreg",
        "family": "classical",
    },
    "creativity_within_linear_svm": {
        "dataset": "creativity",
        "evaluation": "within",
        "model": "linear_svm",
        "family": "classical",
    },
    "creativity_within_rbf_svm": {
        "dataset": "creativity",
        "evaluation": "within",
        "model": "rbf_svm",
        "family": "classical",
    },
    "creativity_within_eegnet": {
        "dataset": "creativity",
        "evaluation": "within",
        "model": "eegnet",
        "family": "deep",
    },

    # -------------------------
    # Cross-subject: design
    # -------------------------
    "design_cross_logreg": {
        "dataset": "design",
        "evaluation": "cross_subject",
        "model": "logreg",
        "family": "classical",
    },
    "design_cross_eegnet": {
        "dataset": "design",
        "evaluation": "cross_subject",
        "model": "eegnet",
        "family": "deep",
    },

    # -------------------------
    # Cross-subject: creativity
    # -------------------------
    "creativity_cross_logreg": {
        "dataset": "creativity",
        "evaluation": "cross_subject",
        "model": "logreg",
        "family": "classical",
    },
    "creativity_cross_eegnet": {
        "dataset": "creativity",
        "evaluation": "cross_subject",
        "model": "eegnet",
        "family": "deep",
    },

    # -------------------------
    # Explicit deep transfer experiments
    # -------------------------
    "design_to_creativity_eegnet": {
        "dataset": "design_to_creativity",
        "evaluation": "cross_dataset",
        "model": "eegnet",
        "family": "deep",
    },
    "creativity_to_design_eegnet": {
        "dataset": "creativity_to_design",
        "evaluation": "cross_dataset",
        "model": "eegnet",
        "family": "deep",
    },
    "design_to_creativity_rest_ig_eegnet": {
        "dataset": "design_to_creativity_rest_ig",
        "evaluation": "binary_cross_dataset",
        "model": "eegnet",
        "family": "deep",
    },
    "creativity_to_design_rest_ig_eegnet": {
        "dataset": "creativity_to_design_rest_ig",
        "evaluation": "binary_cross_dataset",
        "model": "eegnet",
        "family": "deep",
    },

    # -------------------------
    # Umbrella labels for grouping/documentation
    # -------------------------
    "cross_dataset": {
        "dataset": "design_to_creativity_and_reverse",
        "evaluation": "cross_dataset",
        "model": "multiple",
        "family": "mixed",
    },
    "binary_cross_dataset": {
        "dataset": "rest_vs_ig_transfer",
        "evaluation": "binary_cross_dataset",
        "model": "multiple",
        "family": "mixed",
    },
}


def get_experiment(experiment_name: str) -> dict:
    if experiment_name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment: {experiment_name}")
    return EXPERIMENTS[experiment_name]


def list_experiments() -> list[str]:
    return list(EXPERIMENTS.keys())


def list_runnable_experiments() -> list[str]:
    return [
        name for name, cfg in EXPERIMENTS.items()
        if cfg["model"] != "multiple"
    ]


def list_experiments_by_family(family: str) -> list[str]:
    return [name for name, cfg in EXPERIMENTS.items() if cfg["family"] == family]


def list_experiments_by_dataset(dataset: str) -> list[str]:
    return [name for name, cfg in EXPERIMENTS.items() if cfg["dataset"] == dataset]


def list_experiments_by_evaluation(evaluation: str) -> list[str]:
    return [name for name, cfg in EXPERIMENTS.items() if cfg["evaluation"] == evaluation]