from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as mpl_cm

from src.paths import EXPERIMENTS_DIR, FIGURES_DIR, SUMMARY_DIR, TABLES_DIR, ensure_project_dirs


CLASS_LABELS_3 = ["REST", "IG", "IE"]
CLASS_LABELS_2 = ["RST", "IG"]


# ------------------------------------------------------------
# Basic file readers
# ------------------------------------------------------------
def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_json(path: Path):
    path = Path(path)
    if path.exists():
        return read_json(path)
    return None


def safe_read_npy(path: Path):
    path = Path(path)
    if path.exists():
        return np.load(path, allow_pickle=True)
    return None


def safe_read_npz(path: Path):
    path = Path(path)
    if path.exists():
        return np.load(path, allow_pickle=True)
    return None


# ------------------------------------------------------------
# Experiment loading
# ------------------------------------------------------------
def load_experiment_result(experiment_name: str) -> dict:
    exp_dir = EXPERIMENTS_DIR / experiment_name

    return {
        "experiment_name": experiment_name,
        "dir": exp_dir,
        "config": safe_read_json(exp_dir / "config.json"),
        "metrics": safe_read_json(exp_dir / "metrics.json"),
        "notes": safe_read_json(exp_dir / "notes.json"),
        "history": safe_read_json(exp_dir / "history.json"),
        "confusion_matrix": safe_read_npy(exp_dir / "confusion_matrix.npy"),
        "predictions": safe_read_npz(exp_dir / "predictions.npz"),
        "done": (exp_dir / "_DONE").exists(),
    }


# ------------------------------------------------------------
# Name inference helpers
# ------------------------------------------------------------
def infer_dataset_from_name(exp_name: str) -> str:
    if "creativity" in exp_name:
        return "Creativity"
    if "design" in exp_name:
        return "Design"
    return "Mixed"


def infer_setting_from_name(exp_name: str) -> str:
    if "within" in exp_name:
        return "Within-subject"
    if "cross" in exp_name and "_to_" not in exp_name and "cross_dataset" not in exp_name:
        return "Cross-subject"
    if "_to_" in exp_name:
        return "Cross-dataset"
    return "Other"


def infer_model_from_name(exp_name: str) -> str:
    if "linear_svm" in exp_name:
        return "Linear SVM"
    if "rbf_svm" in exp_name:
        return "RBF SVM"
    if "logreg" in exp_name:
        return "LogReg"
    if "eegnet" in exp_name:
        return "EEGNet"
    return "Unknown"


def infer_task_from_name(exp_name: str) -> str:
    if "rest_ig" in exp_name:
        return "REST vs IG"
    return "3-class"


# ------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------
def round_or_nan(x, ndigits: int = 4):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return round(float(x), ndigits)
    except Exception:
        return np.nan


def metric_seg_bal_acc(metrics: dict):
    return metrics.get("report_balanced_accuracy", metrics.get("segment_balanced_accuracy", np.nan))


def metric_seg_acc(metrics: dict):
    return metrics.get("report_accuracy", metrics.get("segment_accuracy", np.nan))


def metric_seg_f1(metrics: dict):
    return metrics.get("report_macro_f1", metrics.get("segment_macro_f1", np.nan))


def get_n_folds(result: dict):
    history = result.get("history") or {}
    if "n_successful_folds" in history:
        return history["n_successful_folds"]
    if "fold_rows" in history and isinstance(history["fold_rows"], list):
        return len(history["fold_rows"])
    return np.nan


def get_segment_bal_acc_std(result: dict):
    history = result.get("history") or {}
    fold_rows = history.get("fold_rows", [])
    if not fold_rows:
        return np.nan

    vals = []
    for row in fold_rows:
        if "segment_balanced_accuracy" in row:
            vals.append(float(row["segment_balanced_accuracy"]))
        elif "best_val_segment_balanced_accuracy" in row:
            vals.append(float(row["best_val_segment_balanced_accuracy"]))

    if len(vals) <= 1:
        return 0.0 if len(vals) == 1 else np.nan

    return float(np.std(vals, ddof=1))


# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------
MODEL_TO_CMAP = {
    "LogReg": "Blues",
    "Linear SVM": "Greens",
    "RBF SVM": "Oranges",
    "EEGNet": "Purples",
    "Unknown": "Greys",
}


def get_bar_color_for_model(model_name: str):
    cmap_name = MODEL_TO_CMAP.get(model_name, "Greys")
    cmap = mpl_cm.get_cmap(cmap_name)
    return cmap(0.65)


def get_confusion_cmap_for_model(model_name: str):
    cmap_name = MODEL_TO_CMAP.get(model_name, "Greys")
    return mpl_cm.get_cmap(cmap_name)


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def plot_pretty_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    std_col: str | None,
    title: str,
    fig_path: Path,
    rotate: int = 0,
):
    if len(df) == 0:
        return

    ensure_project_dirs()

    categories = df[category_col].tolist()
    values = df[value_col].astype(float).tolist()
    stds = df[std_col].astype(float).tolist() if std_col and std_col in df.columns else None
    colors = [get_bar_color_for_model(m) for m in df["Model"].tolist()] if "Model" in df.columns else None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(categories))

    ax.bar(x, values, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_with_counts_and_row_percents(
    cm: np.ndarray,
    class_labels: list[str],
    title: str,
    fig_path: Path,
    cmap,
):
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_perc = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(row_perc, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = row_perc[i, j] * 100
            count = int(cm[i, j])
            text = f"{count}\n{pct:.1f}%"
            color = "white" if row_perc[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main report builders
# ------------------------------------------------------------
REPORT_EXPERIMENTS = [
    "design_within_logreg",
    "design_within_linear_svm",
    "design_within_rbf_svm",
    "creativity_within_logreg",
    "creativity_within_linear_svm",
    "creativity_within_rbf_svm",
    "design_cross_logreg",
    "creativity_cross_logreg",
    "design_within_eegnet",
    "creativity_within_eegnet",
    "design_cross_eegnet",
    "creativity_cross_eegnet",
    "design_to_creativity_eegnet",
    "creativity_to_design_eegnet",
    "design_to_creativity_rest_ig_eegnet",
    "creativity_to_design_rest_ig_eegnet",
]


def build_unified_report_dataframe(loaded_results: dict[str, dict]) -> pd.DataFrame:
    report_rows = []

    for exp, result in loaded_results.items():
        metrics = result["metrics"]
        if metrics is None:
            continue

        row = {
            "Experiment": exp,
            "Dataset": infer_dataset_from_name(exp),
            "Setting": infer_setting_from_name(exp),
            "Model": infer_model_from_name(exp),
            "Task": infer_task_from_name(exp),
            "Segment Balanced Accuracy": round_or_nan(metric_seg_bal_acc(metrics)),
            "Segment Accuracy": round_or_nan(metric_seg_acc(metrics)),
            "Segment Macro F1": round_or_nan(metric_seg_f1(metrics)),
            "STD": round_or_nan(get_segment_bal_acc_std(result)),
            "n_folds": get_n_folds(result),
            "n_test_segments": metrics.get("n_test_segments", np.nan),
            "n_test_subjects": metrics.get("n_test_subjects", np.nan),
        }
        report_rows.append(row)

    report_all_df = pd.DataFrame(report_rows)

    if len(report_all_df) > 0:
        report_all_df = report_all_df.sort_values(
            by=["Setting", "Dataset", "Segment Balanced Accuracy"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    return report_all_df


def generate_report_artifacts():
    ensure_project_dirs()

    loaded_results = {}
    missing_results = []

    for exp in REPORT_EXPERIMENTS:
        res = load_experiment_result(exp)
        if res["metrics"] is None:
            missing_results.append(exp)
        else:
            loaded_results[exp] = res

    report_all_df = build_unified_report_dataframe(loaded_results)
    report_all_df.to_csv(SUMMARY_DIR / "report_all_results_segment_level.csv", index=False)
    report_all_df.to_csv(TABLES_DIR / "report_all_results_segment_level.csv", index=False)

    within_experiments = [
        "design_within_logreg",
        "design_within_linear_svm",
        "design_within_rbf_svm",
        "design_within_eegnet",
        "creativity_within_logreg",
        "creativity_within_linear_svm",
        "creativity_within_rbf_svm",
        "creativity_within_eegnet",
    ]
    table_within_df = (
        report_all_df[report_all_df["Experiment"].isin(within_experiments)]
        .copy()[["Dataset", "Model", "Segment Balanced Accuracy", "STD", "Segment Accuracy", "Segment Macro F1",
                 "n_folds", "n_test_segments", "n_test_subjects", "Experiment"]]
    )
    if len(table_within_df) > 0:
        table_within_df = table_within_df.sort_values(
            by=["Dataset", "Segment Balanced Accuracy"], ascending=[True, False]
        ).reset_index(drop=True)
    table_within_df.to_csv(TABLES_DIR / "table_within_subject_segment.csv", index=False)

    cross_experiments = [
        "design_cross_logreg",
        "creativity_cross_logreg",
        "design_cross_eegnet",
        "creativity_cross_eegnet",
    ]
    table_cross_df = (
        report_all_df[report_all_df["Experiment"].isin(cross_experiments)]
        .copy()[["Dataset", "Model", "Segment Balanced Accuracy", "STD", "Segment Accuracy", "Segment Macro F1",
                 "n_folds", "n_test_segments", "n_test_subjects", "Experiment"]]
    )
    if len(table_cross_df) > 0:
        table_cross_df = table_cross_df.sort_values(
            by=["Dataset", "Segment Balanced Accuracy"], ascending=[True, False]
        ).reset_index(drop=True)
    table_cross_df.to_csv(TABLES_DIR / "table_cross_subject_segment.csv", index=False)

    transfer_experiments = [
        "design_to_creativity_eegnet",
        "creativity_to_design_eegnet",
        "design_to_creativity_rest_ig_eegnet",
        "creativity_to_design_rest_ig_eegnet",
    ]
    table_transfer_df = report_all_df[report_all_df["Experiment"].isin(transfer_experiments)].copy()
    if len(table_transfer_df) > 0:
        table_transfer_df["Train Dataset"] = table_transfer_df["Experiment"].map(
            lambda x: "Design" if x.startswith("design_to_") else "Creativity"
        )
        table_transfer_df["Test Dataset"] = table_transfer_df["Experiment"].map(
            lambda x: "Creativity" if "_to_creativity" in x else "Design"
        )
        table_transfer_df = table_transfer_df[
            ["Train Dataset", "Test Dataset", "Task", "Model", "Segment Balanced Accuracy", "STD",
             "Segment Accuracy", "Segment Macro F1", "n_folds", "n_test_segments", "n_test_subjects", "Experiment"]
        ].sort_values(by=["Train Dataset", "Test Dataset", "Task"]).reset_index(drop=True)
    table_transfer_df.to_csv(TABLES_DIR / "table_cross_dataset_transfer_segment.csv", index=False)

    # bar plots
    if len(table_within_df) > 0:
        for dataset_name in sorted(table_within_df["Dataset"].unique()):
            sub = table_within_df[table_within_df["Dataset"] == dataset_name].copy()
            plot_pretty_bar_chart(
                df=sub,
                category_col="Model",
                value_col="Segment Balanced Accuracy",
                std_col="STD",
                title=f"Within-subject performance - {dataset_name}",
                fig_path=FIGURES_DIR / f"within_{dataset_name.lower()}_segment_bal_acc_pretty.png",
                rotate=15,
            )

    if len(table_cross_df) > 0:
        for dataset_name in sorted(table_cross_df["Dataset"].unique()):
            sub = table_cross_df[table_cross_df["Dataset"] == dataset_name].copy()
            plot_pretty_bar_chart(
                df=sub,
                category_col="Model",
                value_col="Segment Balanced Accuracy",
                std_col="STD",
                title=f"Cross-subject performance - {dataset_name}",
                fig_path=FIGURES_DIR / f"cross_{dataset_name.lower()}_segment_bal_acc_pretty.png",
                rotate=15,
            )

    if len(table_transfer_df) > 0:
        plot_df = table_transfer_df.copy()
        plot_df["Label"] = (
            plot_df["Train Dataset"] + " → " + plot_df["Test Dataset"] + " (" + plot_df["Task"] + ")"
        )
        plot_pretty_bar_chart(
            df=plot_df,
            category_col="Label",
            value_col="Segment Balanced Accuracy",
            std_col="STD",
            title="Cross-dataset transfer performance",
            fig_path=FIGURES_DIR / "cross_dataset_transfer_segment_bal_acc_pretty.png",
            rotate=18,
        )

    # best confusion matrices
    for setting_name, table_df, prefix in [
        ("within", table_within_df, "within_subject"),
        ("cross", table_cross_df, "cross_subject"),
    ]:
        if len(table_df) == 0:
            continue
        for dataset_name in sorted(table_df["Dataset"].unique()):
            sub = table_df[table_df["Dataset"] == dataset_name].copy().sort_values(
                "Segment Balanced Accuracy", ascending=False
            )
            best = sub.iloc[0]
            exp = best["Experiment"]
            result = loaded_results[exp]
            cm = result["confusion_matrix"]
            if cm is None:
                continue

            model_name = infer_model_from_name(exp)
            plot_confusion_matrix_with_counts_and_row_percents(
                cm=cm,
                class_labels=CLASS_LABELS_3,
                title=f"{dataset_name} {setting_name.replace('_', ' ')} best: {model_name}",
                fig_path=FIGURES_DIR / f"{dataset_name.lower()}_{prefix}_best_confusion_matrix_{exp}.png",
                cmap=get_confusion_cmap_for_model(model_name),
            )
            pd.DataFrame(cm, index=CLASS_LABELS_3, columns=CLASS_LABELS_3).to_csv(
                TABLES_DIR / f"{dataset_name.lower()}_{prefix}_best_confusion_matrix_{exp}.csv"
            )

    return {
        "loaded_experiments": sorted(loaded_results.keys()),
        "missing_experiments": missing_results,
        "report_all_path": str(SUMMARY_DIR / "report_all_results_segment_level.csv"),
    }