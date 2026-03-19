from __future__ import annotations

import json
import time
import warnings
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import INT_TO_FINAL_LABEL
from src.eeg_utils import load_cache_with_metadata


CLASS_LABELS_INT = [0, 1, 2]
CLASS_LABELS_NAME = [INT_TO_FINAL_LABEL[i] for i in CLASS_LABELS_INT]


# ------------------------------------------------------------
# Cache selectors
# ------------------------------------------------------------
def load_design_bandpower_cache():
    data, meta = load_cache_with_metadata("design_bandpower")
    return {
        "X_bandpower": data["X_bandpower"],
        "y": data["y"],
        "subject_ids": data["subject_ids"],
        "segment_ids": data["segment_ids"],
        "task_names": data["task_names"] if "task_names" in data.files else None,
        "meta": meta,
    }


def load_creativity_bandpower_cache():
    data, meta = load_cache_with_metadata("creativity_bandpower")
    return {
        "X_bandpower": data["X_bandpower"],
        "y": data["y"],
        "subject_ids": data["subject_ids"],
        "segment_ids": data["segment_ids"],
        "task_names": data["task_names"] if "task_names" in data.files else None,
        "meta": meta,
    }


def get_bandpower_dataset_by_name(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "design":
        return load_design_bandpower_cache()
    if dataset_name == "creativity":
        return load_creativity_bandpower_cache()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_metrics_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=CLASS_LABELS_INT,
            zero_division=0,
        )

    return {
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


def compute_confusion_matrix_fixed(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return confusion_matrix(y_true, y_pred, labels=CLASS_LABELS_INT)


# ------------------------------------------------------------
# Classical model builders
# ------------------------------------------------------------
def normalize_model_type(model_type: str) -> str:
    mapping = {
        "logreg": "logreg",
        "linear_svm": "svm_linear",
        "rbf_svm": "svm_rbf",
        "svm_linear": "svm_linear",
        "svm_rbf": "svm_rbf",
    }
    if model_type not in mapping:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return mapping[model_type]


def get_classical_pipeline_and_param_grid(model_type: str, evaluation_type: str = "within_subject"):
    model_type = normalize_model_type(model_type)

    if evaluation_type == "within_subject":
        rbf_grid = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001, 0.0001]}
        linear_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
        logreg_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    else:
        rbf_grid = {"clf__C": [1, 10], "clf__gamma": ["scale"]}
        linear_grid = {"clf__C": [0.1, 1, 10]}
        logreg_grid = {"clf__C": [0.1, 1, 10]}

    if model_type == "svm_rbf":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, cache_size=1000)),
            ]
        )
        param_grid = rbf_grid

    elif model_type == "svm_linear":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", class_weight="balanced", probability=True, cache_size=1000)),
            ]
        )
        param_grid = linear_grid

    elif model_type == "logreg":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        param_grid = logreg_grid

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return pipeline, param_grid


# ------------------------------------------------------------
# Helper functions copied from notebook logic
# ------------------------------------------------------------
def choose_representative_params(fold_rows):
    serialized = []
    for row in fold_rows:
        bp = row.get("best_params", {})
        serialized.append(json.dumps(bp, sort_keys=True))

    if len(serialized) == 0:
        return {}

    most_common_serialized = Counter(serialized).most_common(1)[0][0]
    return json.loads(most_common_serialized)


def fit_final_classical_model_no_search(
    model_type: str,
    X,
    y,
    chosen_params: dict,
    evaluation_type: str,
):
    pipeline, _ = get_classical_pipeline_and_param_grid(
        model_type=model_type,
        evaluation_type=evaluation_type,
    )
    if chosen_params:
        pipeline.set_params(**chosen_params)
    pipeline.fit(X, y)
    return pipeline


def _majority_label(arr):
    arr = np.asarray(arr).astype(int)
    values, counts = np.unique(arr, return_counts=True)
    return int(values[np.argmax(counts)])


def _softmax_numpy(x):
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = np.column_stack([-x, x])

    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def extract_model_classes(estimator):
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_).astype(int)

    if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
        clf = estimator.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return np.asarray(clf.classes_).astype(int)

    raise AttributeError("Could not extract classes_ from estimator.")


def align_probs_to_full_classes(probs, model_classes, full_classes=CLASS_LABELS_INT):
    probs = np.asarray(probs, dtype=float)
    model_classes = np.asarray(model_classes).astype(int)
    full_classes = np.asarray(full_classes).astype(int)

    aligned = np.zeros((probs.shape[0], len(full_classes)), dtype=float)

    class_to_col = {c: i for i, c in enumerate(model_classes)}
    for j, c in enumerate(full_classes):
        if c in class_to_col:
            aligned[:, j] = probs[:, class_to_col[c]]

    row_sums = aligned.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze() > 0
    aligned[nonzero] = aligned[nonzero] / row_sums[nonzero]

    return aligned


def get_window_level_outputs_from_model(model, X):
    model_classes = extract_model_classes(model)

    if hasattr(model, "predict_proba"):
        probs_raw = model.predict_proba(X)
        probs = align_probs_to_full_classes(probs_raw, model_classes)
        y_pred = np.asarray(CLASS_LABELS_INT)[np.argmax(probs, axis=1)]
        return y_pred.astype(int), probs

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probs_raw = _softmax_numpy(scores)
        probs = align_probs_to_full_classes(probs_raw, model_classes)
        y_pred = np.asarray(CLASS_LABELS_INT)[np.argmax(probs, axis=1)]
        return y_pred.astype(int), probs

    y_pred = model.predict(X).astype(int)
    probs = np.zeros((len(y_pred), len(CLASS_LABELS_INT)), dtype=float)
    for i, label in enumerate(y_pred):
        probs[i, CLASS_LABELS_INT.index(int(label))] = 1.0
    return y_pred, probs


def aggregate_segment_predictions_from_probs(segment_ids, y_true, probs):
    segment_ids = np.asarray(segment_ids)
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)

    unique_segments = np.unique(segment_ids)

    seg_true = []
    seg_pred = []
    seg_probs_all = []

    for seg in unique_segments:
        mask = segment_ids == seg

        seg_probs = probs[mask].mean(axis=0)
        if seg_probs.sum() > 0:
            seg_probs = seg_probs / seg_probs.sum()

        seg_pred_label = int(CLASS_LABELS_INT[int(np.argmax(seg_probs))])
        seg_true_label = _majority_label(y_true[mask])

        seg_true.append(seg_true_label)
        seg_pred.append(seg_pred_label)
        seg_probs_all.append(seg_probs)

    seg_true = np.asarray(seg_true).astype(int)
    seg_pred = np.asarray(seg_pred).astype(int)
    seg_probs_all = np.asarray(seg_probs_all, dtype=float)

    return {
        "segment_ids_unique": unique_segments,
        "segment_y_true": seg_true,
        "segment_y_pred": seg_pred,
        "segment_probs": seg_probs_all,
        "segment_metrics": compute_metrics_dict(seg_true, seg_pred),
        "segment_confusion_matrix": compute_confusion_matrix_fixed(seg_true, seg_pred),
    }


def safe_group_cv_splitter(n_splits, y, groups, random_state=42):
    try:
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        _ = list(cv.split(np.zeros(len(y)), y, groups))
        return cv
    except Exception:
        return GroupKFold(n_splits=n_splits)


# ------------------------------------------------------------
# Classical experiment runners
# ------------------------------------------------------------
def run_within_subject_classical_experiment(
    dataset_name: str,
    model_type: str,
    inner_n_splits: int = 5,
    scoring: str = "balanced_accuracy",
    fit_final_model: bool = True,
):
    ds = get_bandpower_dataset_by_name(dataset_name)

    X_all = ds["X_bandpower"]
    y_all = ds["y"]
    subject_ids_all = ds["subject_ids"]
    segment_ids_all = ds["segment_ids"]

    all_window_y_true = []
    all_window_y_pred = []
    all_window_probs = []
    all_window_segment_ids = []
    fold_rows = []

    subject_list = sorted(np.unique(subject_ids_all).tolist())
    global_start = time.time()

    for subject_pos, subject_id in enumerate(subject_list, start=1):
        subject_mask = subject_ids_all == subject_id

        X_sub = X_all[subject_mask]
        y_sub = y_all[subject_mask]
        seg_sub = segment_ids_all[subject_mask]

        unique_segments = np.unique(seg_sub)
        n_groups = len(unique_segments)

        if n_groups < 2:
            print(f"[SKIP SUBJECT] {subject_id} has < 2 groups")
            continue

        outer_n_splits = min(5, n_groups)
        outer_cv = safe_group_cv_splitter(
            n_splits=outer_n_splits,
            y=y_sub,
            groups=seg_sub,
            random_state=42 + subject_pos,
        )

        for outer_fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X_sub, y_sub, groups=seg_sub),
            start=1,
        ):
            X_train = X_sub[train_idx]
            y_train = y_sub[train_idx]
            seg_train = seg_sub[train_idx]

            X_test = X_sub[test_idx]
            y_test = y_sub[test_idx]
            seg_test = seg_sub[test_idx]

            n_inner_groups = len(np.unique(seg_train))
            inner_splits = min(inner_n_splits, n_inner_groups)

            if inner_splits < 2:
                print(f"[SKIP FOLD] {subject_id} outer_fold={outer_fold_idx} has < 2 inner groups")
                continue

            pipeline, param_grid = get_classical_pipeline_and_param_grid(
                model_type=model_type,
                evaluation_type="within_subject",
            )

            inner_cv = safe_group_cv_splitter(
                n_splits=inner_splits,
                y=y_train,
                groups=seg_train,
                random_state=100 + outer_fold_idx,
            )

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=1,
                refit=True,
                verbose=0,
            )

            fold_start = time.time()
            grid.fit(X_train, y_train, groups=seg_train)

            y_pred, probs = get_window_level_outputs_from_model(grid.best_estimator_, X_test)
            fold_elapsed = time.time() - fold_start

            fold_window_metrics = compute_metrics_dict(y_test, y_pred)

            seg_out = aggregate_segment_predictions_from_probs(
                segment_ids=seg_test,
                y_true=y_test,
                probs=probs,
            )
            fold_segment_metrics = seg_out["segment_metrics"]

            all_window_y_true.append(y_test)
            all_window_y_pred.append(y_pred)
            all_window_probs.append(probs)
            all_window_segment_ids.append(seg_test)

            fold_rows.append(
                {
                    "subject_id": str(subject_id),
                    "outer_fold": int(outer_fold_idx),
                    "window_balanced_accuracy": float(fold_window_metrics["balanced_accuracy"]),
                    "window_accuracy": float(fold_window_metrics["accuracy"]),
                    "window_macro_f1": float(fold_window_metrics["macro_f1"]),
                    "segment_balanced_accuracy": float(fold_segment_metrics["balanced_accuracy"]),
                    "segment_accuracy": float(fold_segment_metrics["accuracy"]),
                    "segment_macro_f1": float(fold_segment_metrics["macro_f1"]),
                    "n_train_windows": int(len(train_idx)),
                    "n_test_windows": int(len(test_idx)),
                    "n_train_segments": int(len(np.unique(seg_train))),
                    "n_test_segments": int(len(np.unique(seg_test))),
                    "best_params": grid.best_params_,
                    "fold_time_sec": round(fold_elapsed, 3),
                }
            )

    if len(all_window_y_true) == 0:
        raise RuntimeError(f"No successful folds were run for dataset={dataset_name}, model_type={model_type}")

    y_true_all = np.concatenate(all_window_y_true).astype(int)
    y_pred_all = np.concatenate(all_window_y_pred).astype(int)
    probs_all = np.concatenate(all_window_probs)
    segment_ids_eval = np.concatenate(all_window_segment_ids)

    window_metrics = compute_metrics_dict(y_true_all, y_pred_all)

    segment_out = aggregate_segment_predictions_from_probs(
        segment_ids=segment_ids_eval,
        y_true=y_true_all,
        probs=probs_all,
    )
    segment_metrics = segment_out["segment_metrics"]
    cm_segment = segment_out["segment_confusion_matrix"]

    total_elapsed = time.time() - global_start
    chosen_params = choose_representative_params(fold_rows)

    best_final_model = None
    final_model_status = "not_fitted"

    if fit_final_model:
        try:
            model_fit_start = time.time()
            best_final_model = fit_final_classical_model_no_search(
                model_type=model_type,
                X=X_all,
                y=y_all,
                chosen_params=chosen_params,
                evaluation_type="within_subject",
            )
            final_model_status = f"fitted_without_grid_search in {time.time() - model_fit_start:.2f} sec"
        except Exception as e:
            print("[WARN] Could not fit final model without grid search:", e)
            final_model_status = f"fit_failed: {e}"

    history = {"fold_rows": fold_rows, "n_successful_folds": len(fold_rows)}

    notes = {
        "evaluation_type": "within_subject",
        "dataset_name": dataset_name,
        "model_type": model_type,
        "scoring": scoring,
        "n_subjects_total": int(len(subject_list)),
        "train_time_sec": round(total_elapsed, 3),
        "final_model_status": final_model_status,
        "report_metric_level": "segment",
        "aggregation_method": "mean_probability_per_segment",
    }

    config = {
        "dataset_name": dataset_name,
        "model_type": model_type,
        "input_type": "bandpower",
        "evaluation_type": "within_subject",
        "cv_type": "StratifiedGroupKFold_or_GroupKFold_fallback",
        "group_variable": "segment_id",
        "selection_metric": "balanced_accuracy",
        "inner_n_splits": inner_n_splits,
        "scoring": scoring,
        "feature_dim": int(X_all.shape[1]),
        "n_samples_total": int(X_all.shape[0]),
        "final_model_strategy": "most_frequent_fold_best_params_no_full_grid_search",
        "chosen_params": chosen_params,
        "probability_output": True,
        "aggregation_method": "mean_probability_per_segment",
        "report_metric_level": "segment",
    }

    metrics = {
        "balanced_accuracy": float(window_metrics["balanced_accuracy"]),
        "accuracy": float(window_metrics["accuracy"]),
        "macro_f1": float(window_metrics["macro_f1"]),
        "segment_balanced_accuracy": float(segment_metrics["balanced_accuracy"]),
        "segment_accuracy": float(segment_metrics["accuracy"]),
        "segment_macro_f1": float(segment_metrics["macro_f1"]),
        "report_balanced_accuracy": float(segment_metrics["balanced_accuracy"]),
        "report_accuracy": float(segment_metrics["accuracy"]),
        "report_macro_f1": float(segment_metrics["macro_f1"]),
        "n_train": int(X_all.shape[0]),
        "n_test": int(len(y_true_all)),
        "n_test_segments": int(len(segment_out["segment_y_true"])),
        "n_test_subjects": int(len(subject_list)),
        "best_params": chosen_params,
        "train_time_sec": round(total_elapsed, 3),
    }

    return {
        "config": config,
        "metrics": metrics,
        "notes": notes,
        "history": history,
        "confusion_matrix_array": cm_segment,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "sklearn_model": best_final_model,
        "extra_prediction_arrays": {
            "class_labels": np.array(CLASS_LABELS_NAME, dtype=object),
            "probs": probs_all,
            "segment_ids": segment_ids_eval,
            "segment_y_true": segment_out["segment_y_true"],
            "segment_y_pred": segment_out["segment_y_pred"],
            "segment_probs": segment_out["segment_probs"],
            "segment_ids_unique": segment_out["segment_ids_unique"],
        },
    }


def run_cross_subject_classical_experiment(
    dataset_name: str,
    model_type: str,
    inner_n_splits: int = 3,
    scoring: str = "balanced_accuracy",
    fit_final_model: bool = True,
):
    ds = get_bandpower_dataset_by_name(dataset_name)

    X_all = ds["X_bandpower"]
    y_all = ds["y"]
    subject_ids_all = ds["subject_ids"]
    segment_ids_all = ds["segment_ids"]

    all_window_y_true = []
    all_window_y_pred = []
    all_window_probs = []
    all_window_segment_ids = []
    all_window_subject_ids = []
    fold_rows = []

    unique_subjects = np.unique(subject_ids_all)
    global_start = time.time()

    logo = LeaveOneGroupOut()

    for outer_fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(X_all, y_all, groups=subject_ids_all),
        start=1,
    ):
        test_subject = str(np.unique(subject_ids_all[test_idx])[0])

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        subj_train = subject_ids_all[train_idx]

        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        subj_test = subject_ids_all[test_idx]
        seg_test = segment_ids_all[test_idx]

        n_train_groups = len(np.unique(subj_train))
        inner_splits = min(inner_n_splits, n_train_groups)

        if inner_splits < 2:
            print(f"[SKIP FOLD] test_subject={test_subject} has < 2 training subject groups")
            continue

        pipeline, param_grid = get_classical_pipeline_and_param_grid(
            model_type=model_type,
            evaluation_type="cross_subject",
        )

        inner_cv = safe_group_cv_splitter(
            n_splits=inner_splits,
            y=y_train,
            groups=subj_train,
            random_state=1000 + outer_fold_idx,
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=1,
            refit=True,
            verbose=0,
        )

        fold_start = time.time()
        grid.fit(X_train, y_train, groups=subj_train)

        y_pred, probs = get_window_level_outputs_from_model(grid.best_estimator_, X_test)
        fold_elapsed = time.time() - fold_start

        fold_window_metrics = compute_metrics_dict(y_test, y_pred)

        seg_out = aggregate_segment_predictions_from_probs(
            segment_ids=seg_test,
            y_true=y_test,
            probs=probs,
        )
        fold_segment_metrics = seg_out["segment_metrics"]

        all_window_y_true.append(y_test)
        all_window_y_pred.append(y_pred)
        all_window_probs.append(probs)
        all_window_segment_ids.append(seg_test)
        all_window_subject_ids.append(subj_test)

        fold_rows.append(
            {
                "test_subject": str(test_subject),
                "outer_fold": int(outer_fold_idx),
                "window_balanced_accuracy": float(fold_window_metrics["balanced_accuracy"]),
                "window_accuracy": float(fold_window_metrics["accuracy"]),
                "window_macro_f1": float(fold_window_metrics["macro_f1"]),
                "segment_balanced_accuracy": float(fold_segment_metrics["balanced_accuracy"]),
                "segment_accuracy": float(fold_segment_metrics["accuracy"]),
                "segment_macro_f1": float(fold_segment_metrics["macro_f1"]),
                "n_train_windows": int(len(train_idx)),
                "n_test_windows": int(len(test_idx)),
                "n_train_subjects": int(len(np.unique(subj_train))),
                "n_test_subjects": int(len(np.unique(subj_test))),
                "n_test_segments": int(len(np.unique(seg_test))),
                "best_params": grid.best_params_,
                "fold_time_sec": round(fold_elapsed, 3),
            }
        )

    if len(all_window_y_true) == 0:
        raise RuntimeError(f"No successful folds were run for dataset={dataset_name}, model_type={model_type}")

    y_true_all = np.concatenate(all_window_y_true).astype(int)
    y_pred_all = np.concatenate(all_window_y_pred).astype(int)
    probs_all = np.concatenate(all_window_probs)
    segment_ids_eval = np.concatenate(all_window_segment_ids)
    subject_ids_eval = np.concatenate(all_window_subject_ids)

    window_metrics = compute_metrics_dict(y_true_all, y_pred_all)

    segment_out = aggregate_segment_predictions_from_probs(
        segment_ids=segment_ids_eval,
        y_true=y_true_all,
        probs=probs_all,
    )
    segment_metrics = segment_out["segment_metrics"]
    cm_segment = segment_out["segment_confusion_matrix"]

    total_elapsed = time.time() - global_start
    chosen_params = choose_representative_params(fold_rows)

    best_final_model = None
    final_model_status = "not_fitted"

    if fit_final_model:
        try:
            model_fit_start = time.time()
            best_final_model = fit_final_classical_model_no_search(
                model_type=model_type,
                X=X_all,
                y=y_all,
                chosen_params=chosen_params,
                evaluation_type="cross_subject",
            )
            final_model_status = f"fitted_without_grid_search in {time.time() - model_fit_start:.2f} sec"
        except Exception as e:
            print("[WARN] Could not fit final model without grid search:", e)
            final_model_status = f"fit_failed: {e}"

    history = {"fold_rows": fold_rows, "n_successful_folds": len(fold_rows)}

    notes = {
        "evaluation_type": "cross_subject",
        "dataset_name": dataset_name,
        "model_type": model_type,
        "scoring": scoring,
        "n_subjects_total": int(len(unique_subjects)),
        "train_time_sec": round(total_elapsed, 3),
        "final_model_status": final_model_status,
        "report_metric_level": "segment",
        "aggregation_method": "mean_probability_per_segment",
    }

    config = {
        "dataset_name": dataset_name,
        "model_type": model_type,
        "input_type": "bandpower",
        "evaluation_type": "cross_subject",
        "cv_type": "LeaveOneGroupOut_outer + StratifiedGroupKFold_or_GroupKFold_inner",
        "group_variable_outer": "subject_id",
        "group_variable_inner": "subject_id",
        "selection_metric": "balanced_accuracy",
        "inner_n_splits": inner_n_splits,
        "scoring": scoring,
        "feature_dim": int(X_all.shape[1]),
        "n_samples_total": int(X_all.shape[0]),
        "final_model_strategy": "most_frequent_fold_best_params_no_full_grid_search",
        "chosen_params": chosen_params,
        "probability_output": True,
        "aggregation_method": "mean_probability_per_segment",
        "report_metric_level": "segment",
    }

    metrics = {
        "balanced_accuracy": float(window_metrics["balanced_accuracy"]),
        "accuracy": float(window_metrics["accuracy"]),
        "macro_f1": float(window_metrics["macro_f1"]),
        "segment_balanced_accuracy": float(segment_metrics["balanced_accuracy"]),
        "segment_accuracy": float(segment_metrics["accuracy"]),
        "segment_macro_f1": float(segment_metrics["macro_f1"]),
        "report_balanced_accuracy": float(segment_metrics["balanced_accuracy"]),
        "report_accuracy": float(segment_metrics["accuracy"]),
        "report_macro_f1": float(segment_metrics["macro_f1"]),
        "n_train": int(X_all.shape[0]),
        "n_test": int(len(y_true_all)),
        "n_test_segments": int(len(segment_out["segment_y_true"])),
        "n_test_subjects": int(len(np.unique(subject_ids_eval))),
        "best_params": chosen_params,
        "train_time_sec": round(total_elapsed, 3),
    }

    return {
        "config": config,
        "metrics": metrics,
        "notes": notes,
        "history": history,
        "confusion_matrix_array": cm_segment,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "sklearn_model": best_final_model,
        "extra_prediction_arrays": {
            "class_labels": np.array(CLASS_LABELS_NAME, dtype=object),
            "probs": probs_all,
            "segment_ids": segment_ids_eval,
            "subject_ids": subject_ids_eval,
            "segment_y_true": segment_out["segment_y_true"],
            "segment_y_pred": segment_out["segment_y_pred"],
            "segment_probs": segment_out["segment_probs"],
            "segment_ids_unique": segment_out["segment_ids_unique"],
        },
    }


# ------------------------------------------------------------
# Deep experiments
# ------------------------------------------------------------
def run_deep_experiment_by_name(experiment_name: str):
    raise NotImplementedError(
        "Deep experiment reruns are not finished in the repo yet. "
        "Saved EEGNet outputs are included under results/experiments/, "
        "but full retraining still requires the raw-window EEG pipeline "
        "to be extracted from the notebook in a memory-safe way."
    )