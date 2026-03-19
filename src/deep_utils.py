from __future__ import annotations

import copy
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from src.config import RANDOM_SEED


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS_NAME = ["REST", "IG", "IE"]


def set_global_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EEGWindowH5Dataset(Dataset):
    def __init__(self, h5_path: str | Path, indices, normalize_per_window=True, eps=1e-6):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalize_per_window = normalize_per_window
        self.eps = eps
        self._h5 = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = int(self.indices[idx])

        x = self._h5["X_windows"][real_idx].astype(np.float32)
        y = int(self._h5["y"][real_idx])

        if self.normalize_per_window:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mean) / np.maximum(std, self.eps)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def make_class_weights(y: np.ndarray, n_classes: int = 3):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_h5_loader(h5_path, indices, batch_size=64, shuffle=False, num_workers=0, pin_memory=False):
    ds = EEGWindowH5Dataset(h5_path=h5_path, indices=indices)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def compute_metrics_dict(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def compute_confusion_matrix_fixed(y_true, y_pred, labels=None):
    if labels is None:
        labels = [0, 1, 2]
    return confusion_matrix(y_true, y_pred, labels=labels)


def aggregate_segment_predictions(segment_ids, y_true, probs):
    segment_ids = np.asarray(segment_ids)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    unique_segments = np.unique(segment_ids)

    seg_true = []
    seg_pred = []
    seg_probs_all = []

    for seg in unique_segments:
        mask = segment_ids == seg
        seg_probs = probs[mask].mean(axis=0)
        seg_pred_label = int(np.argmax(seg_probs))

        vals, counts = np.unique(y_true[mask], return_counts=True)
        seg_true_label = int(vals[np.argmax(counts)])

        seg_true.append(seg_true_label)
        seg_pred.append(seg_pred_label)
        seg_probs_all.append(seg_probs)

    seg_true = np.asarray(seg_true)
    seg_pred = np.asarray(seg_pred)
    seg_probs_all = np.asarray(seg_probs_all)

    return {
        "segment_ids_unique": unique_segments,
        "segment_y_true": seg_true,
        "segment_y_pred": seg_pred,
        "segment_probs": seg_probs_all,
        "segment_metrics": compute_metrics_dict(seg_true, seg_pred),
        "segment_confusion_matrix": compute_confusion_matrix_fixed(seg_true, seg_pred),
    }


def run_train_epoch(model, loader, optimizer, criterion, device=DEVICE, grad_clip_norm=1.0):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        bs = X.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

    return float(running_loss / max(n_samples, 1))


@torch.no_grad()
def run_eval_epoch(model, loader, criterion, device=DEVICE):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    all_y_true = []
    all_y_pred = []
    all_probs = []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        bs = X.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        all_y_true.append(y.cpu().numpy())
        all_y_pred.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    probs = np.concatenate(all_probs)

    return {
        "loss": float(running_loss / max(n_samples, 1)),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "metrics": compute_metrics_dict(y_true, y_pred),
    }