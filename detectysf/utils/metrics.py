"""Evaluation metrics without sklearn dependency."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _precision_recall_f1_binary(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int) -> Dict[str, float]:
    tp = float(np.logical_and(y_true == positive_label, y_pred == positive_label).sum())
    fp = float(np.logical_and(y_true != positive_label, y_pred == positive_label).sum())
    fn = float(np.logical_and(y_true == positive_label, y_pred != positive_label).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float((y_true == y_pred).mean())
    m0 = _precision_recall_f1_binary(y_true, y_pred, positive_label=0)
    m1 = _precision_recall_f1_binary(y_true, y_pred, positive_label=1)
    macro_precision = (m0["precision"] + m1["precision"]) / 2.0
    macro_recall = (m0["recall"] + m1["recall"]) / 2.0
    macro_f1 = (m0["f1"] + m1["f1"]) / 2.0
    return {
        "accuracy": acc,
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
    }

