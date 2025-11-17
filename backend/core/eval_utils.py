"""
Evaluation utilities for the phishing classifier in the Multi-Phishing LangChain backend.

This module focuses on *offline* evaluation and analysis of the saved classifier:
- It operates on a preprocessed combined dataset: <project_root>/data/processed/combined.jsonl.
- It uses the persisted model artefact (vectorizer + classifier) loaded via `load_model`.

Provided capabilities:
    • `evaluate_model_on_processed`:
        - End-to-end evaluation on the processed dataset with a fixed threshold (0.5).
        - Returns classification report and confusion matrix.
    • `sweep_thresholds`:
        - Explores performance across multiple decision thresholds.
    • `roc_curve_data`:
        - Computes ROC curve and AUC with strong guards against degenerate data.
    • `pr_curve_data`:
        - Computes Precision–Recall curve and AUC with added sanitization.
    • `confusion_at_threshold`:
        - Computes a confusion matrix and core metrics at a specific operating threshold.

All outputs are JSON-serializable and intended to back analytics / dashboards
in the frontend without additional post-processing.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)

from .model_loader import load_model


# ---------------------------------------------------------------------------
# Paths / Dataset Location
# ---------------------------------------------------------------------------
# This module provides evaluation utilities for the phishing classifier.
# It operates on a preprocessed combined dataset stored as JSONL and
# on a model artefact loaded via `load_model`.
#
# File location: phish-lc/backend/core/eval_utils.py
# parents[0] = core, [1] = backend, [2] = project root (phish-lc)
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "combined.jsonl"


# ---------------------------------------------------------------------------
# Helper: load processed dataset + run classifier to get p_phishing
# ---------------------------------------------------------------------------

def _load_processed_dataset(
    max_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    """
    Load the processed dataset and return:
      - y_true_labels: list of 'benign' / 'phishing' labels
      - p_phishing: numpy array of shape (n_samples,)
      - label_counts: dict of label -> count

    This function:
      * Reads the combined JSONL file.
      * Optionally subsamples for faster evaluation.
      * Rebuilds the feature matrix using the stored vectorizer.
      * Runs the classifier to obtain phishing probabilities.

    It acts as the common foundation for all evaluation routines so that
    dataset sampling, feature generation, and model inference are aligned.
    """
    if not PROCESSED_PATH.exists():
        # Fail fast with a clear error if the evaluation dataset is missing.
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_PATH}")

    df = pd.read_json(PROCESSED_PATH, lines=True)

    # Ensure that required columns exist in the processed data.
    required_cols = {"subject", "body", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Processed dataset missing columns: {missing}")

    # Optional sampling for speed when dataset is large.
    if max_samples is not None:
        try:
            max_samples_int = int(max_samples)
        except (TypeError, ValueError):
            # Conservative fallback if a non-numeric value slips in.
            max_samples_int = 5000
        if len(df) > max_samples_int:
            df = df.sample(n=max_samples_int, random_state=random_state)

    # Build raw text input by concatenating subject and body.
    subjects = df["subject"].fillna("").astype(str)
    bodies = df["body"].fillna("").astype(str)
    texts = (subjects + "\n\n" + bodies).tolist()

    # Ground-truth labels and label distribution.
    y_true_labels = df["label"].astype(str).tolist()
    label_counts = df["label"].value_counts().to_dict()

    # Load trained artefacts: vectorizer + classifier.
    artefact = load_model()
    vectorizer = artefact["vectorizer"]
    clf = artefact["classifier"]

    # Transform text into features and get predicted probabilities.
    X = vectorizer.transform(texts)
    y_proba = clf.predict_proba(X)

    # Resolve which probability column corresponds to "phishing".
    classes_ = [str(c) for c in getattr(clf, "classes_", [])]
    if "phishing" in classes_:
        phishing_index = classes_.index("phishing")
        p_phishing = y_proba[:, phishing_index]
    else:
        # Fallback: assume second column is phishing if not explicitly labeled.
        p_phishing = y_proba[:, -1]

    return y_true_labels, p_phishing, label_counts


# ---------------------------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------------------------

def evaluate_model_on_processed(
    max_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate the saved classifier on the processed combined dataset.

    Returns a dictionary with:
      - base_stats: sample counts, label distribution, mean p_phishing
      - report: sklearn classification_report as a dict
      - confusion_matrix: confusion matrix with label ordering

    This is designed for:
      - quick model validation after training,
      - dashboard views that summarize performance at a 0.5 threshold,
      - regression testing across model versions.
    """
    y_true_labels, p_phishing, label_counts = _load_processed_dataset(
        max_samples=max_samples,
        random_state=random_state,
    )

    # Binary label prediction using default threshold 0.5.
    y_true = y_true_labels
    y_pred_labels = ["phishing" if p >= 0.5 else "benign" for p in p_phishing]

    # Detailed precision/recall/F1 per class via sklearn.
    report = classification_report(
        y_true,
        y_pred_labels,
        output_dict=True,
        labels=["benign", "phishing"],
        zero_division=0,
    )

    # Confusion matrix with a fixed label order for consistency.
    cm = confusion_matrix(
        y_true,
        y_pred_labels,
        labels=["benign", "phishing"],
    )

    # Basic aggregate statistics.
    base_stats = {
        "n_samples": int(len(y_true_labels)),
        "label_counts": label_counts,
        "p_phishing_mean": float(np.mean(p_phishing)),
    }

    confusion = {
        "labels": ["benign", "phishing"],
        "matrix": cm.tolist(),
    }

    return {
        "base_stats": base_stats,
        "report": report,
        "confusion_matrix": confusion,
    }


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    max_samples: Optional[int] = 5000,
    num_thresholds: int = 21,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Sweep decision thresholds over p_phishing and compute precision, recall,
    F1, accuracy, FP rate, and FN rate for 'phishing' as positive class.

    This is useful for:
      - Selecting an operating threshold that balances F1 / accuracy.
      - Visualizing trade-offs (e.g., ROC-like threshold plots).
      - Comparing business-impact trade-offs for different cutoffs.
    """
    y_true_labels, p_phishing, label_counts = _load_processed_dataset(
        max_samples=max_samples,
        random_state=random_state,
    )

    # Convert textual labels into binary 0/1 with phishing as positive.
    y_true_bin = np.array([1 if y == "phishing" else 0 for y in y_true_labels], dtype=int)
    n = len(y_true_bin)

    if n == 0:
        raise ValueError("No samples found in processed dataset for threshold sweep.")

    # Ensure at least 2 thresholds (0.0 and 1.0).
    if num_thresholds < 2:
        num_thresholds = 2
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    precision_list: List[float] = []
    recall_list: List[float] = []
    f1_list: List[float] = []
    acc_list: List[float] = []
    fp_rate_list: List[float] = []
    fn_rate_list: List[float] = []

    # Evaluate metrics at each threshold.
    for t in thresholds:
        y_pred_bin = (p_phishing >= t).astype(int)

        tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
        tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
        fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
        fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        precision_list.append(float(precision))
        recall_list.append(float(recall))
        f1_list.append(float(f1))
        acc_list.append(float(accuracy))
        fp_rate_list.append(float(fp_rate))
        fn_rate_list.append(float(fn_rate))

    # Select the threshold that maximizes F1 score.
    best_idx = int(np.argmax(f1_list))
    best_threshold = float(thresholds[best_idx])

    best_metrics = {
        "threshold": best_threshold,
        "precision": precision_list[best_idx],
        "recall": recall_list[best_idx],
        "f1": f1_list[best_idx],
        "accuracy": acc_list[best_idx],
        "fp_rate": fp_rate_list[best_idx],
        "fn_rate": fn_rate_list[best_idx],
    }

    return {
        "label_positive": "phishing",
        "n_samples": int(n),
        "label_counts": label_counts,
        "thresholds": thresholds.tolist(),
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "accuracy": acc_list,
        "fp_rate": fp_rate_list,
        "fn_rate": fn_rate_list,
        "best_by_f1": best_metrics,
    }


# ---------------------------------------------------------------------------
# ROC curve (with NaN/inf sanitization + hard guarding)
# ---------------------------------------------------------------------------

def roc_curve_data(
    max_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute ROC curve for phishing as positive class.

    Any NaN/inf values are sanitized so the JSON is always valid.
    On error, a diagonal fallback curve is returned with an 'error' field.

    This function is intentionally defensive so that the UI can render
    meaningful charts even when the input data is degenerate (e.g., only
    one class present).
    """
    try:
        y_true_labels, p_phishing, label_counts = _load_processed_dataset(
            max_samples=max_samples,
            random_state=random_state,
        )
        y_true_bin = np.array(
            [1 if y == "phishing" else 0 for y in y_true_labels], dtype=int
        )
        n = len(y_true_bin)

        if n == 0:
            raise ValueError("No samples in processed dataset.")

        unique_classes = np.unique(y_true_bin)
        if unique_classes.size < 2:
            # Degenerate case: only one class present → fallback diagonal ROC.
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thresholds = np.array([1.0, 0.0])
            auc_val = 0.5
        else:
            # Standard ROC computation with sklearn.
            fpr, tpr, thresholds = roc_curve(y_true_bin, p_phishing)

            # Sanitize to ensure finite, JSON-safe values.
            fpr = np.nan_to_num(fpr, nan=0.0, posinf=1.0, neginf=0.0)
            tpr = np.nan_to_num(tpr, nan=0.0, posinf=1.0, neginf=0.0)
            thresholds = np.nan_to_num(thresholds, nan=0.5, posinf=1.0, neginf=0.0)

            auc_val = auc(fpr, tpr)
            if not np.isfinite(auc_val):
                auc_val = 0.5

        return {
            "label_positive": "phishing",
            "n_samples": int(n),
            "label_counts": label_counts,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(auc_val),
        }

    except Exception as e:
        # Log error to stdout but still return a safe fallback structure.
        print(f"[eval_utils] ROC error: {e}")
        return {
            "label_positive": "phishing",
            "n_samples": 0,
            "label_counts": {},
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0],
            "thresholds": [1.0, 0.0],
            "auc": 0.5,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Precision–Recall curve (with NaN/inf sanitization + hard guarding)
# ---------------------------------------------------------------------------

def pr_curve_data(
    max_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute Precision-Recall curve for phishing as positive class.

    Any NaN/inf values are sanitized so JSON encoding never fails.
    On error, a flat fallback curve is returned with an 'error' field.

    PR curves are particularly useful when dealing with imbalanced datasets,
    such as phishing corpora where the positive class is relatively rare.
    """
    try:
        y_true_labels, p_phishing, label_counts = _load_processed_dataset(
            max_samples=max_samples,
            random_state=random_state,
        )
        y_true_bin = np.array(
            [1 if y == "phishing" else 0 for y in y_true_labels], dtype=int
        )
        n = len(y_true_bin)

        if n == 0:
            raise ValueError("No samples in processed dataset.")

        unique_classes = np.unique(y_true_bin)
        if unique_classes.size < 2:
            # Degenerate case: only one class present.
            positive_rate = float(y_true_bin.mean())
            precision = np.array([positive_rate, positive_rate])
            recall = np.array([0.0, 1.0])
            thresholds = np.array([0.5])
            auc_pr_val = 0.0
        else:
            # Standard precision–recall computation.
            precision, recall, thresholds = precision_recall_curve(
                y_true_bin, p_phishing
            )

            # Sanitize to keep values finite.
            precision = np.nan_to_num(precision, nan=0.0, posinf=1.0, neginf=0.0)
            recall = np.nan_to_num(recall, nan=0.0, posinf=1.0, neginf=0.0)
            thresholds = np.nan_to_num(thresholds, nan=0.5, posinf=1.0, neginf=0.0)

            auc_pr_val = auc(recall, precision)
            if not np.isfinite(auc_pr_val):
                auc_pr_val = 0.0

        return {
            "label_positive": "phishing",
            "n_samples": int(n),
            "label_counts": label_counts,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "auc_pr": float(auc_pr_val),
        }

    except Exception as e:
        # Log error and return safe default PR curve.
        print(f"[eval_utils] PR error: {e}")
        return {
            "label_positive": "phishing",
            "n_samples": 0,
            "label_counts": {},
            "precision": [1.0, 1.0],
            "recall": [0.0, 1.0],
            "thresholds": [0.5],
            "auc_pr": 0.0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Confusion matrix at a specific threshold
# ---------------------------------------------------------------------------

def confusion_at_threshold(
    threshold: float,
    max_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute confusion matrix and basic metrics at a given decision threshold
    on p_phishing.

    This is helpful to:
      - Inspect operational performance at a chosen cutoff.
      - Report metrics in dashboards or API responses.
      - Align SOC teams on expected false positive / false negative behavior.
    """
    y_true_labels, p_phishing, label_counts = _load_processed_dataset(
        max_samples=max_samples,
        random_state=random_state,
    )
    y_true_bin = np.array([1 if y == "phishing" else 0 for y in y_true_labels], dtype=int)
    n = len(y_true_bin)

    if n == 0:
        # Return a fully defined but empty structure if no data is available.
        return {
            "label_positive": "phishing",
            "threshold": float(threshold),
            "n_samples": 0,
            "label_counts": {},
            "labels": ["benign", "phishing"],
            "matrix": [[0, 0], [0, 0]],
            "metrics": {
                "precision_phishing": 0.0,
                "recall_phishing": 0.0,
                "f1_phishing": 0.0,
                "accuracy": 0.0,
                "fp_rate": 0.0,
                "fn_rate": 0.0,
            },
        }

    # Apply the given threshold to convert probabilities into binary predictions.
    y_pred_bin = (p_phishing >= threshold).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision_val + recall_val > 0:
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
    else:
        f1_val = 0.0
    accuracy_val = (tp + tn) / n if n > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Confusion matrix layout:
    #   row 0: true benign
    #   row 1: true phishing
    matrix = [
        [tn, fp],  # true benign row
        [fn, tp],  # true phishing row
    ]

    metrics = {
        "precision_phishing": float(precision_val),
        "recall_phishing": float(recall_val),
        "f1_phishing": float(f1_val),
        "accuracy": float(accuracy_val),
        "fp_rate": float(fp_rate),
        "fn_rate": float(fn_rate),
    }

    return {
        "label_positive": "phishing",
        "threshold": float(threshold),
        "n_samples": int(n),
        "label_counts": label_counts,
        "labels": ["benign", "phishing"],
        "matrix": matrix,
        "metrics": metrics,
    }
