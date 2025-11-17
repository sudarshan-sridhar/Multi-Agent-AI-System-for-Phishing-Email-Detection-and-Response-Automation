"""
Dataset utilities for the Multi-Phishing LangChain backend.

This module exposes:
- High-level dataset summary (`dataset_summary`), including label distribution,
  text length statistics, and imbalance metrics.
- Feature-level statistics (`dataset_feature_stats`), including word/URL counts
  and TF-IDF–based token importance per class.

It operates on the processed dataset:
    <project_root>/data/processed/combined.jsonl

and is intended to power:
- Analytics / EDA views in the UI.
- Offline sanity checks on training data quality.
- Interpretability panels for the phishing classifier.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from .email_utils import extract_urls
from .model_loader import load_model


# ---------------------------------------------------------------------------
# File Location / Base Paths
# ---------------------------------------------------------------------------
# This module provides utilities for analyzing and summarizing the processed
# dataset used to train and evaluate the phishing classifier.
#
# It reads from:
#   phish-lc/data/processed/combined.jsonl
#
# parents[0] = core/, parents[1] = backend/, parents[2] = project root.
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "combined.jsonl"


def _load_df(max_samples: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Load the processed dataset as a DataFrame.

    Responsibilities:
    - Validate that the processed file exists.
    - Enforce presence of core fields required by the model pipeline.
    - Optionally downsample to a maximum number of samples (for speed).
    - Return a clean DataFrame with a fresh integer index.

    This function is the foundation for all downstream dataset
    summary and feature-analysis utilities in this module.
    """
    if not PROCESSED_PATH.exists():
        # Fail fast with a clear message if the expected processed file is missing.
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_PATH}")

    # Read line-delimited JSON into a DataFrame.
    df = pd.read_json(PROCESSED_PATH, lines=True)

    # Ensure essential fields exist. These are required for model pipeline logic.
    required_cols = {"subject", "body", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Processed dataset missing columns: {missing}")

    # Optional downsampling to limit cost of analysis on very large datasets.
    if max_samples is not None:
        try:
            max_samples_int = int(max_samples)
        except (TypeError, ValueError):
            # Defensive fallback in case a non-integer slips through.
            max_samples_int = 5000
        if len(df) > max_samples_int:
            df = df.sample(n=max_samples_int, random_state=random_state)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def dataset_summary(
    max_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute a high-level dataset summary, including:

      • Number of samples
      • Label distribution
      • Optional source distribution (if dataset includes such metadata)
      • Text length statistics (avg / median / p95 / min / max)
      • Class imbalance ratio (majority : minority)
      • Phishing ratio

    Returned as a plain dict for easy JSON serialization, powering
    exploratory data analysis and dashboard views.
    """
    df = _load_df(max_samples=max_samples, random_state=random_state)

    n_samples = int(len(df))
    label_counts = df["label"].value_counts().to_dict()

    # Optional "source" metadata column (depends on dataset creation pipeline)
    if "source" in df.columns:
        source_counts = df["source"].value_counts().to_dict()
    else:
        source_counts = {}

    # Build raw text block from subject+body for length analysis
    subjects = df["subject"].fillna("").astype(str)
    bodies = df["body"].fillna("").astype(str)
    texts = (subjects + "\n\n" + bodies).tolist()

    # Compute basic word-count statistics for message length distribution
    lengths: List[int] = []
    for t in texts:
        lengths.append(len(t.split()))

    if lengths:
        arr = np.array(lengths, dtype=float)
        avg_length = float(arr.mean())
        median_length = float(np.median(arr))
        p95_length = float(np.percentile(arr, 95))
        min_length = float(arr.min())
        max_length = float(arr.max())
    else:
        # Edge case: no data (dataset empty)
        avg_length = median_length = p95_length = min_length = max_length = 0.0

    # Compute imbalance ratio: majority_count / minority_count
    if label_counts:
        counts = list(label_counts.values())
        if len(counts) >= 2:
            majority = max(counts)
            minority = min(counts)
            if minority > 0:
                imbalance_ratio_raw = float(majority / minority)
            else:
                # If minority is zero, imbalance is infinite (but we record separately)
                imbalance_ratio_raw = float("inf")
        else:
            # Single-class dataset: define ratio as 1.0 to avoid division by zero.
            imbalance_ratio_raw = 1.0
    else:
        # No labels at all; treat as neutral.
        imbalance_ratio_raw = 1.0

    # Normalize ∞ → finite+flag for JSON safety
    is_inf = not np.isfinite(imbalance_ratio_raw)
    imbalance_ratio = float(imbalance_ratio_raw) if not is_inf else 0.0

    phishing_count = int(label_counts.get("phishing", 0))
    phishing_ratio = float(phishing_count / n_samples) if n_samples > 0 else 0.0

    return {
        "n_samples": n_samples,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "avg_length": avg_length,
        "median_length": median_length,
        "p95_length": p95_length,
        "min_length": min_length,
        "max_length": max_length,
        "imbalance_ratio": imbalance_ratio,          # always finite for JSON
        "imbalance_infinite": bool(is_inf),          # separate flag for ∞
        "phishing_ratio": phishing_ratio,
    }


# ---------------------------------------------------------------------------
# Token & feature distribution
# ---------------------------------------------------------------------------

def _word_and_url_counts(texts: List[str]) -> Tuple[List[int], List[int]]:
    """
    Compute simple per-sample statistics:
      - word count (token count)
      - URL count (via extract_urls)

    These low-level features are used for dataset feature summaries,
    and provide a lightweight view on message complexity and URL density.
    """
    word_counts: List[int] = []
    url_counts: List[int] = []

    for t in texts:
        tokens = t.split()
        word_counts.append(len(tokens))

        urls = extract_urls(t)
        url_counts.append(len(urls))

    return word_counts, url_counts


def dataset_feature_stats(
    max_samples: Optional[int] = 5000,
    random_state: int = 42,
    top_k: int = 30,
) -> Dict[str, Any]:
    """
    Compute distribution and token-level interpretability metrics:

      • word counts per email
      • URL counts per email
      • vocabulary size (TF-IDF)
      • top phishing-indicative tokens (highest mean TF-IDF difference)
      • top benign-indicative tokens

    `top_k` controls how many of the strongest tokens (by class-difference score)
    are returned for each class.

    This provides insight into what features most strongly influence
    the classifier's decision boundary and is suitable for debugging or UI charts.
    """
    df = _load_df(max_samples=max_samples, random_state=random_state)
    n_samples = int(len(df))

    subjects = df["subject"].fillna("").astype(str)
    bodies = df["body"].fillna("").astype(str)
    texts = (subjects + "\n\n" + bodies).tolist()
    labels = df["label"].astype(str).tolist()

    # Compute per-email counts for descriptive statistics
    word_counts, url_counts = _word_and_url_counts(texts)

    # Load TF-IDF vectorizer + classifier artefacts
    # Note: `clf` is loaded for completeness; current logic only uses the vectorizer.
    artefact = load_model()
    vectorizer = artefact["vectorizer"]
    clf = artefact["classifier"]

    # Transform raw text → TF-IDF feature matrix
    X = vectorizer.transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    vocab_size = int(len(feature_names))

    labels_arr = np.array(labels, dtype=str)
    phishing_mask = labels_arr == "phishing"
    benign_mask = labels_arr == "benign"

    # Compute class-conditional mean TF-IDF values
    if phishing_mask.sum() > 0:
        X_phish = X[phishing_mask]
        mean_phish = np.asarray(X_phish.mean(axis=0)).ravel()
    else:
        # No phishing samples in subset; fall back to zeros.
        mean_phish = np.zeros(X.shape[1], dtype=float)

    if benign_mask.sum() > 0:
        X_benign = X[benign_mask]
        mean_benign = np.asarray(X_benign.mean(axis=0)).ravel()
    else:
        # No benign samples in subset; fall back to zeros.
        mean_benign = np.zeros(X.shape[1], dtype=float)

    # Score is simple difference: positive → phishing-leaning token
    scores = mean_phish - mean_benign

    # Bound top_k to [1, 200] to avoid excessive payloads
    top_k = max(1, min(int(top_k), 200))
    top_phish_indices = np.argsort(scores)[::-1][:top_k]   # largest (phishing)
    top_benign_indices = np.argsort(scores)[:top_k]        # most negative (benign)

    def _build_token_list(indices: np.ndarray) -> List[Dict[str, Any]]:
        """
        Build a list of {token, score} for a given set of TF-IDF indices.

        `score` is defined as mean_phish - mean_benign, so:
        - positive score → token more associated with phishing.
        - negative score → token more associated with benign emails.
        """
        items: List[Dict[str, Any]] = []
        for idx in indices:
            idx_int = int(idx)
            token = str(feature_names[idx_int])
            score = float(scores[idx_int])
            items.append({"token": token, "score": score})
        return items

    top_phishing_tokens = _build_token_list(top_phish_indices)
    top_benign_tokens = _build_token_list(top_benign_indices)

    # Convert numpy → native Python for JSON serialization
    word_counts_py = [int(x) for x in word_counts]
    url_counts_py = [int(x) for x in url_counts]

    return {
        "n_samples": n_samples,
        "word_counts": word_counts_py,
        "url_counts": url_counts_py,
        "vocab_size": vocab_size,
        "top_phishing_tokens": top_phishing_tokens,
        "top_benign_tokens": top_benign_tokens,
    }
